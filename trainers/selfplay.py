"""
trainers/selfplay.py -- Historical self-play via ELO for **Hide & Seek 2.0**.

# historical self-play via ELO

OpenAI's 2019 emergent-tool-use result hinged on a self-play autocurriculum:
agents kept facing progressively stronger versions of themselves, which is what
drove the ramp-use / box-locking / box-surfing arms race. We reproduce that here
with a fixed-size pool of **frozen parameter snapshots**, each carrying an ELO
rating, and we sample opponents weighted by ELO so the learner spends most of its
time against credible (not trivially weak) past selves.

Design constraints (CONTRACT §0, §8)
------------------------------------
* The pool is a **pure jnp ring buffer**: a pytree of stacked parameters plus
  parallel arrays for ELO ratings, validity flags, and the insertion cursor.
  Keeping it as a registered pytree means the whole self-play machinery can live
  inside ``jax.jit`` / ``jax.lax.scan`` alongside the rollout -- no host
  transfer to pick or push an opponent.
* All "control flow" is functional: ``push_snapshot`` overwrites the ring slot at
  the cursor with ``jax.tree_util.tree_map`` + dynamic indexing; ``sample_opponent``
  draws a slot index from an ELO-softmax over the *valid* slots with
  ``jax.random.categorical``. No Python branching on traced values.

The trainer (``mappo.py``) uses this to, with probability ``past_opponent_prob``,
swap the *opponent team's* effective parameters for a fraction of the vectorized
envs to a sampled historical snapshot.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import flax

# utils.elo is built in parallel; import defensively so this module stays
# importable standalone. The fallbacks mirror the documented signatures exactly
# (`expected_score`, `update_elo`) so behaviour is identical if the real module
# is unavailable at import time.
try:  # pragma: no cover
    from utils.elo import expected_score as _expected_score
    from utils.elo import update_elo as _update_elo
except Exception:  # pragma: no cover

    def _expected_score(rating_a: jnp.ndarray, rating_b: jnp.ndarray) -> jnp.ndarray:
        """Logistic expected score of A vs B (standard ELO, base-10, 400 scale)."""
        return 1.0 / (1.0 + jnp.power(10.0, (rating_b - rating_a) / 400.0))

    def _update_elo(
        rating_a: jnp.ndarray,
        rating_b: jnp.ndarray,
        score_a: jnp.ndarray,
        k: float,
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Symmetric ELO update; ``score_a`` in {1=win, 0.5=draw, 0=loss}."""
        exp_a = _expected_score(rating_a, rating_b)
        exp_b = 1.0 - exp_a
        new_a = rating_a + k * (score_a - exp_a)
        new_b = rating_b + k * ((1.0 - score_a) - exp_b)
        return new_a, new_b


PyTree = Any


# ===========================================================================
# OpponentPool -- fixed-size frozen-snapshot ring buffer with ELO
# ===========================================================================
@flax.struct.dataclass
class OpponentPool:
    """Fixed-size ring buffer of frozen policy snapshots + ELO ratings.

    Stored as a registered pytree so it threads cleanly through ``jit``/``scan``.
    The leading axis of every leaf in :attr:`params` (and of :attr:`ratings`,
    :attr:`valid`) is the pool capacity ``opponent_pool_size``.

    Attributes
    ----------
    params:
        A pytree with the *same structure* as a single actor's parameters, but
        with an extra leading ``capacity`` axis -- i.e. ``capacity`` stacked
        frozen snapshots. ``jax.tree_util.tree_map(lambda x: x[i], pool.params)``
        recovers snapshot ``i``.
    ratings:
        ``(capacity,)`` float32 -- ELO rating of each slot.
    valid:
        ``(capacity,)`` bool -- whether a slot has been populated yet (the ring
        starts empty; sampling restricts to valid slots).
    cursor:
        ``()`` int32 -- next write position (mod capacity).
    capacity:
        ``()`` int32 -- pool size (kept as data for jit-friendliness).
    """

    params: PyTree
    ratings: jnp.ndarray
    valid: jnp.ndarray
    cursor: jnp.ndarray
    capacity: jnp.ndarray


def init_opponent_pool(
    template_params: PyTree, capacity: int, elo_init: float
) -> OpponentPool:
    """Allocate an empty :class:`OpponentPool` shaped after ``template_params``.

    Parameters
    ----------
    template_params:
        One example of the actor parameter pytree (e.g. from ``init_actor``).
        Each leaf is tiled ``capacity`` times along a new leading axis to form
        the snapshot store. The initial contents are zeros (and are never read
        until overwritten, because ``valid`` starts all-False).
    capacity:
        Pool size (``cfg.train.opponent_pool_size``).
    elo_init:
        Initial ELO for freshly pushed snapshots (``cfg.train.elo_init``).

    Returns
    -------
    OpponentPool
        An empty pool (``valid`` all False, ``cursor`` 0).
    """
    # Pre-allocate the stacked snapshot store: zeros_like with a leading capacity
    # axis. This fixes the pytree's shapes so push/sample never re-trace.
    stacked = jax.tree_util.tree_map(
        lambda x: jnp.zeros((capacity,) + x.shape, dtype=x.dtype), template_params
    )
    return OpponentPool(
        params=stacked,
        ratings=jnp.full((capacity,), elo_init, dtype=jnp.float32),
        valid=jnp.zeros((capacity,), dtype=bool),
        cursor=jnp.asarray(0, dtype=jnp.int32),
        capacity=jnp.asarray(capacity, dtype=jnp.int32),
    )


@jax.jit
def push_snapshot(
    pool: OpponentPool, params: PyTree, rating: jnp.ndarray
) -> OpponentPool:
    """Insert a frozen snapshot at the ring cursor, advancing it (mod capacity).

    Pure-functional ring write: every leaf of ``params`` is scattered into the
    matching leaf of ``pool.params`` at index ``cursor`` via
    :func:`jax.lax.dynamic_update_index_in_dim`. No host transfer, jit-safe.

    Parameters
    ----------
    pool:
        Current pool.
    params:
        The actor parameter pytree to freeze (structure must match the pool's
        per-slot structure).
    rating:
        Scalar ELO to assign the new snapshot (typically the learner's current
        rating, so a fresh snapshot enters at the live skill level).

    Returns
    -------
    OpponentPool
        The pool with slot ``cursor`` overwritten, marked valid, rated, and the
        cursor advanced.
    """
    idx = pool.cursor

    new_params = jax.tree_util.tree_map(
        lambda store, leaf: jax.lax.dynamic_update_index_in_dim(
            store, leaf, idx, axis=0
        ),
        pool.params,
        params,
    )
    new_ratings = pool.ratings.at[idx].set(jnp.asarray(rating, jnp.float32))
    new_valid = pool.valid.at[idx].set(True)
    new_cursor = (idx + 1) % pool.capacity

    return pool.replace(
        params=new_params,
        ratings=new_ratings,
        valid=new_valid,
        cursor=new_cursor,
    )


@jax.jit
def sample_opponent(
    key: jnp.ndarray, pool: OpponentPool
) -> Tuple[jnp.ndarray, PyTree, jnp.ndarray]:
    """Sample a snapshot index weighted by ELO over the *valid* slots.

    We softmax the ratings (scaled like standard ELO, /400) so stronger past
    selves are drawn more often, but every valid slot keeps non-zero mass to
    preserve curriculum diversity. Invalid slots get ``-inf`` logits and so are
    never selected. If the pool is entirely empty we fall back to slot 0 (the
    caller is expected to gate on ``pool.valid.any()`` before *using* the
    result; see :func:`pool_is_empty`).

    Parameters
    ----------
    key:
        PRNGKey for the categorical draw.
    pool:
        The opponent pool.

    Returns
    -------
    (idx, params, rating):
        ``idx`` -- the chosen slot index ``()`` int32;
        ``params`` -- that slot's frozen parameter pytree (extracted with
        dynamic indexing, so it stays jit-friendly);
        ``rating`` -- the slot's ELO ``()`` float32.
    """
    # ELO-softmax logits; mask invalid slots to -inf so they get zero prob.
    logits = pool.ratings / 400.0
    masked_logits = jnp.where(pool.valid, logits, -jnp.inf)
    # Guard the all-invalid case: if no slot is valid, every logit is -inf and
    # categorical would be ill-defined; replace with uniform-over-slot-0.
    any_valid = jnp.any(pool.valid)
    safe_logits = jnp.where(
        any_valid,
        masked_logits,
        jnp.zeros_like(masked_logits).at[0].set(0.0),
    )
    idx = jax.random.categorical(key, safe_logits).astype(jnp.int32)

    params = jax.tree_util.tree_map(
        lambda store: jax.lax.dynamic_index_in_dim(
            store, idx, axis=0, keepdims=False
        ),
        pool.params,
    )
    rating = pool.ratings[idx]
    return idx, params, rating


@jax.jit
def pool_is_empty(pool: OpponentPool) -> jnp.ndarray:
    """Return a scalar bool: ``True`` iff no slot has been populated yet."""
    return jnp.logical_not(jnp.any(pool.valid))


@jax.jit
def set_rating(pool: OpponentPool, idx: jnp.ndarray, rating: jnp.ndarray) -> OpponentPool:
    """Functionally update the ELO rating of slot ``idx`` (jit-safe scatter)."""
    new_ratings = pool.ratings.at[idx].set(jnp.asarray(rating, jnp.float32))
    return pool.replace(ratings=new_ratings)


# ===========================================================================
# EloManager -- thin stateful wrapper over utils.elo for win/loss bookkeeping
# ===========================================================================
class EloManager:
    """Track the **learner's** live ELO and apply win/loss updates vs opponents.

    This is the one deliberately *non*-jitted convenience object (it holds the
    Python-side ``k`` hyperparameter and the current learner rating between
    updates). The numerical core delegates to ``utils.elo`` so the rating math
    matches the rest of the stack exactly. All array ops remain ``jnp`` so the
    manager can be fed by metrics that came straight off device.

    Parameters
    ----------
    elo_init:
        Starting rating for the live learner (``cfg.train.elo_init``).
    elo_k:
        ELO K-factor / learning rate (``cfg.train.elo_k``).
    """

    def __init__(self, elo_init: float, elo_k: float) -> None:
        self.k: float = float(elo_k)
        self.learner_rating: jnp.ndarray = jnp.asarray(elo_init, dtype=jnp.float32)

    def expected_score(self, opponent_rating: jnp.ndarray) -> jnp.ndarray:
        """Expected score of the learner vs an opponent of ``opponent_rating``."""
        return _expected_score(self.learner_rating, opponent_rating)

    def update(
        self, opponent_rating: jnp.ndarray, learner_score: jnp.ndarray
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Apply one ELO update from a match outcome and store the new rating.

        Parameters
        ----------
        opponent_rating:
            ELO of the opponent the learner just played (scalar).
        learner_score:
            Match result from the learner's perspective: ``1.0`` win, ``0.5``
            draw, ``0.0`` loss (may be a soft win-rate over a batch of envs).

        Returns
        -------
        (new_learner_rating, new_opponent_rating):
            Both updated ratings. The learner's is also stored on ``self`` so the
            next snapshot push enters the pool at the current skill level.
        """
        new_learner, new_opp = _update_elo(
            self.learner_rating,
            jnp.asarray(opponent_rating, jnp.float32),
            jnp.asarray(learner_score, jnp.float32),
            self.k,
        )
        self.learner_rating = new_learner
        return new_learner, new_opp

    def rating(self) -> jnp.ndarray:
        """Return the learner's current rating (scalar jnp array)."""
        return self.learner_rating


@partial(jax.jit, static_argnums=())
def batched_elo_update(
    learner_rating: jnp.ndarray,
    opponent_ratings: jnp.ndarray,
    learner_scores: jnp.ndarray,
    k: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Vectorized ELO update for a batch of (opponent, score) match results.

    A fully on-device alternative to :class:`EloManager.update` for when the
    trainer wants to fold many per-opponent outcomes from one rollout into the
    ratings without leaving ``jit``. Updates are applied as a single mean-field
    step (the learner rating moves by the average residual), which is the
    standard way to aggregate a batch of simultaneous ELO matches.

    Parameters
    ----------
    learner_rating:
        Scalar current learner ELO.
    opponent_ratings:
        ``(n,)`` ELO of each opponent faced.
    learner_scores:
        ``(n,)`` learner score per match in [0, 1].
    k:
        Scalar K-factor.

    Returns
    -------
    (new_learner_rating, new_opponent_ratings):
        ``new_learner_rating`` scalar; ``new_opponent_ratings`` shape ``(n,)``.
    """
    exp = _expected_score(learner_rating, opponent_ratings)  # (n,)
    residual = learner_scores - exp                          # (n,)
    new_learner = learner_rating + k * jnp.mean(residual)
    new_opponents = opponent_ratings + k * ((1.0 - learner_scores) - (1.0 - exp))
    return new_learner, new_opponents


__all__ = [
    "OpponentPool",
    "init_opponent_pool",
    "push_snapshot",
    "sample_opponent",
    "pool_is_empty",
    "set_rating",
    "EloManager",
    "batched_elo_update",
]
