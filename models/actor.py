"""
models/actor.py -- Decentralized recurrent actor for Hide & Seek 2.0.

``ActorRNN`` is the policy network each agent runs **on its own local, masked
observation** (CTDE: *decentralized execution*). It never sees privileged global
state -- in particular it cannot tell a real hider from a decoy except by
attending over what it can actually perceive.

Forward pipeline::

    local obs (entities, entity_mask, self)
        -> EntityTransformer  (permutation-invariant masked set encoder)
        -> ScannedGRU         (memory / object permanence across time)
        -> hybrid action heads:
             * continuous: a diagonal Gaussian over `action_move_dim`
               (force_x, force_y, torque), tanh-squash documented below;
             * discrete: one distrax.Categorical per `action_discrete_nvec`
               entry  (grab/release, lock/unlock, decoy on/off).

The ``__call__`` signature follows the PureJaxRL recurrent convention:
``(carry, (obs, dones)) -> (carry, pi)`` so it scans cleanly over time.

Action (log-)probability bookkeeping lives in the module-level helpers
:func:`sample_and_logprob` and :func:`eval_logprob`, which sum the continuous
and discrete components into a single scalar log-prob / entropy per agent (per
the contract: total logprob = move logprob + sum of categorical logprobs).

**Tanh-squashed move action (correctness, see PINNED P2).** The continuous
``move`` command must live in ``[-1, 1]`` (CONTRACT §4). We model it as a
*tanh-squashed* diagonal Gaussian: a base :class:`distrax.MultivariateNormalDiag`
pushed through a :class:`distrax.Tanh` bijector (wrapped in
:class:`distrax.Block` so all ``action_move_dim`` components share one event).
The resulting :class:`distrax.Transformed` distribution lives directly in
``[-1, 1]`` space, and its ``log_prob`` includes the exact tanh change-of-variables
``log|det J|`` correction. Crucially:

* the action returned/stored is the **squashed** value in ``[-1, 1]``;
* :func:`sample_and_logprob` scores that squashed sample, and
  :func:`eval_logprob` scores the stored squashed action *directly* (no
  ``arctanh`` round-trip of a raw Gaussian sample) -- so the two are mutually
  consistent and the PPO ratio is correct even when the pre-squash sample is
  large (where the old raw-Gaussian + ``arctanh`` path saturated in float32 and
  corrupted the ratio);
* entropy reflects the squashing. The squashed distribution has no closed-form
  entropy, so we use the standard single-sample estimate
  ``H[base] - E[log|det J|] ≈ H[base] + sum(log(1 - tanh(raw)^2))`` evaluated at
  one sample (documented inline). ``H[base]`` is the exact diagonal-Gaussian
  entropy.

To keep the (manual-friendly) Jacobian well-conditioned the actor head clamps
``log_std`` to ``[-4, 2]`` and the pre-squash sample to ``[-5, 5]``.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp

from config import ModelConfig
from models.memory import ScannedGRU
from models.transformer import EntityTransformer

# A container for the hybrid policy: a continuous (tanh-squashed) move
# distribution plus a list of independent categorical interaction distributions.
# ``move_dist`` is a distrax.Transformed (Gaussian -> Tanh) living in [-1, 1].
Pi = Tuple[distrax.Distribution, List[distrax.Categorical]]

# Clamp ranges keeping the tanh squash numerically well-conditioned (PINNED P2):
# bound log_std so the base Gaussian neither collapses nor explodes, and clamp
# the pre-squash sample so tanh does not saturate to exactly +-1 in float32.
_LOG_STD_MIN: float = -4.0
_LOG_STD_MAX: float = 2.0
_RAW_SAMPLE_CLAMP: float = 5.0


def _make_move_dist(
    move_mean: jnp.ndarray,
    move_std: jnp.ndarray,
) -> distrax.Distribution:
    """Build the tanh-squashed diagonal-Gaussian move distribution.

    The base is a :class:`distrax.MultivariateNormalDiag`; it is pushed through a
    :class:`distrax.Tanh` bijector wrapped in :class:`distrax.Block` (``ndims=1``)
    so the squash acts on the whole ``action_move_dim`` event at once and the
    resulting :class:`distrax.Transformed` distribution lives in ``[-1, 1]`` with
    an exact tanh change-of-variables ``log|det J|`` correction in ``log_prob``.

    Args:
        move_mean: Gaussian mean ``(..., action_move_dim)``.
        move_std: Gaussian (already-positive) std ``(..., action_move_dim)``.

    Returns:
        A :class:`distrax.Transformed` distribution over ``[-1, 1]^d``.
    """
    base = distrax.MultivariateNormalDiag(loc=move_mean, scale_diag=move_std)
    bijector = distrax.Block(distrax.Tanh(), ndims=1)
    return distrax.Transformed(base, bijector)


class ActorRNN(nn.Module):
    """Recurrent, decentralized actor: entity-Transformer -> GRU -> hybrid heads.

    Attributes:
        cfg: Shared :class:`ModelConfig`. Provides every dimension
            (``d_model``, ``gru_hidden``, ``action_move_dim``,
            ``action_discrete_nvec``, ``log_std_init``) -- nothing is hard-coded.
    """

    cfg: ModelConfig

    @nn.compact
    def __call__(
        self,
        carry: jnp.ndarray,
        x: Tuple[Dict[str, jnp.ndarray], jnp.ndarray],
    ) -> Tuple[jnp.ndarray, Pi]:
        """Run the actor for a TIME-major batch of local observations.

        Args:
            carry: GRU hidden state ``(*batch_dims, gru_hidden)``.
            x: Tuple ``(obs, dones)`` where:

                * ``obs`` is the **local** observation dict (time-major), using
                  contract keys (CONTRACT §3): ``"entities"`` ``(T, *B, E, Fe)``,
                  ``"entity_mask"`` ``(T, *B, E)`` bool, ``"self"`` ``(T, *B, Fs)``.
                  (``"agent_active"`` may be present but is not needed here; the
                  trainer masks inactive agents when forming the loss.)
                * ``dones`` is ``(T, *B)`` bool episode-boundary flags for the GRU.

        Returns:
            ``(new_carry, pi)`` where ``pi`` is a tuple
            ``(move_dist, interact_dists)``:

            * ``move_dist``: a **tanh-squashed** diagonal Gaussian -- a
                :class:`distrax.Transformed` (``MultivariateNormalDiag`` ->
                ``Block(Tanh)``) over ``[-1, 1]^{action_move_dim}`` (the continuous
                force/torque command). Sampling yields a value already in
                ``[-1, 1]`` and ``log_prob`` includes the exact tanh
                change-of-variables correction (CONTRACT §4 / PINNED P2), so no
                separate squashing or ``arctanh`` round-trip is needed downstream.
            * ``interact_dists``: one :class:`distrax.Categorical` per entry of
                ``cfg.action_discrete_nvec`` (grab, lock, decoy).
        """
        obs, dones = x

        # --- 1. Encode the LOCAL masked entity set (decentralized perception). ---
        embed = EntityTransformer(self.cfg, name="encoder")(
            obs["entities"], obs["entity_mask"], obs["self"]
        )  # (T, *B, 2 * d_model)

        # --- 2. Recurrent memory over time, reset at episode boundaries. ---
        carry, gru_out = ScannedGRU(self.cfg.gru_hidden, name="memory")(
            carry, (embed, dones)
        )  # gru_out: (T, *B, gru_hidden)

        # A shared trunk before the heads gives the policy a little extra capacity
        # to recombine the recurrent summary before branching into heads.
        trunk = nn.relu(nn.Dense(self.cfg.gru_hidden, name="trunk")(gru_out))

        # --- 3a. Continuous locomotion head: tanh-squashed diagonal Gaussian. ---
        # State-independent log_std parameter (a common, stable choice for
        # continuous-control PPO): one learnable scalar per move dimension. We
        # clamp log_std to [_LOG_STD_MIN, _LOG_STD_MAX] so the base Gaussian
        # (and thus the tanh Jacobian) stays well-conditioned (PINNED P2).
        move_mean = nn.Dense(self.cfg.action_move_dim, name="move_mean")(trunk)
        log_std = self.param(
            "move_log_std",
            lambda _key, shape: jnp.full(shape, self.cfg.log_std_init),
            (self.cfg.action_move_dim,),
        )
        log_std = jnp.clip(log_std, _LOG_STD_MIN, _LOG_STD_MAX)
        move_std = jnp.exp(log_std)
        # Broadcast the per-dim std across all batch/time axes of the mean.
        move_std = jnp.broadcast_to(move_std, move_mean.shape)
        # Tanh-squashed Gaussian over [-1, 1]: exact log|det J| handled by distrax.
        move_dist = _make_move_dist(move_mean, move_std)

        # --- 3b. Discrete interaction heads: one Categorical per nvec entry. ---
        interact_dists: List[distrax.Categorical] = []
        for i, n in enumerate(self.cfg.action_discrete_nvec):
            logits = nn.Dense(n, name=f"interact_logits_{i}")(trunk)
            interact_dists.append(distrax.Categorical(logits=logits))

        return carry, (move_dist, interact_dists)


# ---------------------------------------------------------------------------
# Module-level (log-)probability helpers.
#
# The hybrid action is a dict (CONTRACT §4):
#   {"move": (..., action_move_dim) float in [-1,1] (the TANH-SQUASHED action),
#    "interact": (..., n_discrete) int}
#
# Total logprob = squashed-move logprob (the distrax.Transformed log_prob, which
# already includes the tanh log|det J| correction) + sum of the categorical
# logprobs. Total entropy = squashed-move entropy (single-sample estimate, see
# below) + sum of categorical entropies. Both are returned per-agent (no batch
# reduction here).
#
# Entropy of a tanh-squashed Gaussian has no closed form, so we use the standard
# single-sample estimator (PINNED P2):
#     H[squashed] = H[base Gaussian] - E[ log|det J| ]
#                 ≈ H[base Gaussian] + sum_d log(1 - tanh(raw_d)^2)
# evaluated at a single pre-squash sample ``raw`` (``log|det J|`` of the tanh map
# is ``sum_d log(1 - tanh(raw_d)^2)``, hence the sign flip turns the subtraction
# into an addition). ``H[base Gaussian]`` is the exact diagonal-Gaussian entropy.
# ``sample_and_logprob`` uses the freshly drawn ``raw``; ``eval_logprob`` recovers
# ``raw`` from the stored squashed action via the bijector inverse FOR THE ENTROPY
# TERM ONLY -- the log-prob itself is scored directly on the squashed action.
# ---------------------------------------------------------------------------
def _tanh_logdetj(raw: jnp.ndarray) -> jnp.ndarray:
    """Return ``sum_d log(1 - tanh(raw_d)^2)`` (tanh map log|det J|), per item.

    Computed in a numerically stable form,
    ``log(1 - tanh(x)^2) = 2 * (log 2 - x - softplus(-2x))``, which avoids the
    ``1 - tanh^2`` cancellation that underflows for large ``|x|``.

    Args:
        raw: Pre-squash values ``(..., action_move_dim)``.

    Returns:
        ``(...)`` summed log|det J| over the ``action_move_dim`` event axis.
    """
    per_dim = 2.0 * (jnp.log(2.0) - raw - jax.nn.softplus(-2.0 * raw))
    return jnp.sum(per_dim, axis=-1)


def _squashed_move_entropy(
    move_dist: distrax.Distribution,
    raw: jnp.ndarray,
) -> jnp.ndarray:
    """Single-sample entropy estimate of the tanh-squashed move distribution.

    ``H[squashed] ≈ H[base Gaussian] + sum_d log(1 - tanh(raw_d)^2)`` evaluated at
    one pre-squash sample ``raw`` (see module comment above).

    Args:
        move_dist: The :class:`distrax.Transformed` move distribution; its
            ``.distribution`` attribute is the base diagonal Gaussian.
        raw: A pre-squash sample ``(..., action_move_dim)``.

    Returns:
        ``(...)`` entropy estimate of the squashed move distribution.
    """
    base_entropy = move_dist.distribution.entropy()  # exact Gaussian entropy
    return base_entropy + _tanh_logdetj(raw)


def sample_and_logprob(
    pi: Pi,
    key: jax.Array,
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Sample a hybrid action and return its joint log-prob and entropy.

    Args:
        pi: The policy tuple ``(move_dist, interact_dists)`` from
            :meth:`ActorRNN.__call__` -- ``move_dist`` is a tanh-squashed Gaussian
            (:class:`distrax.Transformed`) over ``[-1, 1]``.
        key: A ``PRNGKey`` (explicit randomness, per the contract).

    Returns:
        ``(action, logprob, entropy)`` where:

        * ``action``: dict with ``"move"`` ``(..., action_move_dim)`` float in
            ``[-1, 1]`` (the **squashed** sample) and ``"interact"``
            ``(..., n_discrete)`` int.
        * ``logprob``: ``(...)`` joint log-prob = squashed-move logprob (incl. the
            tanh ``log|det J|`` correction) + sum of discrete logprobs.
        * ``entropy``: ``(...)`` joint entropy = squashed-move entropy
            (single-sample estimate) + sum of discrete entropies.
    """
    move_dist, interact_dists = pi

    # Split the key: one for the continuous head, one per discrete head.
    keys = jax.random.split(key, 1 + len(interact_dists))
    move_key, interact_keys = keys[0], keys[1:]

    # Continuous: draw a *raw* pre-squash sample from the base Gaussian, clamp it
    # to [-_RAW_SAMPLE_CLAMP, _RAW_SAMPLE_CLAMP] so tanh does not saturate to
    # exactly +-1 in float32, then push it through the tanh bijector to get the
    # squashed action in [-1, 1]. log_prob is taken on the Transformed dist (so it
    # includes the exact tanh log|det J| correction); entropy uses the single-
    # sample estimate from the same raw sample.
    raw_move = move_dist.distribution.sample(seed=move_key)
    raw_move = jnp.clip(raw_move, -_RAW_SAMPLE_CLAMP, _RAW_SAMPLE_CLAMP)
    move_action = move_dist.bijector.forward(raw_move)  # squashed -> [-1, 1]
    move_logprob = move_dist.log_prob(move_action)
    move_entropy = _squashed_move_entropy(move_dist, raw_move)

    # Discrete: sample + score each categorical head, then stack to (..., n_disc).
    interact_samples = []
    interact_logprob = jnp.zeros_like(move_logprob)
    interact_entropy = jnp.zeros_like(move_entropy)
    for d, k in zip(interact_dists, interact_keys):
        a = d.sample(seed=k)
        interact_samples.append(a)
        interact_logprob = interact_logprob + d.log_prob(a)
        interact_entropy = interact_entropy + d.entropy()
    interact_action = jnp.stack(interact_samples, axis=-1).astype(jnp.int32)

    action = {"move": move_action, "interact": interact_action}
    logprob = move_logprob + interact_logprob
    entropy = move_entropy + interact_entropy
    return action, logprob, entropy


def eval_logprob(
    pi: Pi,
    action: Dict[str, jnp.ndarray],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Evaluate the joint log-prob and entropy of a *given* hybrid action.

    Used in the PPO update to score stored actions under the current policy. The
    stored ``move`` is the **squashed** action in ``[-1, 1]``; it is scored
    *directly* by the tanh-squashed distribution (no ``arctanh`` round-trip of a
    raw sample), so it is mutually consistent with :func:`sample_and_logprob`.

    Args:
        pi: The policy tuple ``(move_dist, interact_dists)`` -- ``move_dist`` is a
            tanh-squashed Gaussian (:class:`distrax.Transformed`) over ``[-1, 1]``.
        action: Action dict with ``"move"`` (the **tanh-squashed** action stored
            during rollout, in ``[-1, 1]``) and ``"interact"`` (int,
            ``(..., n_discrete)``).

    Returns:
        ``(logprob, entropy)`` -- the joint log-prob and entropy, summed over the
        continuous + discrete components, shape ``(...)`` each.
    """
    move_dist, interact_dists = pi

    # Clip strictly inside (-1, 1) for numerical safety, then score the squashed
    # action DIRECTLY under the tanh-squashed distribution. Its log_prob already
    # contains the tanh log|det J| correction -- no arctanh round-trip of a raw
    # Gaussian sample (which would saturate in float32 and corrupt the PPO ratio).
    squashed = jnp.clip(action["move"], -1.0 + 1e-6, 1.0 - 1e-6)
    move_logprob = move_dist.log_prob(squashed)
    # For the entropy term only, recover a pre-squash ``raw`` via the bijector
    # inverse (== arctanh). This feeds the single-sample entropy estimate; it does
    # NOT touch the log-prob above. Clamp raw for the same float32 safety reason.
    raw_move = move_dist.bijector.inverse(squashed)
    raw_move = jnp.clip(raw_move, -_RAW_SAMPLE_CLAMP, _RAW_SAMPLE_CLAMP)
    move_entropy = _squashed_move_entropy(move_dist, raw_move)

    interact = action["interact"]  # (..., n_discrete) int
    interact_logprob = jnp.zeros_like(move_logprob)
    interact_entropy = jnp.zeros_like(move_entropy)
    for i, d in enumerate(interact_dists):
        interact_logprob = interact_logprob + d.log_prob(interact[..., i])
        interact_entropy = interact_entropy + d.entropy()

    logprob = move_logprob + interact_logprob
    entropy = move_entropy + interact_entropy
    return logprob, entropy
