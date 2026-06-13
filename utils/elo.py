"""
utils/elo.py -- Pure ELO rating math for historical self-play.

Used by ``trainers/selfplay.py`` (``EloManager`` / ``OpponentPool``) to rank
frozen opponent snapshots and to update ratings from match outcomes.

Everything here is a pure function of its arguments and works on both Python
scalars and ``jnp`` arrays (the functions are written with ``jnp`` ops, which
broadcast and accept scalars), so they are ``vmap``/``jit`` friendly.

ELO recap
---------
The expected score of player A vs B with ratings ``ra``/``rb`` is the logistic

    E_A = 1 / (1 + 10 ** ((rb - ra) / 400)).

After a match with realized score ``score_a in [0, 1]`` (1=win, 0.5=draw, 0=loss),
A's rating updates by ``ra' = ra + k * (score_a - E_A)`` and, in a zero-sum
two-player game, B's by the symmetric ``rb' = rb + k * (score_b - E_B)`` where
``score_b = 1 - score_a`` and ``E_B = 1 - E_A``.
"""
from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp

__all__ = ["ELO_SCALE", "expected_score", "update_elo"]

# Standard ELO logistic scale (rating diff that yields ~10:1 expected odds).
ELO_SCALE: float = 400.0


def expected_score(ra: jnp.ndarray, rb: jnp.ndarray) -> jnp.ndarray:
    """Expected score of player A against player B.

    Args:
        ra: Rating(s) of player A. Scalar or array.
        rb: Rating(s) of player B. Broadcasts against ``ra``.

    Returns:
        ``E_A`` in ``(0, 1)`` with the broadcast shape of ``ra``/``rb``.
    """
    ra = jnp.asarray(ra, dtype=jnp.float32)
    rb = jnp.asarray(rb, dtype=jnp.float32)
    return 1.0 / (1.0 + jnp.power(10.0, (rb - ra) / ELO_SCALE))


def update_elo(
    ra: jnp.ndarray,
    rb: jnp.ndarray,
    score_a: jnp.ndarray,
    k: float = 16.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Symmetric zero-sum ELO update for a single (vectorized) match.

    Args:
        ra: Rating(s) of player A before the match.
        rb: Rating(s) of player B before the match.
        score_a: Realized score for A in ``[0, 1]`` (1=win, 0.5=draw, 0=loss).
            Player B's realized score is implicitly ``1 - score_a`` (zero-sum).
        k: K-factor (update step size). Larger => faster rating swings.

    Returns:
        ``(ra_new, rb_new)`` updated ratings, broadcasting over the inputs.
    """
    ra = jnp.asarray(ra, dtype=jnp.float32)
    rb = jnp.asarray(rb, dtype=jnp.float32)
    score_a = jnp.asarray(score_a, dtype=jnp.float32)

    exp_a = expected_score(ra, rb)
    exp_b = 1.0 - exp_a            # E_B = 1 - E_A for a two-player logistic
    score_b = 1.0 - score_a       # zero-sum realized outcome

    ra_new = ra + k * (score_a - exp_a)
    rb_new = rb + k * (score_b - exp_b)
    return ra_new, rb_new
