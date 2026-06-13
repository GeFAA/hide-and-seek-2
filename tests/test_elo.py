"""
tests/test_elo.py -- ELO rating math for historical self-play (utils/elo.py).

The ELO helpers are written with ``jnp`` ops (vmap/jit friendly), so this module
guards its dependency with ``pytest.importorskip("jax")``: without JAX the test
is skipped rather than erroring during collection.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # noqa: F841  (skip whole module without JAX)
import jax.numpy as jnp  # noqa: E402

from utils.elo import expected_score, update_elo  # noqa: E402


def test_expected_score_symmetry() -> None:
    """E(ra, rb) + E(rb, ra) == 1 (the logistic is antisymmetric in the diff)."""
    ra, rb = 1320.0, 1180.0
    ea = float(expected_score(ra, rb))
    eb = float(expected_score(rb, ra))
    assert ea + eb == pytest.approx(1.0, abs=1e-5)
    # Higher-rated player has the larger expected score.
    assert ea > 0.5 > eb


def test_expected_score_equal_ratings() -> None:
    """Equal ratings give a coin-flip expected score of 0.5."""
    assert float(expected_score(1200.0, 1200.0)) == pytest.approx(0.5, abs=1e-6)


def test_update_elo_zero_sum() -> None:
    """The rating mass added to A equals the mass removed from B (zero-sum)."""
    ra, rb = 1200.0, 1200.0
    ra_new, rb_new = update_elo(ra, rb, score_a=1.0, k=16.0)
    ra_new, rb_new = float(ra_new), float(rb_new)
    # Total rating is conserved.
    assert (ra_new + rb_new) == pytest.approx(ra + rb, abs=1e-4)
    # A's gain mirrors B's loss.
    assert (ra_new - ra) == pytest.approx(-(rb_new - rb), abs=1e-4)


def test_win_raises_winner_rating() -> None:
    """A win strictly raises the winner's rating and lowers the loser's."""
    ra, rb = 1200.0, 1200.0
    ra_new, rb_new = update_elo(ra, rb, score_a=1.0, k=16.0)
    assert float(ra_new) > ra
    assert float(rb_new) < rb
    # With equal ratings and K=16, an outright win moves each by exactly K/2 = 8.
    assert float(ra_new) == pytest.approx(ra + 8.0, abs=1e-4)
    assert float(rb_new) == pytest.approx(rb - 8.0, abs=1e-4)


def test_draw_is_neutral_for_equal_ratings() -> None:
    """A draw between equal-rated players leaves both ratings unchanged."""
    ra, rb = 1200.0, 1200.0
    ra_new, rb_new = update_elo(ra, rb, score_a=0.5, k=16.0)
    assert float(ra_new) == pytest.approx(ra, abs=1e-4)
    assert float(rb_new) == pytest.approx(rb, abs=1e-4)


def test_update_elo_vectorized() -> None:
    """The update broadcasts over arrays of matches (vmap/jit friendly)."""
    ra = jnp.array([1200.0, 1400.0])
    rb = jnp.array([1200.0, 1000.0])
    score_a = jnp.array([1.0, 0.0])
    ra_new, rb_new = update_elo(ra, rb, score_a, k=16.0)
    assert ra_new.shape == (2,)
    assert rb_new.shape == (2,)
    # Zero-sum holds elementwise.
    total = jnp.asarray(ra_new + rb_new)
    assert bool(jnp.allclose(total, ra + rb, atol=1e-3))
