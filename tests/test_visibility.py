"""
tests/test_visibility.py -- ray-casting / line-of-sight geometry (utils/visibility.py).

Exercises the visibility primitive that drives both ``entity_mask`` (CONTRACT §3)
and the reward's ``any_hider_seen`` signal (CONTRACT §7) through the **public**
:func:`utils.visibility.compute_visibility` (the per-observer mask that
:func:`compute_visibility_batch` vmaps). We assert the three required LOS cases:

* a wall segment placed *between* observer and target blocks line-of-sight;
* with a clear line (no occluder) the target is visible;
* a degenerate (zero-length) segment -- how inactive walls / opened doors are
  represented per PINNED P3 -- never blocks anything.

Only the documented public API is used, so the test is robust to private
helper renames. Guarded with ``importorskip("jax")`` for clean collection
without JAX.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # noqa: F841  (skip whole module without JAX)
import jax.numpy as jnp  # noqa: E402

from config import default_config  # noqa: E402
from utils.visibility import compute_visibility  # noqa: E402

# A single target sitting straight ahead of an observer at the origin.
_OBSERVER = jnp.array([0.0, 0.0], dtype=jnp.float32)
_HEADING = 0.0  # facing +x, so a target at (+x, 0) is dead-center in the cone.
_TARGET = jnp.array([[4.0, 0.0]], dtype=jnp.float32)  # (E=1, 2)
_ACTIVE = jnp.array([True])

# A vertical wall crossing the x=2 line between observer (0,0) and target (4,0).
_BLOCKING_WALL = jnp.array([[2.0, -1.5, 2.0, 1.5]], dtype=jnp.float32)  # (1, 4)
# A zero-length (degenerate) segment: both endpoints coincide -> occludes nothing.
_DEGENERATE_WALL = jnp.array([[2.0, 0.0, 2.0, 0.0]], dtype=jnp.float32)  # (1, 4)
# A wall that is nowhere near the observer->target line.
_OFFSIDE_WALL = jnp.array([[8.0, 8.0, 9.0, 9.0]], dtype=jnp.float32)  # (1, 4)

_NO_FOG = jnp.zeros((0, 2), dtype=jnp.float32)


def _visible(walls: jnp.ndarray, active: jnp.ndarray = _ACTIVE) -> bool:
    """Run :func:`compute_visibility` for the fixed target and return its bool."""
    cfg = default_config().env
    vis = compute_visibility(
        _OBSERVER, _HEADING, _TARGET, active, walls, _NO_FOG, cfg
    )
    assert vis.shape == (1,)
    return bool(vis[0])


def test_clear_line_is_visible() -> None:
    """With an off-to-the-side wall, an in-range, in-cone target is visible."""
    assert _visible(_OFFSIDE_WALL) is True


def test_wall_between_blocks_los() -> None:
    """A wall straddling the segment observer->target hides the target."""
    assert _visible(_BLOCKING_WALL) is False


def test_degenerate_segment_never_blocks() -> None:
    """A zero-length (inactive/opened) segment never occludes a clear sightline."""
    assert _visible(_DEGENERATE_WALL) is True


def test_inactive_entity_never_visible() -> None:
    """An inactive entity is masked out regardless of geometry."""
    assert _visible(_OFFSIDE_WALL, active=jnp.array([False])) is False


def test_target_behind_observer_out_of_cone() -> None:
    """A target behind the observer falls outside the forward vision cone."""
    cfg = default_config().env
    behind = jnp.array([[-4.0, 0.0]], dtype=jnp.float32)
    vis = compute_visibility(
        _OBSERVER, _HEADING, behind, _ACTIVE, _OFFSIDE_WALL, _NO_FOG, cfg
    )
    assert bool(vis[0]) is False
