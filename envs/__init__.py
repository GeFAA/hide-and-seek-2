"""
envs/ -- The **Hide & Seek 2.0** simulation core.

Public surface (per CONTRACT.md §1):

* :class:`HideAndSeekEnv` -- the functional, ``jit``/``vmap``-safe environment.
* :class:`State`, :class:`GameState`, :class:`PhysicsState` -- the registered
  ``flax.struct.dataclass`` pytrees that hold all per-env state.

Also re-exported for convenience: :func:`generate_episode` (procedural world
generation) and :func:`physics_step` (the vectorized 2.5D physics integrator).
"""
from __future__ import annotations

from envs.hide_and_seek import HideAndSeekEnv
from envs.physics import physics_step
from envs.procedural import generate_episode
from envs.state import GameState, PhysicsState, State

__all__ = [
    "HideAndSeekEnv",
    "State",
    "GameState",
    "PhysicsState",
    "generate_episode",
    "physics_step",
]
