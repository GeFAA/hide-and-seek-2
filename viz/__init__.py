"""
viz/ -- The Hide & Seek 2.0 **3D trajectory viewer** package.

This package is the bridge between a (JAX) Hide & Seek 2.0 episode and a
browser-based Three.js viewer. It is deliberately split so the *contract* and
the *demo generator* stay dependency-free while only the live recorder reaches
for jax/numpy:

* :mod:`viz.schema`               -- the single source of truth for the on-disk
  trajectory format (``format="hns2-traj"``, version 1). **Stdlib only.**
* :mod:`viz.make_demo_trajectory` -- a pure-stdlib synthetic episode generator
  that showcases the 2.0 mechanics so the viewer works out of the box on any
  machine (no jax, no numpy). Produces ``web/trajectories/demo_trajectory.json``.
* :mod:`viz.recorder`             -- :class:`TrajectoryRecorder` that pulls real
  rollouts off the device and maps them onto the schema (lazy jax/numpy import).
* :mod:`viz.serve`                -- a tiny static HTTP server so the ES-module /
  importmap viewer can be loaded over ``http://`` (never ``file://``).
* ``viz/web/*``                   -- the Three.js dark viewer that plays a file.

The schema helpers are re-exported here for convenience so callers can simply::

    from viz import make_trajectory, save_trajectory, validate_trajectory

without having to know they physically live in :mod:`viz.schema`. We re-export
**only** the stdlib-safe schema surface; importing :mod:`viz` therefore never
drags in jax or numpy.
"""
from __future__ import annotations

# Re-export the full, stdlib-only schema surface. Importing this package must
# never require jax/numpy, so we pull from viz.schema (which is pure stdlib) and
# explicitly do *not* import viz.recorder here.
from viz.schema import (
    FORMAT,
    VERSION,
    ENTITY_TYPES,
    FRAME_ENT_KEYS,
    TEAM_HIDER,
    TEAM_SEEKER,
    TEAM_NONE,
    make_entity_meta,
    make_frame_ent,
    make_frame,
    make_trajectory,
    save_trajectory,
    load_trajectory,
    validate_trajectory,
)

__all__ = [
    "FORMAT",
    "VERSION",
    "ENTITY_TYPES",
    "FRAME_ENT_KEYS",
    "TEAM_HIDER",
    "TEAM_SEEKER",
    "TEAM_NONE",
    "make_entity_meta",
    "make_frame_ent",
    "make_frame",
    "make_trajectory",
    "save_trajectory",
    "load_trajectory",
    "validate_trajectory",
]
