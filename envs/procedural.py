"""
envs/procedural.py -- Vectorized per-episode world generation for **H&S 2.0**.

:func:`generate_episode` maps a ``PRNGKey`` to a fresh
``(PhysicsState, GameState)`` pair. Every episode randomizes:

* **Team sizes** -- hider / seeker counts in ``[min_team_size, max_team_size]``.
* **Props** -- box count, and a random *light vs heavy* assignment per box
  (2.0 cooperative physics), plus ramps, decoys, walls and doors.
* **Layout** -- positions sampled inside the arena (with simple spacing jitter).
* **Fog patches** -- random centers (2.0 fog of war).

Everything uses **fixed maximum sizes + active masks** so output shapes are
*static* (no data-dependent shapes) and the whole function is ``jit`` / ``vmap``
friendly: inactive slots are padded and flagged ``active = False``.

Entity row layout (fixed slot ranges; agents first so an agent's id matches in
both the ``A`` and ``E`` index spaces):

```
[0                 : n_hiders_max)                 -> hider slots
[n_hiders_max      : max_agents)                   -> seeker slots
[max_agents        : +n_boxes_max)                 -> boxes
[..                : +n_ramps_max)                 -> ramps
[..                : +n_decoys_max)                -> decoys
[..                : +n_walls_max)                 -> walls
[..                : +n_doors_max)                 -> doors
```
"""
from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from config import EnvConfig, TYPE_TO_ID
from envs.state import GameState, PhysicsState

__all__ = ["generate_episode", "entity_slot_ranges"]

# Large finite "inf-proxy" mass for static colliders (walls/doors): keeps
# ``F / mass`` numerically sane while making them effectively immovable.
_STATIC_MASS: float = 1.0e6


def entity_slot_ranges(cfg: EnvConfig) -> dict:
    """Return the fixed ``[start, stop)`` entity-row range for each category.

    The slot layout is part of the internal contract (agents first). Returned as
    plain Python ints (host-side; safe to use for static slicing).

    Parameters
    ----------
    cfg:
        :class:`config.EnvConfig`.

    Returns
    -------
    ranges:
        Dict mapping category name -> ``(start, stop)``.
    """
    a = cfg.max_agents
    h = cfg.n_hiders_max
    boxes = (a, a + cfg.n_boxes_max)
    ramps = (boxes[1], boxes[1] + cfg.n_ramps_max)
    decoys = (ramps[1], ramps[1] + cfg.n_decoys_max)
    walls = (decoys[1], decoys[1] + cfg.n_walls_max)
    doors = (walls[1], walls[1] + cfg.n_doors_max)
    return {
        "hiders": (0, h),
        "seekers": (h, a),
        "agents": (0, a),
        "boxes": boxes,
        "ramps": ramps,
        "decoys": decoys,
        "walls": walls,
        "doors": doors,
    }


def _sample_positions(key: jnp.ndarray, n: int, half: float, margin: float) -> jnp.ndarray:
    """Sample ``n`` uniform xy positions inside the arena (with a wall margin).

    Parameters
    ----------
    key:
        PRNGKey.
    n:
        Number of positions (static).
    half:
        Arena half-extent (``arena_size / 2``).
    margin:
        Keep-out distance from the arena walls.

    Returns
    -------
    pos:
        ``(n, 2)`` float32 positions.
    """
    lo, hi = -half + margin, half - margin
    return jax.random.uniform(key, (n, 2), minval=lo, maxval=hi)


def generate_episode(key: jnp.ndarray, cfg: EnvConfig) -> Tuple[PhysicsState, GameState]:
    """Generate a randomized episode as ``(PhysicsState, GameState)``.

    Fully vectorized & jit-friendly: fixed shapes, active masks, no Python branch
    on traced values.

    Parameters
    ----------
    key:
        ``jax.random.PRNGKey`` seeding all per-episode randomness.
    cfg:
        :class:`config.EnvConfig` (provides max sizes, masses, arena, fog count).

    Returns
    -------
    physics:
        Fresh :class:`PhysicsState` (per single env).
    game:
        Fresh :class:`GameState` (per single env).
    """
    E = cfg.max_entities
    A = cfg.max_agents
    half = cfg.arena_size * 0.5
    ranges = entity_slot_ranges(cfg)

    keys = jax.random.split(key, 12)

    # ---------------------------------------------------------------- #
    # 2.0: per-episode team sizes -- random counts within the configured range.
    # ---------------------------------------------------------------- #
    n_hiders = jax.random.randint(
        keys[0], (), cfg.min_team_size, cfg.max_team_size + 1
    )
    n_seekers = jax.random.randint(
        keys[1], (), cfg.min_team_size, cfg.max_team_size + 1
    )

    # Per-category active counts (random within max). Props are optional.
    n_boxes = jax.random.randint(keys[2], (), 1, cfg.n_boxes_max + 1)
    n_ramps = jax.random.randint(keys[3], (), 0, cfg.n_ramps_max + 1)
    n_decoys = jax.random.randint(keys[4], (), 0, cfg.n_decoys_max + 1)
    n_walls = jax.random.randint(keys[5], (), 0, cfg.n_walls_max + 1)
    n_doors = jax.random.randint(keys[6], (), 0, cfg.n_doors_max + 1)

    # ---------------------------------------------------------------- #
    # Build per-slot "active" masks from the sampled counts. We compare each
    # slot's *local* index against the count for its category (branch-free).
    # ---------------------------------------------------------------- #
    def slot_active(start: int, stop: int, count: jnp.ndarray) -> jnp.ndarray:
        local = jnp.arange(stop - start)
        return local < count

    active = jnp.zeros((E,), dtype=bool)
    active = active.at[ranges["hiders"][0]:ranges["hiders"][1]].set(
        slot_active(*ranges["hiders"], n_hiders)
    )
    active = active.at[ranges["seekers"][0]:ranges["seekers"][1]].set(
        slot_active(*ranges["seekers"], n_seekers)
    )
    active = active.at[ranges["boxes"][0]:ranges["boxes"][1]].set(
        slot_active(*ranges["boxes"], n_boxes)
    )
    active = active.at[ranges["ramps"][0]:ranges["ramps"][1]].set(
        slot_active(*ranges["ramps"], n_ramps)
    )
    active = active.at[ranges["decoys"][0]:ranges["decoys"][1]].set(
        slot_active(*ranges["decoys"], n_decoys)
    )
    active = active.at[ranges["walls"][0]:ranges["walls"][1]].set(
        slot_active(*ranges["walls"], n_walls)
    )
    active = active.at[ranges["doors"][0]:ranges["doors"][1]].set(
        slot_active(*ranges["doors"], n_doors)
    )

    # ---------------------------------------------------------------- #
    # Type ids per slot (static category assignment).
    # ---------------------------------------------------------------- #
    type_id = jnp.full((E,), -1, dtype=jnp.int32)
    type_id = type_id.at[ranges["hiders"][0]:ranges["hiders"][1]].set(TYPE_TO_ID["hider"])
    type_id = type_id.at[ranges["seekers"][0]:ranges["seekers"][1]].set(TYPE_TO_ID["seeker"])
    type_id = type_id.at[ranges["boxes"][0]:ranges["boxes"][1]].set(TYPE_TO_ID["box_light"])
    type_id = type_id.at[ranges["ramps"][0]:ranges["ramps"][1]].set(TYPE_TO_ID["ramp"])
    type_id = type_id.at[ranges["decoys"][0]:ranges["decoys"][1]].set(TYPE_TO_ID["decoy"])
    type_id = type_id.at[ranges["walls"][0]:ranges["walls"][1]].set(TYPE_TO_ID["wall"])
    type_id = type_id.at[ranges["doors"][0]:ranges["doors"][1]].set(TYPE_TO_ID["door"])

    # ---------------------------------------------------------------- #
    # 2.0: variable mass -- randomly promote some boxes to heavy (box_heavy).
    # ~40% of boxes become heavy; heavy boxes need cooperative pushing.
    # ---------------------------------------------------------------- #
    box_s, box_e = ranges["boxes"]
    heavy_roll = jax.random.uniform(keys[7], (box_e - box_s,))
    is_heavy_box = heavy_roll < 0.4
    box_types = jnp.where(is_heavy_box, TYPE_TO_ID["box_heavy"], TYPE_TO_ID["box_light"])
    type_id = type_id.at[box_s:box_e].set(box_types)

    # ---------------------------------------------------------------- #
    # Positions: all entities sampled uniformly (simple layout). Seekers are
    # nudged toward one edge so the prep phase has spatial meaning.
    # ---------------------------------------------------------------- #
    pos_xy = _sample_positions(keys[8], E, half, margin=cfg.agent_radius * 2.0)
    # Push seekers toward +y edge at start (gives hiders room during prep).
    seek_s, seek_e = ranges["seekers"]
    seeker_shift = jnp.zeros((E, 2)).at[seek_s:seek_e, 1].set(half * 0.6)
    pos_xy = pos_xy + seeker_shift
    pos_xy = jnp.clip(pos_xy, -half + cfg.agent_radius, half - cfg.agent_radius)

    pos = jnp.zeros((E, 3), dtype=jnp.float32)
    pos = pos.at[:, :2].set(pos_xy)  # z = 0 (everyone starts grounded)

    # ---------------------------------------------------------------- #
    # Per-entity sizes & masses.
    # ---------------------------------------------------------------- #
    size = jnp.full((E,), cfg.agent_radius, dtype=jnp.float32)
    size = size.at[box_s:box_e].set(0.5)
    size = size.at[ranges["ramps"][0]:ranges["ramps"][1]].set(0.7)
    size = size.at[ranges["decoys"][0]:ranges["decoys"][1]].set(0.3)
    size = size.at[ranges["walls"][0]:ranges["walls"][1]].set(0.6)
    size = size.at[ranges["doors"][0]:ranges["doors"][1]].set(0.5)

    mass = jnp.ones((E,), dtype=jnp.float32)
    mass = mass.at[ranges["agents"][0]:ranges["agents"][1]].set(1.0)
    # Boxes: light vs heavy mass (2.0 variable mass).
    box_mass = jnp.where(is_heavy_box, cfg.box_heavy_mass, cfg.box_light_mass)
    mass = mass.at[box_s:box_e].set(box_mass)
    mass = mass.at[ranges["ramps"][0]:ranges["ramps"][1]].set(cfg.box_light_mass * 1.5)
    mass = mass.at[ranges["decoys"][0]:ranges["decoys"][1]].set(0.5)
    # Static obstacles: inf-proxy mass.
    mass = mass.at[ranges["walls"][0]:ranges["walls"][1]].set(_STATIC_MASS)
    mass = mass.at[ranges["doors"][0]:ranges["doors"][1]].set(_STATIC_MASS)

    # Headings: agents face a random direction; props irrelevant.
    heading = jax.random.uniform(keys[9], (E,), minval=-jnp.pi, maxval=jnp.pi)
    ang_vel = jnp.zeros((E,), dtype=jnp.float32)
    vel = jnp.zeros((E, 3), dtype=jnp.float32)
    grounded = jnp.ones((E,), dtype=bool)  # z=0 everywhere at reset

    physics = PhysicsState(
        pos=pos,
        vel=vel,
        heading=heading,
        ang_vel=ang_vel,
        mass=mass,
        size=size,
        type_id=type_id,
        grounded=grounded,
        active=active,
    )

    # ================================================================ #
    # GameState
    # ================================================================ #
    # team: 0=hider, 1=seeker, -1 for pad agent slots / non-agents.
    team = jnp.full((A,), -1, dtype=jnp.int32)
    team = team.at[ranges["hiders"][0]:ranges["hiders"][1]].set(0)
    team = team.at[ranges["seekers"][0]:ranges["seekers"][1]].set(1)
    # Deactivate pad agent slots' team.
    agent_active = active[:A]
    team = jnp.where(agent_active, team, -1)

    stamina = jnp.full((A,), cfg.stamina_max, dtype=jnp.float32)
    holding = jnp.full((A,), -1, dtype=jnp.int32)

    locked = jnp.zeros((E,), dtype=bool)
    locked_by = jnp.full((E,), -1, dtype=jnp.int32)

    # 2.0: deception -- the decoy slots carry the *true* decoy identity (privileged).
    is_decoy = jnp.zeros((E,), dtype=bool)
    dec_s, dec_e = ranges["decoys"]
    is_decoy = is_decoy.at[dec_s:dec_e].set(active[dec_s:dec_e])
    decoy_timer = jnp.zeros((E,), dtype=jnp.int32)
    emitted_noise = jnp.zeros((E,), dtype=jnp.float32)

    # 2.0: destructible walls -- full hp where a wall is active, else 0.
    wall_hp = jnp.zeros((E,), dtype=jnp.float32)
    wall_s, wall_e = ranges["walls"]
    # Give each active wall full health (use wall_break_speed-agnostic pool).
    wall_hp = wall_hp.at[wall_s:wall_e].set(
        jnp.where(active[wall_s:wall_e], 100.0, 0.0)
    )

    door_progress = jnp.zeros((E,), dtype=jnp.float32)

    # 2.0: fog of war -- random fog patch centers inside the arena.
    fog_pos = _sample_positions(keys[10], cfg.n_fog_patches, half, margin=cfg.fog_radius)

    game = GameState(
        team=team,
        stamina=stamina,
        holding=holding,
        locked=locked,
        locked_by=locked_by,
        is_decoy=is_decoy,
        decoy_timer=decoy_timer,
        emitted_noise=emitted_noise,
        wall_hp=wall_hp,
        door_progress=door_progress,
        fog_pos=fog_pos,
        phase=jnp.array(0, dtype=jnp.int32),   # start in prep
        step=jnp.array(0, dtype=jnp.int32),
    )

    return physics, game
