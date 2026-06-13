"""
utils/visibility.py -- GPU ray-casting & visibility for Hide & Seek 2.0.

This module is the sensing backbone: it computes what each agent can actually
*see*, which drives both the per-agent observation mask (CONTRACT §3,
``entity_mask``) and the reward signal (CONTRACT §7, ``any_hider_seen``).

Everything is written in vectorized ``jnp`` with **no Python loops over rays or
entities** and is designed to be ``vmap``-mapped over observers. There is no
host-side control flow and no ``.item()``; all branching is ``jnp.where`` /
``lax.select``.

Geometry model (intentionally 2-D / planar)
-------------------------------------------
Walls and other occluders are represented as line **segments** in the (x, y)
plane: an array of shape ``(M, 4)`` packed as ``[x0, y0, x1, y1]`` per row. This
is the standard "2.5D" simplification used by the env (elevation ``z`` only gates
climbing, not occlusion). Where this is a deliberate simplification (e.g. we treat
occluders as infinitely-tall segments and ignore z for line-of-sight), it is
called out inline.

Public API
----------
* :func:`cast_rays`        -- batched ray-vs-segment intersection -> per-ray range.
* :func:`lidar_scan`       -- a fan of rays from an agent, fog-attenuated.
* :func:`in_vision_cone`   -- is a target within the observer's forward cone?
* :func:`compute_visibility` -- full per-entity boolean visibility mask combining
  active / range / cone / line-of-sight, with fog-of-war range attenuation.
* :func:`walls_to_segments` -- pack active wall/door entities into occluder segments.
* :func:`compute_visibility_batch` -- vmap :func:`compute_visibility` over observers
  to produce the ``(A, E)`` ``entity_mask`` of the obs contract (CONTRACT §3).

Branch-free guarantee (CONTRACT §0)
-----------------------------------
There is **no host-side Python branching on array shapes** anywhere in the hot
path (no ``if segments.shape[0] == 0`` guards). Empty / degenerate occluder and
fog arrays are handled purely arithmetically:

* an empty ``(0, ...)`` occluder/fog array makes the ``jnp.any(..., axis=1)``
  reduction collapse to all-``False`` (the identity of ``or``), so nothing is
  blocked / attenuated;
* a **degenerate** (zero-length) segment -- both endpoints equal, so its edge
  vector is ``~0`` -- has a (near-)zero system determinant and is therefore
  caught by the parallel/zero-denominator mask, yielding *no* intersection
  (range stays at ``max_range``). Opened doors / broken walls are emitted as such
  degenerate segments so they stop occluding.
"""
from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp

__all__ = [
    "cast_rays",
    "lidar_scan",
    "in_vision_cone",
    "compute_visibility",
    "walls_to_segments",
    "compute_visibility_batch",
]

# Small epsilon to guard divisions by (near-)zero denominators (parallel rays).
_EPS: float = 1e-8


def _ray_segment_distance(
    origin: jnp.ndarray,      # (2,)
    direction: jnp.ndarray,   # (R, 2) unit direction per ray
    seg: jnp.ndarray,         # (M, 4) -> x0,y0,x1,y1
    max_range: float,
) -> jnp.ndarray:
    """Distance from ``origin`` along each ray to each segment (or ``max_range``).

    Solves, for ray ``origin + t * dir`` (t>=0) and segment ``p + u*(q-p)``
    (u in [0,1]), the 2x2 linear system for ``(t, u)`` analytically and vectorizes
    over the (R rays) x (M segments) outer product.

    Returns:
        ``(R, M)`` array of hit distances; entries where the ray misses (no valid
        ``t>=0`` with ``u in [0,1]``) are set to ``max_range``.
    """
    # Ray params: origin o (2,), dir d (R,2).
    ox, oy = origin[0], origin[1]
    dx = direction[:, 0][:, None]   # (R,1)
    dy = direction[:, 1][:, None]   # (R,1)

    # Segment params broadcast to (1,M).
    x0 = seg[:, 0][None, :]
    y0 = seg[:, 1][None, :]
    x1 = seg[:, 2][None, :]
    y1 = seg[:, 3][None, :]
    ex = (x1 - x0)                   # segment edge vector (1,M)
    ey = (y1 - y0)

    # Solve  o + t d = p + u e   =>   t d - u e = p - o.
    # Matrix [[dx, -ex], [dy, -ey]] @ [t, u] = [x0-ox, y0-oy].
    denom = dx * (-ey) - (-ex) * dy          # (R,M) determinant
    safe_denom = jnp.where(jnp.abs(denom) < _EPS, 1.0, denom)

    rhsx = (x0 - ox)                          # (1,M) broadcasts to (R,M)
    rhsy = (y0 - oy)

    # Cramer's rule.
    t = (rhsx * (-ey) - (-ex) * rhsy) / safe_denom
    u = (dx * rhsy - dy * rhsx) / safe_denom

    parallel = jnp.abs(denom) < _EPS
    valid = (~parallel) & (t >= 0.0) & (t <= max_range) & (u >= 0.0) & (u <= 1.0)

    # Where invalid, push the hit to max_range so it never wins the per-ray min.
    return jnp.where(valid, t, max_range)


def cast_rays(
    origin: jnp.ndarray,    # (2,)
    angles: jnp.ndarray,    # (R,)
    segments: jnp.ndarray,  # (M, 4)
    max_range: float,
) -> jnp.ndarray:
    """Cast ``R`` rays from ``origin`` and return the nearest hit distance per ray.

    Fully vectorized ray-vs-segment intersection: no Python loop over rays or
    segments. Rays that hit nothing within ``max_range`` return ``max_range``.

    Args:
        origin: Ray origin ``(2,)`` in world coords.
        angles: Per-ray absolute heading in radians, shape ``(R,)``.
        segments: Occluder segments ``(M, 4)`` packed ``[x0, y0, x1, y1]``.
            Inactive/degenerate segments may be passed as zero-length (both
            endpoints equal) — they are naturally skipped (treated as parallel /
            never producing a valid ``u in [0,1]`` hit closer than ``max_range``).
        max_range: Maximum ray length; also the "no hit" sentinel distance.

    Returns:
        ``(R,)`` float32 distances to the closest segment per ray.
    """
    origin = jnp.asarray(origin, jnp.float32)
    angles = jnp.asarray(angles, jnp.float32)
    segments = jnp.asarray(segments, jnp.float32)

    direction = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (R,2)

    # No host-side empty guard (CONTRACT §0): with M == 0 the (R, 0) per-segment
    # distance reduces over an empty axis to ``max_range`` (the identity of min,
    # because misses are encoded as ``max_range``). We supply that identity
    # explicitly via ``initial=`` so the empty-axis ``min`` is well-defined.
    dists = _ray_segment_distance(origin, direction, segments, max_range)  # (R,M)
    return jnp.min(dists, axis=1, initial=max_range).astype(jnp.float32)    # (R,)


def _fog_attenuated_range(
    origin: jnp.ndarray,      # (2,)
    angles: jnp.ndarray,      # (R,)
    fog_centers: jnp.ndarray, # (Nf, 2)
    base_range: float,
    fog_radius: float,
    fog_vision_mult: float,
    max_proj: jnp.ndarray | None = None,  # (R,) or None: cap on along-ray projection
) -> jnp.ndarray:
    """Per-ray effective range after fog attenuation.

    # 2.0: fog of war / lighting
    A ray whose forward direction passes through a fog patch has its effective
    sensing range scaled down by ``fog_vision_mult``. We approximate "passes
    through fog" by testing, for each fog center, that:

    * the fog center is **in front of** the observer (along-ray projection
      ``proj > 0``);
    * its **perpendicular** distance to the ray line is within ``fog_radius``;
    * (optionally, when ``max_proj`` is given) the fog lies *between* the observer
      and the target -- ``proj <= max_proj`` -- so a fog patch *behind* the entity
      never spuriously dims it (fixes a fog false-negative).

    This is a cheap, fully-vectorized proxy for true volumetric occlusion (called
    out as a simplification). It is branch-free: an empty fog array (``Nf == 0``)
    makes the per-ray ``any`` reduce to ``False`` (no attenuation).

    Args:
        max_proj: Optional ``(R,)`` per-ray cap on the along-ray fog projection
            (typically the observer->target distance). ``None`` disables the cap.

    Returns:
        ``(R,)`` per-ray effective max range.
    """
    direction = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (R,2)

    # Vector from origin to each fog center: (R,Nf,2). Broadcast rays x fogs.
    rel = fog_centers[None, :, :] - origin[None, None, :]   # (1,Nf,2)

    d = direction[:, None, :]                                # (R,1,2)
    # Projection length of fog center onto ray direction (signed; >0 = in front).
    proj = jnp.sum(rel * d, axis=-1)                         # (R,Nf)
    # Perpendicular (line) distance from fog center to the ray line.
    closest = origin[None, None, :] + proj[..., None] * d    # (R,Nf,2)
    perp = jnp.linalg.norm(fog_centers[None, :, :] - closest, axis=-1)  # (R,Nf)

    intersects = (proj > 0.0) & (perp <= fog_radius)         # (R,Nf)
    if max_proj is not None:
        # Only fog patches BETWEEN observer and target attenuate (no behind-fog
        # false-negatives). Branch is host-side on a Python ``None`` flag, not on
        # a traced value -- safe per CONTRACT §0.
        intersects = intersects & (proj <= max_proj[:, None])

    # ``initial=False`` makes the empty-fog (Nf == 0) reduction well-defined.
    ray_in_fog = jnp.any(intersects, axis=1, initial=False)  # (R,)

    return jnp.where(
        ray_in_fog,
        base_range * fog_vision_mult,
        base_range,
    ).astype(jnp.float32)


def lidar_scan(
    agent_pos: jnp.ndarray,       # (2,)
    agent_heading: float,         # scalar (radians)
    walls_segments: jnp.ndarray,  # (M, 4)
    fog_centers: jnp.ndarray,     # (Nf, 2)
    cfg: Any,
) -> jnp.ndarray:
    """Emit a fan of lidar rays from an agent and return clipped hit distances.

    The ray fan spans the agent's vision cone, centered on ``agent_heading``, with
    ``cfg.lidar_n_rays`` rays. Distances are clipped to the (possibly
    fog-attenuated) per-ray effective range.

    Args:
        agent_pos: Agent (x, y) position ``(2,)``.
        agent_heading: Agent facing angle in radians.
        walls_segments: Occluder segments ``(M, 4)``.
        fog_centers: Fog patch centers ``(Nf, 2)``.
        cfg: An ``EnvConfig`` providing ``lidar_n_rays``, ``lidar_range``,
            ``vision_cone_deg``, ``fog_radius``, ``fog_vision_mult``.

    Returns:
        ``(cfg.lidar_n_rays,)`` float32 hit distances (== effective range on miss).
    """
    agent_pos = jnp.asarray(agent_pos, jnp.float32)
    cone = jnp.deg2rad(cfg.vision_cone_deg)

    # Symmetric fan across the cone, centered on heading. ``lidar_n_rays`` is a
    # static Python int from config; pass it straight to ``linspace`` (no silent
    # float->int cast).
    offsets = jnp.linspace(-cone / 2.0, cone / 2.0, cfg.lidar_n_rays)
    angles = agent_heading + offsets                                # (R,)

    raw = cast_rays(agent_pos, angles, walls_segments, cfg.lidar_range)  # (R,)

    # 2.0: fog of war / lighting -- attenuate per-ray range, then re-clip.
    eff_range = _fog_attenuated_range(
        agent_pos, angles, jnp.asarray(fog_centers, jnp.float32),
        cfg.lidar_range, cfg.fog_radius, cfg.fog_vision_mult,
    )
    return jnp.minimum(raw, eff_range).astype(jnp.float32)


def in_vision_cone(
    observer_pos: jnp.ndarray,    # (2,)
    observer_heading: float,      # scalar (radians)
    target_pos: jnp.ndarray,      # (..., 2)
    cfg: Any,
) -> jnp.ndarray:
    """Whether each target lies within the observer's forward vision cone.

    Args:
        observer_pos: Observer (x, y) ``(2,)``.
        observer_heading: Observer facing angle (radians).
        target_pos: Target position(s); trailing axis size 2, any leading shape.
        cfg: ``EnvConfig`` providing ``vision_cone_deg``.

    Returns:
        Boolean array of the target's leading shape: True where the angle between
        the observer's heading and the observer->target direction is within half
        the cone. A target exactly at the observer is treated as in-cone.
    """
    observer_pos = jnp.asarray(observer_pos, jnp.float32)
    target_pos = jnp.asarray(target_pos, jnp.float32)

    rel = target_pos - observer_pos                        # (...,2)
    target_ang = jnp.arctan2(rel[..., 1], rel[..., 0])     # (...,)

    # Smallest signed angular difference, wrapped to [-pi, pi].
    diff = jnp.arctan2(
        jnp.sin(target_ang - observer_heading),
        jnp.cos(target_ang - observer_heading),
    )
    half_cone = jnp.deg2rad(cfg.vision_cone_deg) / 2.0

    dist = jnp.linalg.norm(rel, axis=-1)
    at_observer = dist < _EPS                              # degenerate => in-cone
    return (jnp.abs(diff) <= half_cone) | at_observer


def _segment_blocks_los(
    observer_pos: jnp.ndarray,    # (2,)
    target_pos: jnp.ndarray,      # (E, 2)
    walls_segments: jnp.ndarray,  # (M, 4)
    cfg: Any,
) -> jnp.ndarray:
    """Whether the segment observer->target is blocked by any wall, per target.

    Casts one ray per target toward that target and checks whether any wall is hit
    *before* the target distance. Fully vectorized over the ``E`` targets and ``M``
    walls. Private (not part of the CONTRACT §1 public API).

    Args:
        observer_pos: Observer (x, y) ``(2,)``.
        target_pos: Target positions ``(E, 2)``.
        walls_segments: Wall segments ``(M, 4)``.
        cfg: ``EnvConfig`` providing ``arena_size`` (for a static cast sentinel).

    Returns:
        ``(E,)`` boolean array, True where line-of-sight to that target is blocked.
    """
    observer_pos = jnp.asarray(observer_pos, jnp.float32)
    target_pos = jnp.asarray(target_pos, jnp.float32)
    walls_segments = jnp.asarray(walls_segments, jnp.float32)

    rel = target_pos - observer_pos                        # (E,2)
    dist = jnp.linalg.norm(rel, axis=-1)                   # (E,)
    angles = jnp.arctan2(rel[:, 1], rel[:, 0])             # (E,)

    direction = jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)  # (E,2)
    # STATIC cast sentinel (CONTRACT §0): a fixed, non-traced upper bound on any
    # in-arena distance. Using a *traced* ``jnp.max(dist)+1`` sentinel could shrink
    # toward 0 when the observer ~ entity, spuriously turning a near-zero ``hit``
    # into a block; the static ``2 * arena_size`` avoids that. Misses are encoded
    # as this sentinel and are filtered out below by ``hit < dist`` anyway. With
    # M == 0 the per-segment ``any`` reduces (initial=False) to "not blocked".
    max_range = float(cfg.arena_size) * 2.0
    hit = _ray_segment_distance(
        observer_pos, direction, walls_segments, max_range
    )  # (E,M)

    # A wall blocks if its hit distance is strictly less than the target distance
    # (minus a small epsilon so a wall *at* the target doesn't self-occlude).
    blocked = hit < (dist[:, None] - _EPS)                 # (E,M)
    return jnp.any(blocked, axis=1, initial=False)


def compute_visibility(
    observer_pos: jnp.ndarray,     # (2,)
    observer_heading: float,       # scalar (radians)
    entity_pos: jnp.ndarray,       # (E, 2)
    entity_active: jnp.ndarray,    # (E,) bool
    walls_segments: jnp.ndarray,   # (M, 4)
    fog_centers: jnp.ndarray,      # (Nf, 2)
    cfg: Any,
) -> jnp.ndarray:
    """Per-entity boolean visibility mask for one observer.

    Combines, as a logical AND:
      1. ``entity_active``            -- the entity exists this episode,
      2. within (fog-attenuated) ``vision_range`` of the observer,
      3. within the observer's ``vision_cone``,
      4. line-of-sight not blocked by any wall segment.

    # 2.0: fog of war / lighting -- entities whose line to the observer passes
    through a fog patch have the effective vision range scaled by
    ``fog_vision_mult`` (so fog both hides distant entities and shortens LOS).

    Args:
        observer_pos: Observer (x, y) ``(2,)``.
        observer_heading: Observer facing angle (radians).
        entity_pos: Entity positions ``(E, 2)``.
        entity_active: Existence mask ``(E,)``.
        walls_segments: Wall segments ``(M, 4)``.
        fog_centers: Fog centers ``(Nf, 2)``.
        cfg: ``EnvConfig`` with ``vision_range``, ``vision_cone_deg``,
            ``fog_radius``, ``fog_vision_mult``.

    Returns:
        ``(E,)`` boolean visibility mask. Designed to be ``vmap``-mapped over
        observers to produce the ``(A, E)`` ``entity_mask`` of the obs contract.
    """
    observer_pos = jnp.asarray(observer_pos, jnp.float32)
    entity_pos = jnp.asarray(entity_pos, jnp.float32)
    entity_active = jnp.asarray(entity_active, dtype=bool)
    fog_centers = jnp.asarray(fog_centers, jnp.float32)

    rel = entity_pos - observer_pos                        # (E,2)
    dist = jnp.linalg.norm(rel, axis=-1)                   # (E,)
    angles = jnp.arctan2(rel[:, 1], rel[:, 0])             # (E,)

    # 2.0: fog of war / lighting -- per-entity effective vision range. We reuse
    # the ray fog model: cast a single "ray" toward each entity and shrink range
    # if that bearing passes through fog. ``max_proj=dist`` restricts attenuation
    # to fog patches that lie BETWEEN the observer and the entity (proj>0 AND
    # proj<=dist_to_entity AND perp<=fog_radius) -- a fog patch behind the entity
    # must not dim it (fixes a fog false-negative).
    eff_range = _fog_attenuated_range(
        observer_pos, angles, fog_centers,
        cfg.vision_range, cfg.fog_radius, cfg.fog_vision_mult,
        max_proj=dist,
    )  # (E,)

    in_range = dist <= eff_range
    in_cone = in_vision_cone(observer_pos, observer_heading, entity_pos, cfg)
    not_blocked = ~_segment_blocks_los(observer_pos, entity_pos, walls_segments, cfg)

    return entity_active & in_range & in_cone & not_blocked


def walls_to_segments(physics: Any, cfg: Any) -> jnp.ndarray:
    """Pack the active wall/door entities into occluder line segments.

    # 2.0: destructible env / interactable env
    Builds the ``(M, 4)`` occluder array consumed by :func:`compute_visibility`
    (packed ``[x0, y0, x1, y1]`` per row), where ``M = cfg.n_walls_max +
    cfg.n_doors_max``. Walls and doors occupy a fixed, contiguous block of entity
    rows (agents-first layout; see ``envs/procedural.entity_slot_ranges``), so we
    slice that block statically -- no host branching on traced values.

    Each entity becomes a segment centered on its planar position, oriented along
    its ``heading``, with half-length equal to its ``size`` (the half-extent used
    for collision)::

        endpoints = pos[:2] ± size * [cos(heading), sin(heading)]

    An **inactive** wall/door (``physics.active == False`` -- e.g. an *opened* door
    or a *broken* wall) is emitted as a **degenerate, zero-length** segment (both
    endpoints collapsed onto the center). A zero-length segment has a ~0 edge
    vector, so it is caught by the parallel/zero-denominator mask in
    :func:`_ray_segment_distance` and occludes nothing. This is the branch-free
    mechanism by which opened doors / broken walls stop blocking line-of-sight.

    Args:
        physics: ``PhysicsState`` with ``pos (E,3)``, ``heading (E,)``,
            ``size (E,)`` and ``active (E,)``.
        cfg: ``EnvConfig`` providing ``max_agents``, ``n_boxes_max``,
            ``n_ramps_max``, ``n_decoys_max``, ``n_walls_max``, ``n_doors_max``.

    Returns:
        ``(M, 4)`` float32 segment array, ``M = cfg.n_walls_max + cfg.n_doors_max``.
    """
    # Static (host-side) slot arithmetic -- mirrors entity_slot_ranges(); walls and
    # doors form one contiguous [start, stop) block of entity rows.
    start = (
        int(cfg.max_agents)
        + int(cfg.n_boxes_max)
        + int(cfg.n_ramps_max)
        + int(cfg.n_decoys_max)
    )
    stop = start + int(cfg.n_walls_max) + int(cfg.n_doors_max)

    pos = jnp.asarray(physics.pos[start:stop, :2], jnp.float32)   # (M,2)
    heading = jnp.asarray(physics.heading[start:stop], jnp.float32)  # (M,)
    size = jnp.asarray(physics.size[start:stop], jnp.float32)     # (M,)
    active = jnp.asarray(physics.active[start:stop], dtype=bool)  # (M,)

    # Half-length vector along the entity heading; zeroed when inactive so the
    # segment collapses to a single point (occludes nothing).
    half_len = jnp.where(active, size, 0.0)                       # (M,)
    ux = jnp.cos(heading) * half_len                              # (M,)
    uy = jnp.sin(heading) * half_len                              # (M,)

    x0 = pos[:, 0] - ux
    y0 = pos[:, 1] - uy
    x1 = pos[:, 0] + ux
    y1 = pos[:, 1] + uy
    return jnp.stack([x0, y0, x1, y1], axis=-1).astype(jnp.float32)  # (M,4)


def compute_visibility_batch(
    physics: Any,
    game: Any,
    observer_ids: jnp.ndarray,  # (A,) int -- agent row indices (== entity ids)
    cfg: Any,
) -> jnp.ndarray:
    """Per-observer, per-entity visibility mask ``(A, E)`` (CONTRACT §3 ``entity_mask``).

    This is the env-facing batch entry point: it pulls observer/entity geometry
    straight out of the ``PhysicsState`` / ``GameState`` and ``vmap``s the
    single-observer :func:`compute_visibility` over the ``A`` observers.

    Because the layout is **agents-first**, an agent's row index equals its entity
    id, so ``observer_ids`` indexes both ``physics.pos`` (its own position) and the
    entity axis it is observing.

    Args:
        physics: ``PhysicsState`` with ``pos (E,3)``, ``heading (E,)``,
            ``active (E,)``.
        game: ``GameState`` with ``fog_pos (n_fog_patches, 2)``.
        observer_ids: int array of agent row indices ``(A,)`` (``0..A-1``).
        cfg: ``EnvConfig`` (vision range/cone, fog, arena size, wall/door counts).

    Returns:
        ``(A, E)`` boolean visibility mask: for each observer, which entities are
        active AND within fog-attenuated range AND in the vision cone AND not
        occluded by an active wall/door segment.
    """
    observer_ids = jnp.asarray(observer_ids, jnp.int32)

    observer_pos = jnp.asarray(physics.pos[observer_ids, :2], jnp.float32)   # (A,2)
    observer_heading = jnp.asarray(physics.heading[observer_ids], jnp.float32)  # (A,)
    entity_pos = jnp.asarray(physics.pos[:, :2], jnp.float32)                # (E,2)
    entity_active = jnp.asarray(physics.active, dtype=bool)                  # (E,)
    walls = walls_to_segments(physics, cfg)                                  # (M,4)
    fog = jnp.asarray(game.fog_pos, jnp.float32)                             # (Nf,2)

    # vmap the single-observer kernel over the A observers; entity tensors, walls,
    # fog and cfg are shared (in_axes=None).
    return jax.vmap(
        compute_visibility, in_axes=(0, 0, None, None, None, None, None)
    )(observer_pos, observer_heading, entity_pos, entity_active, walls, fog, cfg)
