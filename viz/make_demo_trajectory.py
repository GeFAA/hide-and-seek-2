"""
viz/make_demo_trajectory.py -- SYNTHETIC Hide & Seek 2.0 episode generator.

PURE STDLIB ONLY (``math``, ``json``, ``random``, ``argparse``, ``os``) -- importing
or running this module never touches jax or numpy, so the 3D viewer has a clean
SET OF NAMED SCENARIOS to play the moment the repo is cloned, on any machine.

What it produces
----------------
A whole *family* of ``hns2-traj`` v1 documents (see :mod:`viz.schema`) -- one per
named scenario, in the spirit of OpenAI's hide-and-seek clips -- plus a
``manifest.json`` the viewer reads to populate its scenario menu. Every scenario is
a tidy ~190-220 frame episode inside the SAME **clean square arena**: four
continuous border walls (north / south / east + a split west wall with a central
DOOR gap) and -- in most scenarios -- a breakable interior wall. Each wall carries
a real ``heading`` (h) so the viewer orients it; walls render length = half_len*2.

Scenarios (manifest ids)
------------------------
* ``showcase`` -- all 2.0 mechanics in one run (coop heavy-box shove, decoy pulse,
  ramp climb with z up & gr->0, a wall breaks, the door opens, a hider is spotted).
* ``fort``     -- two hiders fetch every light box AND coop-shove the heavy box into
  a tight L-shaped corner barricade, then LOCK them (lk=1); seekers are blocked.
* ``ramp``     -- hiders fort up early; a seeker drags the ramp (hd/hb on the ramp)
  to the barricade and CLIMBS it (z->~1.4, gr->0 on top), drops inside, late spot.
* ``chase``    -- an OPEN arena (only the 4 borders + door, no interior wall, no
  boxes): 2 seekers chase 2 weaving, dodging hiders with sprint bursts (stamina dips).
* ``doors``    -- hiders coop-push the HEAVY box into the west DOOR gap and jam it
  shut for the whole episode (door stays closed, a=1); seekers pile up outside.
* ``decoy``    -- hiders flee one way while a DECOY pulses on the OPPOSITE side and
  the two seekers veer toward it (their paths bend to the decoy); no hider spotted.

Design
------
Every scenario reuses the proven clean-arena building blocks: the
:func:`clamp_arena` / :func:`heading_from` helpers, the eased-waypoint
:class:`AgentPlan`, drifting :func:`fog_at`, and a single generic
:func:`build_trajectory` that turns a declarative :class:`Scenario` (entity table,
agent plans + small per-frame callbacks) into a validated document. ``E`` is fixed
per scenario (padded with inert walls) so each file has a stable slot layout.

CLI
---
* ``python -m viz.make_demo_trajectory``            -> build ALL scenarios + manifest.
* ``python -m viz.make_demo_trajectory --scenario fort`` -> (re)build just one.
* ``python -m viz.make_demo_trajectory --list``     -> list scenario ids/titles.
``demo_trajectory.json`` is always written as a copy of ``showcase.json`` so old
links / docs keep working.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Callable, Dict, List, Optional, Tuple

from viz import schema

# --------------------------------------------------------------------------- #
# Fixed arena geometry / tuning (kept independent of config.py: stdlib promise)
# --------------------------------------------------------------------------- #
ARENA_SIZE = 12.0          # arena spans [-6, +6] in x and y
HALF = ARENA_SIZE / 2.0
DT = 0.1                   # seconds per control step (playback speed)
PI2 = math.pi / 2.0

# Per-type collision radius / half-extent and mass (cosmetic; drives the viewer).
SIZE = {
    "hider": 0.40, "seeker": 0.40,
    "box_light": 0.55, "box_heavy": 0.80,
    "ramp": 0.95, "decoy": 0.35,
    "wall": 1.20, "door": 1.20,
}
MASS = {
    "hider": 1.0, "seeker": 1.0,
    "box_light": 1.0, "box_heavy": 6.0,
    "ramp": 3.0, "decoy": 0.5,
    "wall": 1.0e6, "door": 1.0e6,
}

# Movement / stamina tuning shared by every scenario.
SPRINT_DRAIN = 0.020
STAMINA_REGEN = 0.010
STAMINA_FLOOR = 0.05


# --------------------------------------------------------------------------- #
# Small stdlib math helpers (no numpy). REUSED by every scenario.
# --------------------------------------------------------------------------- #
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def clamp_arena(x: float, y: float, margin: float = 0.75) -> Tuple[float, float]:
    """Clamp an (x, y) point inside the playable arena, keeping off the walls."""
    lim = HALF - margin
    return clamp(x, -lim, lim), clamp(y, -lim, lim)


def smoothstep(a: float, b: float, t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)
    return a + (b - a) * s


def heading_from(dx: float, dy: float, fallback: float) -> float:
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return fallback
    return math.atan2(dy, dx)


def phase_for(t: int, prep_steps: int) -> str:
    return "prep" if t < prep_steps else "main"


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


# --------------------------------------------------------------------------- #
# Per-agent scripted, eased waypoint plan. REUSED by every scenario.
# --------------------------------------------------------------------------- #
class AgentPlan:
    """A scripted, eased waypoint plan for one agent.

    Each leg is ``(t0, t1, (x, y), carry_id, sprint)`` and the agent eases from the
    previous leg's target to this leg's target across ``[t0, t1)``.
    """

    def __init__(self, start: Tuple[float, float]):
        self.start = start
        self.legs: List[Tuple[int, int, Tuple[float, float], int, bool]] = []

    def add(self, t0: int, t1: int, target: Tuple[float, float],
            carry: int = -1, sprint: bool = False) -> "AgentPlan":
        self.legs.append((t0, t1, target, carry, sprint))
        return self

    def hold(self, t0: int, t1: int, carry: int = -1) -> "AgentPlan":
        """Stay put (at the previous leg's target) across ``[t0, t1)``."""
        target = self.legs[-1][2] if self.legs else self.start
        self.legs.append((t0, t1, target, carry, False))
        return self

    def _leg_origin(self, idx: int) -> Tuple[float, float]:
        return self.start if idx == 0 else self.legs[idx - 1][2]

    def sample(self, t: int) -> Tuple[float, float, int, bool, bool]:
        """Resolve ``(x, y, carry_id, sprint, moving)`` at step ``t``."""
        if not self.legs or t < self.legs[0][0]:
            return self.start[0], self.start[1], -1, False, False
        for idx, (t0, t1, target, carry, sprint) in enumerate(self.legs):
            if t0 <= t < t1:
                ox, oy = self._leg_origin(idx)
                frac = (t - t0) / max(1, (t1 - t0))
                x = smoothstep(ox, target[0], frac)
                y = smoothstep(oy, target[1], frac)
                moving = frac < 0.999
                return x, y, carry, sprint and moving, moving
        last = self.legs[-1]
        return last[2][0], last[2][1], -1, False, False

    def pos_at(self, t: int) -> Tuple[float, float]:
        x, y, _, _, _ = self.sample(t)
        return x, y


# --------------------------------------------------------------------------- #
# Fog patches that slowly drift. REUSED by every scenario.
# --------------------------------------------------------------------------- #
_DEFAULT_FOG_SPECS = (
    (-3.0, 1.0, 0.9, 0.6, 0.0),
    (2.5, 2.0, -0.8, 0.7, 1.7),
    (0.5, -3.0, 0.7, -0.9, 3.1),
)


def fog_at(t: int, steps: int,
           specs: Tuple[Tuple[float, float, float, float, float], ...] = _DEFAULT_FOG_SPECS,
           base_r: float = 2.3) -> List[List[float]]:
    """Return drifting fog ``[x, y, r]`` triples at step ``t`` (deterministic)."""
    frac = t / max(1, steps - 1)
    patches: List[List[float]] = []
    for (cx, cy, ax, ay, ph) in specs:
        ang = 2.0 * math.pi * frac + ph
        x = cx + ax * math.sin(ang)
        y = cy + ay * math.cos(ang * 0.8 + ph)
        r = base_r + 0.4 * math.sin(ang * 1.3 + ph)
        x, y = clamp_arena(x, y, margin=0.0)
        patches.append([x, y, max(1.2, r)])
    return patches


# --------------------------------------------------------------------------- #
# Clean square arena: the four continuous border walls + a central west DOOR.
# Each border half-length slightly exceeds HALF so the corners overlap cleanly.
# Returned as id -> (x, y, heading, half_len); length rendered = half_len*2.
# --------------------------------------------------------------------------- #
def border_walls(wall_n: int, wall_e: int, wall_s: int,
                 wall_wt: int, wall_wb: int) -> Dict[int, Tuple[float, float, float, float]]:
    return {
        wall_n:  (0.0,  6.0, 0.0, 6.25),   # north
        wall_s:  (0.0, -6.0, 0.0, 6.25),   # south
        wall_e:  (6.0,  0.0, PI2, 6.25),   # east
        wall_wt: (-6.0,  3.4, PI2, 2.45),  # west wall, upper segment (above the door)
        wall_wb: (-6.0, -3.4, PI2, 2.45),  # west wall, lower segment (below the door)
    }


# West door fills the central gap in the split west wall.
DOOR_GEOM: Tuple[float, float, float, float] = (-6.0, 0.0, PI2, 1.15)


# --------------------------------------------------------------------------- #
# Declarative scenario description consumed by the generic builder.
# --------------------------------------------------------------------------- #
class PropState:
    """Per-frame dynamic state for a single non-agent, non-wall prop (box/ramp/decoy)."""

    __slots__ = ("x", "y", "z", "h", "a", "lk", "hd", "hb", "no", "dc", "gr")

    def __init__(self, x: float, y: float, z: float = 0.0, h: float = 0.0,
                 a: int = 1, lk: int = 0, hd: int = 0, hb: int = -1,
                 no: float = 0.0, dc: int = 0, gr: int = 1):
        self.x = x; self.y = y; self.z = z; self.h = h
        self.a = a; self.lk = lk; self.hd = hd; self.hb = hb
        self.no = no; self.dc = dc; self.gr = gr


class Scenario:
    """Everything the generic :func:`build_trajectory` needs to emit one episode.

    The heavy machinery (agent plans -> eased motion, velocity headings, stamina,
    walls/door, padding, fog, frame assembly + score) lives in the builder. A
    scenario only supplies its entity table, agent plans, and a few small per-frame
    callbacks; ``ctx`` (a dict the builder fills each frame) gives those callbacks
    everything they need:
        ``ctx = {t, phase, is_main, prep, steps, agent_state, carried_by, scenario}``
    where ``agent_state[aid] = {x, y, h, sprint, moving, st}``.
    """

    def __init__(
        self,
        sid: str,
        title: str,
        steps: int,
        prep: int,
        seed: int,
        entity_spec: List[Tuple[str, int, bool]],
        walls: Dict[int, Tuple[float, float, float, float]],
        door_id: Optional[int],
        plans: Dict[int, AgentPlan],
        hider_ids: Tuple[int, ...],
        seeker_ids: Tuple[int, ...],
        total_entities: int,
        prop_dynamics: Callable[[Dict], Dict[int, PropState]],
        score_fn: Callable[[Dict], Tuple[float, float, bool]],
        seen_fn: Callable[[int, Dict], int],
        agent_overrides: Optional[Callable[[int, Dict], Dict]] = None,
        wall_active: Optional[Callable[[int, Dict], int]] = None,
        door_active: Optional[Callable[[Dict], int]] = None,
        seekers_locked_in_prep: bool = True,
        fog_specs: Tuple[Tuple[float, float, float, float, float], ...] = _DEFAULT_FOG_SPECS,
    ):
        self.sid = sid
        self.title = title
        self.steps = steps
        self.prep = prep
        self.seed = seed
        self.entity_spec = entity_spec
        self.walls = walls
        self.door_id = door_id
        self.plans = plans
        self.hider_ids = hider_ids
        self.seeker_ids = seeker_ids
        self.total_entities = total_entities
        self.prop_dynamics = prop_dynamics
        self.score_fn = score_fn
        self.seen_fn = seen_fn
        self.agent_overrides = agent_overrides
        self.wall_active = wall_active
        self.door_active = door_active
        self.seekers_locked_in_prep = seekers_locked_in_prep
        self.fog_specs = fog_specs

    @property
    def agent_ids(self) -> Tuple[int, ...]:
        return tuple(self.hider_ids) + tuple(self.seeker_ids)


# --------------------------------------------------------------------------- #
# Static entity table (id == index), padded to total_entities with inert walls.
# --------------------------------------------------------------------------- #
def build_entities(scn: Scenario) -> List[Dict]:
    entities: List[Dict] = []
    for i, (etype, team, is_decoy) in enumerate(scn.entity_spec):
        size = SIZE[etype]
        if i in scn.walls:
            size = scn.walls[i][3]
        elif scn.door_id is not None and i == scn.door_id:
            size = DOOR_GEOM[3]
        entities.append(schema.make_entity_meta(
            id=i, type=etype, team=team, size=size, mass=MASS[etype], is_decoy=is_decoy,
        ))
    for i in range(len(scn.entity_spec), scn.total_entities):  # inert padding walls
        entities.append(schema.make_entity_meta(
            id=i, type="wall", team=schema.TEAM_NONE,
            size=SIZE["wall"], mass=MASS["wall"], is_decoy=False,
        ))
    return entities


# --------------------------------------------------------------------------- #
# Generic episode builder: turns a Scenario into a validated hns2-traj document.
# --------------------------------------------------------------------------- #
def build_trajectory(scn: Scenario) -> Dict:
    """Build and return a (validated-on-save) ``hns2-traj`` v1 document."""
    random.Random(scn.seed)  # reserved for future jitter; choreography is fixed
    entities = build_entities(scn)
    E = len(entities)
    steps, prep = scn.steps, scn.prep

    # Per-agent heading memory + previous position (so headings follow velocity).
    headings: Dict[int, float] = {}
    prev_pos: Dict[int, Tuple[float, float]] = {}
    stamina: Dict[int, float] = {}
    for aid in scn.agent_ids:
        sx, sy = scn.plans[aid].start
        headings[aid] = math.pi if aid in scn.seeker_ids else 0.0
        prev_pos[aid] = (sx, sy)
        stamina[aid] = 1.0

    frames: List[Dict] = []
    for t in range(steps):
        phase = phase_for(t, prep)
        is_main = phase == "main"

        # --- agents: eased motion, velocity heading, stamina drain on sprint ---
        agent_state: Dict[int, Dict] = {}
        carried_by: Dict[int, int] = {}
        for aid in scn.agent_ids:
            plan = scn.plans[aid]
            x, y, carry, sprint, moving = plan.sample(t)
            x, y = clamp_arena(x, y)
            px, py = prev_pos[aid]
            headings[aid] = heading_from(x - px, y - py, headings[aid])
            prev_pos[aid] = (x, y)
            if sprint and moving:
                stamina[aid] = clamp(stamina[aid] - SPRINT_DRAIN, STAMINA_FLOOR, 1.0)
            else:
                stamina[aid] = clamp(stamina[aid] + STAMINA_REGEN, STAMINA_FLOOR, 1.0)
            agent_state[aid] = {"x": x, "y": y, "h": headings[aid],
                                "sprint": sprint, "moving": moving, "st": stamina[aid]}
            if carry >= 0:
                carried_by[carry] = aid

        ctx = {
            "t": t, "phase": phase, "is_main": is_main, "prep": prep, "steps": steps,
            "agent_state": agent_state, "carried_by": carried_by, "scenario": scn,
        }

        sh, ss, seen_any = scn.score_fn(ctx)
        props = scn.prop_dynamics(ctx)
        fog = fog_at(t, steps, scn.fog_specs)
        seekers_frozen = scn.seekers_locked_in_prep and not is_main

        ent: List[Dict] = []
        for eid in range(E):
            etype = entities[eid]["type"]

            # ---- agents (hiders + seekers) ----
            if eid in scn.agent_ids:
                st = agent_state[eid]
                is_seeker = eid in scn.seeker_ids
                z, grounded, locked = 0.0, 1, 0
                if is_seeker and seekers_frozen:
                    locked = 1
                no = 0.25 if (st["sprint"] and st["moving"]) else 0.05
                sn = scn.seen_fn(eid, ctx)
                hd = 0
                hb = -1
                if scn.agent_overrides is not None:
                    ov = scn.agent_overrides(eid, ctx)
                    z = ov.get("z", z)
                    grounded = ov.get("gr", grounded)
                    locked = ov.get("lk", locked)
                    no = ov.get("no", no)
                    sn = ov.get("sn", sn)
                    hd = ov.get("hd", hd)
                    hb = ov.get("hb", hb)
                ent.append(schema.make_frame_ent(
                    id=eid, x=st["x"], y=st["y"], z=z, h=st["h"],
                    a=1, lk=locked, hd=hd, hb=hb, no=no, dc=0, gr=grounded,
                    st=st["st"], sn=sn,
                ))
                continue

            # ---- dynamic props (boxes / ramp / decoy), supplied per frame ----
            if eid in props:
                ps = props[eid]
                ent.append(schema.make_frame_ent(
                    id=eid, x=ps.x, y=ps.y, z=ps.z, h=ps.h,
                    a=ps.a, lk=ps.lk, hd=ps.hd, hb=ps.hb,
                    no=ps.no, dc=ps.dc, gr=ps.gr, st=-1.0, sn=0,
                ))
                continue

            # ---- walls ----
            if eid in scn.walls:
                wx, wy, wh, _ = scn.walls[eid]
                active = scn.wall_active(eid, ctx) if scn.wall_active is not None else 1
                ent.append(schema.make_frame_ent(
                    id=eid, x=wx, y=wy, z=0.0, h=wh,
                    a=active, lk=1, hd=0, hb=-1, no=0.0, dc=0, gr=1, st=-1.0, sn=0,
                ))
                continue

            # ---- door ----
            if scn.door_id is not None and eid == scn.door_id:
                dx, dy, dh, _ = DOOR_GEOM
                active = scn.door_active(ctx) if scn.door_active is not None else 1
                ent.append(schema.make_frame_ent(
                    id=eid, x=dx, y=dy, z=0.0, h=dh,
                    a=active, lk=1, hd=0, hb=-1, no=0.0, dc=0, gr=1, st=-1.0, sn=0,
                ))
                continue

            # ---- inert padding walls: parked outside the arena, never active ----
            ent.append(schema.make_frame_ent(
                id=eid, x=HALF + 4.0, y=HALF + 4.0, z=0.0, h=0.0,
                a=0, lk=1, hd=0, hb=-1, no=0.0, dc=0, gr=1, st=-1.0, sn=0,
            ))

        frames.append(schema.make_frame(
            t=t, phase=phase, sh=sh, ss=ss,
            seen_any=bool(seen_any), fog=fog, ent=ent,
        ))

    meta = {
        "title": scn.title,
        "seed": int(scn.seed),
        "arena_size": ARENA_SIZE,
        "dt": DT,
        "max_steps": int(steps),
        "prep_steps": int(prep),
        "entity_types": list(schema.ENTITY_TYPES),
        "max_agents": 6,
        "max_entities": E,
    }
    return schema.make_trajectory(meta, entities, frames)


# --------------------------------------------------------------------------- #
# Shared scoring / helper factories used by several scenarios.
# --------------------------------------------------------------------------- #
def hiders_win_score(ctx: Dict) -> Tuple[float, float, bool]:
    """Hiders steadily gain in main while never spotted (no seen_any ever)."""
    if ctx["is_main"]:
        return float(ctx["t"] - ctx["prep"] + 1), 0.0, False
    return 0.0, 0.0, False


def make_late_spot_score(seen_step: int) -> Callable[[Dict], Tuple[float, float, bool]]:
    """Hiders gain until ``seen_step``; once spotted the seekers swing the score."""

    def _score(ctx: Dict) -> Tuple[float, float, bool]:
        if not ctx["is_main"]:
            return 0.0, 0.0, False
        t = ctx["t"]
        if t >= seen_step:
            seen_frames = t - seen_step + 1
            base = seen_step - ctx["prep"]
            return float(base), float(2.0 * seen_frames), True
        return float(t - ctx["prep"] + 1), 0.0, False

    return _score


def decoy_noise_at(since: int) -> float:
    """Pulsing decoy noise that ramps up then keeps pulsing."""
    return clamp(0.6 * clamp(since / 12.0, 0.0, 1.0)
                 + 0.15 * (0.5 + 0.5 * math.sin(since * 0.7)), 0.0, 1.0)


# =========================================================================== #
# SCENARIO 1: showcase -- all 2.0 mechanics in one run.
# =========================================================================== #
def scenario_showcase() -> Scenario:
    steps, prep, seed = 220, 96, 7
    # Entity table: agents, props, walls/door, padded to E.
    spec: List[Tuple[str, int, bool]] = [
        ("hider", schema.TEAM_HIDER, False),    # 0 hider A
        ("hider", schema.TEAM_HIDER, False),    # 1 hider B
        ("seeker", schema.TEAM_SEEKER, False),  # 2 seeker A
        ("seeker", schema.TEAM_SEEKER, False),  # 3 seeker B
        ("box_light", schema.TEAM_NONE, False),  # 4 light box 1
        ("box_light", schema.TEAM_NONE, False),  # 5 light box 2
        ("box_light", schema.TEAM_NONE, False),  # 6 light box 3
        ("box_heavy", schema.TEAM_NONE, False),  # 7 heavy box (coop)
        ("ramp", schema.TEAM_NONE, False),       # 8 ramp
        ("decoy", schema.TEAM_NONE, True),       # 9 decoy (true decoy identity)
        ("wall", schema.TEAM_NONE, False),       # 10 border N
        ("wall", schema.TEAM_NONE, False),       # 11 border E
        ("wall", schema.TEAM_NONE, False),       # 12 border S
        ("wall", schema.TEAM_NONE, False),       # 13 interior fragile wall (breaks)
        ("door", schema.TEAM_NONE, False),       # 14 door (opens mid-main)
        ("wall", schema.TEAM_NONE, False),       # 15 border W-top
        ("wall", schema.TEAM_NONE, False),       # 16 border W-bottom
    ]
    HA, HB, SA, SB = 0, 1, 2, 3
    BOX_L1, BOX_L2, BOX_L3, BOX_HEAVY, RAMP, DECOY = 4, 5, 6, 7, 8, 9
    WALL_N, WALL_E, WALL_S, WALL_FRAG, DOOR, WALL_WT, WALL_WB = 10, 11, 12, 13, 14, 15, 16
    E = 18

    walls = border_walls(WALL_N, WALL_E, WALL_S, WALL_WT, WALL_WB)
    walls[WALL_FRAG] = (1.9, 1.1, PI2, 1.70)  # interior wall a seeker rams

    RAMP_POS = (3.9, 2.7)
    DECOY_POS = (-3.0, -3.7)
    BARRICADE = {BOX_L1: (-4.6, 3.0), BOX_L2: (-3.6, 3.5), BOX_L3: (-2.6, 3.0)}
    BOX_START = {BOX_L1: (-4.4, -3.4), BOX_L2: (-1.2, -4.2), BOX_L3: (2.6, -3.2)}
    HEAVY_START = (0.6, -0.4)
    HEAVY_END = (-2.6, 4.2)
    SEEKER_START = {SA: (4.7, -3.6), SB: (4.7, 3.6)}
    HIDER_START = {HA: (-2.4, -1.2), HB: (-3.6, 0.6)}

    # ---- agent plans ----
    pa, pb = AgentPlan(HIDER_START[HA]), AgentPlan(HIDER_START[HB])
    pa.add(4, 16, BOX_START[BOX_L1])
    pa.add(16, 30, BARRICADE[BOX_L1], carry=BOX_L1)
    pa.add(30, 42, BOX_START[BOX_L2])
    pa.add(42, 56, BARRICADE[BOX_L2], carry=BOX_L2)
    pb.add(6, 24, BOX_START[BOX_L3])
    pb.add(24, 40, BARRICADE[BOX_L3], carry=BOX_L3)
    pa.add(56, 66, (HEAVY_START[0] - 1.0, HEAVY_START[1]))
    pb.add(40, 54, (HEAVY_START[0] + 1.0, HEAVY_START[1]))
    pa.add(66, 88, (HEAVY_END[0] - 1.0, HEAVY_END[1]))
    pb.add(66, 88, (HEAVY_END[0] + 1.0, HEAVY_END[1]))
    pa.add(88, prep, (-2.6, 2.2))
    pb.add(88, prep, (-1.2, 1.8))
    pa.add(prep, prep + 44, (-3.2, -0.6), sprint=True)
    pa.add(prep + 44, prep + 92, (-3.6, -2.6), sprint=True)
    pa.add(prep + 92, steps, (-2.6, -3.6))
    pb.add(prep, prep + 38, (2.3, 2.1), sprint=True)
    pb.add(prep + 38, prep + 84, (3.4, -0.6), sprint=True)
    pb.add(prep + 84, steps, (1.2, -2.2), sprint=True)

    sa, sb = AgentPlan(SEEKER_START[SA]), AgentPlan(SEEKER_START[SB])
    sa.add(0, prep, SEEKER_START[SA])
    sb.add(0, prep, SEEKER_START[SB])
    frag = (walls[WALL_FRAG][0], walls[WALL_FRAG][1])
    sa.add(prep, prep + 18, frag, sprint=True)
    sa.add(prep + 18, prep + 62, (-1.6, 0.2), sprint=True)
    sa.add(prep + 62, steps, (-3.9, -1.7), sprint=True)
    sb.add(prep, prep + 20, RAMP_POS, sprint=True)
    sb.add(prep + 20, prep + 40, (RAMP_POS[0] - 0.5, RAMP_POS[1] - 0.7))
    # Trail hider B down the EAST wall, always a clear gap to its east, then cut low.
    sb.add(prep + 40, prep + 70, (4.4, 0.8), sprint=True)
    sb.add(prep + 70, prep + 96, (4.2, -2.0), sprint=True)
    sb.add(prep + 96, steps, (2.4, -2.8), sprint=True)
    plans = {HA: pa, HB: pb, SA: sa, SB: sb}

    decoy_on = prep - 22
    wall_break = prep + 16
    door_open = prep + (steps - prep) // 2
    seen_step = prep + int((steps - prep) * 0.62)
    ramp_up_t0, ramp_up_t1 = prep + 20, prep + 40

    box_pos = {BOX_L1: BOX_START[BOX_L1], BOX_L2: BOX_START[BOX_L2],
               BOX_L3: BOX_START[BOX_L3], BOX_HEAVY: HEAVY_START}

    def props(ctx: Dict) -> Dict[int, PropState]:
        t = ctx["t"]
        ast = ctx["agent_state"]
        carried_by = ctx["carried_by"]
        out: Dict[int, PropState] = {}
        for bid in (BOX_L1, BOX_L2, BOX_L3):
            holder = carried_by.get(bid, -1)
            if holder >= 0:
                box_pos[bid] = (ast[holder]["x"], ast[holder]["y"])
            bx, by = box_pos[bid]
            out[bid] = PropState(bx, by, hd=1 if holder >= 0 else 0, hb=holder)
        if 66 <= t < 88:
            box_pos[BOX_HEAVY] = ((ast[HA]["x"] + ast[HB]["x"]) / 2.0,
                                  (ast[HA]["y"] + ast[HB]["y"]) / 2.0)
        elif t >= 88:
            box_pos[BOX_HEAVY] = HEAVY_END
        hx, hy = box_pos[BOX_HEAVY]
        out[BOX_HEAVY] = PropState(hx, hy)
        out[RAMP] = PropState(RAMP_POS[0], RAMP_POS[1])
        if decoy_on <= t:
            out[DECOY] = PropState(DECOY_POS[0], DECOY_POS[1],
                                   no=decoy_noise_at(t - decoy_on), dc=1)
        else:
            out[DECOY] = PropState(DECOY_POS[0], DECOY_POS[1], no=0.0, dc=0)
        return out

    def agent_ov(eid: int, ctx: Dict) -> Dict:
        t = ctx["t"]
        if eid == SB and ramp_up_t0 <= t < ramp_up_t1:
            mid = (ramp_up_t0 + ramp_up_t1) / 2.0
            climb = ((t - ramp_up_t0) / max(1, mid - ramp_up_t0) if t <= mid
                     else 1.0 - (t - mid) / max(1, ramp_up_t1 - mid))
            z = round(1.2 * clamp(climb, 0.0, 1.0), 4)
            return {"z": z, "gr": 0 if z > 0.05 else 1}
        return {}

    def wall_act(eid: int, ctx: Dict) -> int:
        if eid == WALL_FRAG and ctx["is_main"] and ctx["t"] >= wall_break:
            return 0
        return 1

    def door_act(ctx: Dict) -> int:
        return 0 if (ctx["is_main"] and ctx["t"] >= door_open) else 1

    def seen(eid: int, ctx: Dict) -> int:
        return 1 if (eid == HB and ctx["is_main"] and ctx["t"] >= seen_step) else 0

    return Scenario(
        sid="showcase", title="Synthetic Showcase", steps=steps, prep=prep, seed=seed,
        entity_spec=spec, walls=walls, door_id=DOOR, plans=plans,
        hider_ids=(HA, HB), seeker_ids=(SA, SB), total_entities=E,
        prop_dynamics=props, score_fn=make_late_spot_score(seen_step), seen_fn=seen,
        agent_overrides=agent_ov, wall_active=wall_act, door_active=door_act,
    )


# =========================================================================== #
# SCENARIO 2: fort -- coop fort-building + locking; seekers blocked.
# =========================================================================== #
def scenario_fort() -> Scenario:
    steps, prep, seed = 210, 96, 11
    spec: List[Tuple[str, int, bool]] = [
        ("hider", schema.TEAM_HIDER, False),    # 0
        ("hider", schema.TEAM_HIDER, False),    # 1
        ("seeker", schema.TEAM_SEEKER, False),  # 2
        ("seeker", schema.TEAM_SEEKER, False),  # 3
        ("box_light", schema.TEAM_NONE, False),  # 4
        ("box_light", schema.TEAM_NONE, False),  # 5
        ("box_light", schema.TEAM_NONE, False),  # 6
        ("box_heavy", schema.TEAM_NONE, False),  # 7
        ("wall", schema.TEAM_NONE, False),       # 8 border N
        ("wall", schema.TEAM_NONE, False),       # 9 border E
        ("wall", schema.TEAM_NONE, False),       # 10 border S
        ("door", schema.TEAM_NONE, False),       # 11 door
        ("wall", schema.TEAM_NONE, False),       # 12 border W-top
        ("wall", schema.TEAM_NONE, False),       # 13 border W-bottom
    ]
    HA, HB, SA, SB = 0, 1, 2, 3
    BOX_L1, BOX_L2, BOX_L3, BOX_HEAVY = 4, 5, 6, 7
    WALL_N, WALL_E, WALL_S, DOOR, WALL_WT, WALL_WB = 8, 9, 10, 11, 12, 13
    E = 16

    walls = border_walls(WALL_N, WALL_E, WALL_S, WALL_WT, WALL_WB)

    # Tight L-shaped barricade in the NE corner: heavy box at the corner of the L,
    # three light boxes forming the two short arms.
    HEAVY_END = (4.4, 4.4)                       # corner of the L
    BARRICADE = {BOX_L1: (4.4, 2.9), BOX_L2: (4.4, 1.5),   # vertical arm (down)
                 BOX_L3: (2.9, 4.4)}                       # horizontal arm (left)
    BOX_START = {BOX_L1: (-3.8, -3.4), BOX_L2: (0.4, -4.2), BOX_L3: (-2.0, -3.6)}
    HEAVY_START = (1.2, 1.0)
    SEEKER_START = {SA: (-4.6, -2.0), SB: (-4.6, 2.0)}     # held at the west pad
    HIDER_START = {HA: (1.0, 1.6), HB: (2.2, 0.4)}

    pa, pb = AgentPlan(HIDER_START[HA]), AgentPlan(HIDER_START[HB])
    # Each hider fetches boxes; A: L1 + L3, B: L2, then both coop-shove the heavy box.
    pa.add(4, 16, BOX_START[BOX_L1])
    pa.add(16, 30, BARRICADE[BOX_L1], carry=BOX_L1)
    pa.add(30, 42, BOX_START[BOX_L3])
    pa.add(42, 56, BARRICADE[BOX_L3], carry=BOX_L3)
    pb.add(6, 22, BOX_START[BOX_L2])
    pb.add(22, 38, BARRICADE[BOX_L2], carry=BOX_L2)
    pb.add(38, 52, (HEAVY_START[0], HEAVY_START[1] - 1.1))   # flank below heavy box
    pa.add(56, 66, (HEAVY_START[0] - 1.1, HEAVY_START[1]))   # flank left of heavy box
    # Coop-shove the heavy box up to the corner from two clearly-offset sides.
    pa.add(66, 90, (HEAVY_END[0] - 1.9, HEAVY_END[1] - 0.6), sprint=True)
    pb.add(52, 90, (HEAVY_END[0] - 0.6, HEAVY_END[1] - 2.0), sprint=True)
    # Settle just inside the locked fort for the rest of the episode.
    pa.add(90, prep, (2.6, 2.6))
    pb.add(90, prep, (3.2, 2.0))
    pa.hold(prep, prep + 60)
    pa.add(prep + 60, steps, (1.8, 1.8))      # shuffle deeper into the fort
    pb.hold(prep, prep + 70)
    pb.add(prep + 70, steps, (2.4, 1.2))

    # Seekers: released, rush the fort, get visibly blocked at the box wall, mill about.
    sa, sb = AgentPlan(SEEKER_START[SA]), AgentPlan(SEEKER_START[SB])
    sa.add(0, prep, SEEKER_START[SA])
    sb.add(0, prep, SEEKER_START[SB])
    sa.add(prep, prep + 46, (1.8, 1.0), sprint=True)
    sa.add(prep + 46, prep + 70, (2.7, 1.0), sprint=True)   # press against the wall
    sa.add(prep + 70, prep + 96, (2.4, -0.4))               # rebuffed, slides off
    sa.add(prep + 96, steps, (1.0, 0.2))
    sb.add(prep, prep + 50, (1.2, 2.0), sprint=True)
    sb.add(prep + 50, prep + 74, (1.0, 2.8), sprint=True)   # press against the wall
    sb.add(prep + 74, prep + 100, (-0.2, 2.6))              # rebuffed
    sb.add(prep + 100, steps, (0.2, 1.0))
    plans = {HA: pa, HB: pb, SA: sa, SB: sb}

    box_pos = {BOX_L1: BOX_START[BOX_L1], BOX_L2: BOX_START[BOX_L2],
               BOX_L3: BOX_START[BOX_L3], BOX_HEAVY: HEAVY_START}
    lock_step = 92  # once parked + shoved, boxes lock (lk=1) and never move again.

    def props(ctx: Dict) -> Dict[int, PropState]:
        t = ctx["t"]
        ast = ctx["agent_state"]
        carried_by = ctx["carried_by"]
        locked = 1 if t >= lock_step else 0
        out: Dict[int, PropState] = {}
        for bid in (BOX_L1, BOX_L2, BOX_L3):
            holder = carried_by.get(bid, -1)
            if holder >= 0 and not locked:
                box_pos[bid] = (ast[holder]["x"], ast[holder]["y"])
            elif locked:
                box_pos[bid] = BARRICADE[bid]
            bx, by = box_pos[bid]
            out[bid] = PropState(bx, by, lk=locked,
                                 hd=1 if (holder >= 0 and not locked) else 0,
                                 hb=holder if not locked else -1)
        if 66 <= t < 90:
            box_pos[BOX_HEAVY] = ((ast[HA]["x"] + ast[HB]["x"]) / 2.0 + 1.1,
                                  (ast[HA]["y"] + ast[HB]["y"]) / 2.0 + 1.3)
        elif t >= 90:
            box_pos[BOX_HEAVY] = HEAVY_END
        hx, hy = box_pos[BOX_HEAVY]
        out[BOX_HEAVY] = PropState(hx, hy, lk=locked)
        return out

    # Hiders win: never spotted.
    return Scenario(
        sid="fort", title="Fort Building", steps=steps, prep=prep, seed=seed,
        entity_spec=spec, walls=walls, door_id=DOOR, plans=plans,
        hider_ids=(HA, HB), seeker_ids=(SA, SB), total_entities=E,
        prop_dynamics=props, score_fn=hiders_win_score,
        seen_fn=lambda eid, ctx: 0,
    )


# =========================================================================== #
# SCENARIO 3: ramp -- a seeker drags the ramp to the fort and climbs in.
# =========================================================================== #
def scenario_ramp() -> Scenario:
    steps, prep, seed = 215, 90, 13
    spec: List[Tuple[str, int, bool]] = [
        ("hider", schema.TEAM_HIDER, False),    # 0
        ("hider", schema.TEAM_HIDER, False),    # 1
        ("seeker", schema.TEAM_SEEKER, False),  # 2
        ("seeker", schema.TEAM_SEEKER, False),  # 3
        ("box_light", schema.TEAM_NONE, False),  # 4
        ("box_light", schema.TEAM_NONE, False),  # 5
        ("box_light", schema.TEAM_NONE, False),  # 6
        ("box_heavy", schema.TEAM_NONE, False),  # 7
        ("ramp", schema.TEAM_NONE, False),       # 8
        ("wall", schema.TEAM_NONE, False),       # 9 border N
        ("wall", schema.TEAM_NONE, False),       # 10 border E
        ("wall", schema.TEAM_NONE, False),       # 11 border S
        ("door", schema.TEAM_NONE, False),       # 12 door
        ("wall", schema.TEAM_NONE, False),       # 13 border W-top
        ("wall", schema.TEAM_NONE, False),       # 14 border W-bottom
    ]
    HA, HB, SA, SB = 0, 1, 2, 3
    BOX_L1, BOX_L2, BOX_L3, BOX_HEAVY, RAMP = 4, 5, 6, 7, 8
    WALL_N, WALL_E, WALL_S, DOOR, WALL_WT, WALL_WB = 9, 10, 11, 12, 13, 14
    E = 16

    walls = border_walls(WALL_N, WALL_E, WALL_S, WALL_WT, WALL_WB)

    # Fort in the NE corner (built fast in early prep). Heavy box anchors the wall.
    HEAVY_END = (3.6, 4.4)
    BARRICADE = {BOX_L1: (4.6, 3.4), BOX_L2: (2.4, 4.4), BOX_L3: (4.6, 2.0)}
    BOX_START = {BOX_L1: (2.0, 2.0), BOX_L2: (1.2, 3.0), BOX_L3: (2.6, 1.0)}
    HEAVY_START = (1.6, 2.2)
    RAMP_START = (-3.6, -3.4)                    # ramp parked far away, SW
    RAMP_AT_FORT = (2.2, 2.4)                    # where the seeker plants the ramp
    SEEKER_START = {SA: (-4.6, -2.4), SB: (-4.6, 2.4)}
    HIDER_START = {HA: (3.4, 2.6), HB: (2.6, 3.4)}

    # Hiders work separate sides so their fetch/shove paths never cross the centre:
    # A (east) hauls L1 + L3 and flanks the heavy box from the EAST; B (west) hauls L2
    # and flanks from the WEST. They coop-shove it north together.
    pa, pb = AgentPlan(HIDER_START[HA]), AgentPlan(HIDER_START[HB])
    pa.add(4, 14, BOX_START[BOX_L1])
    pa.add(14, 26, BARRICADE[BOX_L1], carry=BOX_L1)
    pa.add(26, 36, BOX_START[BOX_L3])
    pa.add(36, 48, BARRICADE[BOX_L3], carry=BOX_L3)
    pb.add(4, 16, BOX_START[BOX_L2])
    pb.add(16, 28, BARRICADE[BOX_L2], carry=BOX_L2)
    pb.add(28, 40, (HEAVY_START[0] - 1.1, HEAVY_START[1] - 0.3))   # flank heavy box, WEST
    pa.add(48, 58, (HEAVY_START[0] + 1.1, HEAVY_START[1]))          # flank heavy box, EAST
    # Each hider ends its shove ON the side it will tuck into (A=east/SE, B=west/NW).
    pa.add(58, 80, (HEAVY_END[0] + 0.4, HEAVY_END[1] - 1.9), sprint=True)
    pb.add(40, 80, (HEAVY_END[0] - 1.6, HEAVY_END[1] - 0.7), sprint=True)
    pa.add(80, prep, (4.2, 2.0))                 # hider A tucks SE, off the climb line
    pb.add(80, prep, (2.0, 4.2))                 # hider B tucks NW, off the climb line
    # MAIN: hiders hunker in the fort; B is spotted late (the seeker crests the wall
    # and drops toward the centre) and bolts into the NW corner to escape the landing.
    pa.hold(prep, steps)
    pb.hold(prep, prep + 100)
    pb.add(prep + 100, prep + 124, (1.2, 4.6), sprint=True)
    pb.add(prep + 124, steps, (2.6, 4.4))        # cornered when seen

    sa, sb = AgentPlan(SEEKER_START[SA]), AgentPlan(SEEKER_START[SB])
    sa.add(0, prep, SEEKER_START[SA])
    sb.add(0, prep, SEEKER_START[SB])
    # Seeker A fetches the ramp, drags it to the fort, climbs it, drops inside.
    drag_t0, drag_t1 = prep + 20, prep + 56     # ramp is carried during the drag
    climb_t0, climb_t1 = prep + 60, prep + 92   # z rises on top, gr->0
    sa.add(prep, prep + 18, RAMP_START, sprint=True)
    sa.add(drag_t0, drag_t1, RAMP_AT_FORT, carry=RAMP, sprint=True)
    sa.add(drag_t1, climb_t0, (RAMP_AT_FORT[0] + 0.3, RAMP_AT_FORT[1] + 0.3))
    sa.add(climb_t0, climb_t1, (HEAVY_END[0], HEAVY_END[1] - 0.5))  # crest the box wall
    sa.add(climb_t1, steps, (1.8, 1.6))          # dropped inside the fort (off the corner)
    # Seeker B circles the fort wide along the EAST then south edge (clears A's drag).
    sb.add(prep, prep + 40, (3.2, -1.0), sprint=True)
    sb.add(prep + 40, prep + 80, (3.0, -3.2), sprint=True)
    sb.add(prep + 80, steps, (2.4, 0.2))
    plans = {HA: pa, HB: pb, SA: sa, SB: sb}

    seen_step = climb_t1 + 8                      # spotted just after the seeker lands

    box_pos = {BOX_L1: BOX_START[BOX_L1], BOX_L2: BOX_START[BOX_L2],
               BOX_L3: BOX_START[BOX_L3], BOX_HEAVY: HEAVY_START}
    ramp_pos = {"p": RAMP_START}

    def props(ctx: Dict) -> Dict[int, PropState]:
        t = ctx["t"]
        ast = ctx["agent_state"]
        carried_by = ctx["carried_by"]
        out: Dict[int, PropState] = {}
        for bid in (BOX_L1, BOX_L2, BOX_L3):
            holder = carried_by.get(bid, -1)
            if holder >= 0:
                box_pos[bid] = (ast[holder]["x"], ast[holder]["y"])
            bx, by = box_pos[bid]
            out[bid] = PropState(bx, by, hd=1 if holder >= 0 else 0, hb=holder)
        if 58 <= t < 80:
            box_pos[BOX_HEAVY] = ((ast[HA]["x"] + ast[HB]["x"]) / 2.0 + 0.7,
                                  (ast[HA]["y"] + ast[HB]["y"]) / 2.0 + 1.1)
        elif t >= 80:
            box_pos[BOX_HEAVY] = HEAVY_END
        hx, hy = box_pos[BOX_HEAVY]
        out[BOX_HEAVY] = PropState(hx, hy)
        # Ramp: dragged by seeker A while carried, else stays where last placed.
        ramp_holder = carried_by.get(RAMP, -1)
        if ramp_holder >= 0:
            ramp_pos["p"] = (ast[ramp_holder]["x"], ast[ramp_holder]["y"])
        rx, ry = ramp_pos["p"]
        # Ramp heading points toward the fort once planted (cosmetic incline dir).
        rh = math.atan2(HEAVY_END[1] - ry, HEAVY_END[0] - rx)
        out[RAMP] = PropState(rx, ry, h=rh,
                              hd=1 if ramp_holder >= 0 else 0, hb=ramp_holder)
        return out

    def agent_ov(eid: int, ctx: Dict) -> Dict:
        t = ctx["t"]
        if eid == SA and climb_t0 <= t < climb_t1:
            # Rise to ~1.4 on top, hold near the apex, anti box-surf: gr=0 while up.
            frac = (t - climb_t0) / max(1, climb_t1 - climb_t0)
            up = smoothstep(0.0, 1.4, clamp(frac / 0.6, 0.0, 1.0))
            z = round(up, 4)
            return {"z": z, "gr": 0 if z > 0.05 else 1}
        return {}

    return Scenario(
        sid="ramp", title="Ramp Use", steps=steps, prep=prep, seed=seed,
        entity_spec=spec, walls=walls, door_id=DOOR, plans=plans,
        hider_ids=(HA, HB), seeker_ids=(SA, SB), total_entities=E,
        prop_dynamics=props, score_fn=make_late_spot_score(seen_step),
        seen_fn=lambda eid, ctx: 1 if (eid == HB and ctx["is_main"]
                                       and ctx["t"] >= seen_step) else 0,
        agent_overrides=agent_ov,
    )


# =========================================================================== #
# SCENARIO 4: chase -- OPEN arena pursuit + dodging with sprint/stamina.
# =========================================================================== #
def scenario_chase() -> Scenario:
    steps, prep, seed = 200, 90, 17
    # Open arena: just the 4 borders + door. No interior wall, no boxes.
    spec: List[Tuple[str, int, bool]] = [
        ("hider", schema.TEAM_HIDER, False),    # 0
        ("hider", schema.TEAM_HIDER, False),    # 1
        ("seeker", schema.TEAM_SEEKER, False),  # 2
        ("seeker", schema.TEAM_SEEKER, False),  # 3
        ("wall", schema.TEAM_NONE, False),       # 4 border N
        ("wall", schema.TEAM_NONE, False),       # 5 border E
        ("wall", schema.TEAM_NONE, False),       # 6 border S
        ("door", schema.TEAM_NONE, False),       # 7 door
        ("wall", schema.TEAM_NONE, False),       # 8 border W-top
        ("wall", schema.TEAM_NONE, False),       # 9 border W-bottom
    ]
    HA, HB, SA, SB = 0, 1, 2, 3
    WALL_N, WALL_E, WALL_S, DOOR, WALL_WT, WALL_WB = 4, 5, 6, 7, 8, 9
    E = 12

    walls = border_walls(WALL_N, WALL_E, WALL_S, WALL_WT, WALL_WB)

    SEEKER_START = {SA: (-4.0, -1.0), SB: (-4.0, 1.0)}
    HIDER_START = {HA: (3.6, -2.6), HB: (3.2, 2.8)}

    # Hiders weave & dodge across the open arena with sprint bursts.
    pa, pb = AgentPlan(HIDER_START[HA]), AgentPlan(HIDER_START[HB])
    # Light prep jitter so they aren't statues, then a frantic main chase.
    pa.add(0, 30, (4.2, -3.4))
    pa.add(30, 60, (2.6, -4.2))
    pa.add(60, prep, (3.8, -1.8))
    pb.add(0, 30, (4.2, 3.6))
    pb.add(30, 60, (2.4, 4.2))
    pb.add(60, prep, (4.2, 1.6))
    # MAIN: serpentine dodging in separated lanes -- hider A keeps to the SOUTH
    # half, hider B to the NORTH half (alternating sprint legs; stamina dips).
    pa.add(prep, prep + 24, (-1.0, -4.2), sprint=True)
    pa.add(prep + 24, prep + 46, (-4.2, -4.0), sprint=True)
    pa.add(prep + 46, prep + 64, (-4.4, -1.6))                 # slow leg -> regen
    pa.add(prep + 64, prep + 88, (-1.4, -2.6), sprint=True)
    pa.add(prep + 88, steps, (2.6, -3.4), sprint=True)
    pb.add(prep, prep + 22, (0.6, 4.2), sprint=True)
    pb.add(prep + 22, prep + 44, (-2.8, 4.0), sprint=True)
    pb.add(prep + 44, prep + 62, (-4.4, 1.6))                  # slow leg -> regen
    pb.add(prep + 62, prep + 86, (-1.4, 2.6), sprint=True)
    pb.add(prep + 86, steps, (2.6, 3.4), sprint=True)

    # Two seekers pursue, each trailing one hider in its lane (a visible gap).
    sa, sb = AgentPlan(SEEKER_START[SA]), AgentPlan(SEEKER_START[SB])
    sa.add(0, prep, SEEKER_START[SA])
    sb.add(0, prep, SEEKER_START[SB])
    sa.add(prep, prep + 24, (0.8, -2.8), sprint=True)
    sa.add(prep + 24, prep + 50, (-2.2, -3.0), sprint=True)
    sa.add(prep + 50, prep + 70, (-2.8, -0.6), sprint=True)
    sa.add(prep + 70, prep + 92, (-0.2, -1.0), sprint=True)
    sa.add(prep + 92, steps, (1.8, -2.0), sprint=True)
    sb.add(prep, prep + 22, (1.6, 2.8), sprint=True)
    sb.add(prep + 22, prep + 46, (-1.4, 2.8), sprint=True)
    sb.add(prep + 46, prep + 68, (-2.8, 0.6), sprint=True)
    sb.add(prep + 68, prep + 90, (-0.2, 1.0), sprint=True)
    sb.add(prep + 90, steps, (1.8, 2.0), sprint=True)
    plans = {HA: pa, HB: pb, SA: sa, SB: sb}

    # Hider A spotted mid-run; score swings, then it escapes again (spot ends).
    seen_t0 = prep + int((steps - prep) * 0.34)
    seen_t1 = prep + int((steps - prep) * 0.50)

    def score(ctx: Dict) -> Tuple[float, float, bool]:
        if not ctx["is_main"]:
            return 0.0, 0.0, False
        t = ctx["t"]
        # Seekers bank points only during the spot window; hiders gain otherwise.
        seen_frames = max(0, min(t, seen_t1) - seen_t0 + 1) if t >= seen_t0 else 0
        hider_frames = (t - ctx["prep"] + 1) - seen_frames
        seen_now = seen_t0 <= t < seen_t1
        return float(max(0, hider_frames)), float(2.0 * seen_frames), seen_now

    return Scenario(
        sid="chase", title="Running & Chasing", steps=steps, prep=prep, seed=seed,
        entity_spec=spec, walls=walls, door_id=DOOR, plans=plans,
        hider_ids=(HA, HB), seeker_ids=(SA, SB), total_entities=E,
        prop_dynamics=lambda ctx: {}, score_fn=score,
        seen_fn=lambda eid, ctx: 1 if (eid == HA and ctx["is_main"]
                                       and seen_t0 <= ctx["t"] < seen_t1) else 0,
    )


# =========================================================================== #
# SCENARIO 5: doors -- hiders jam the west DOOR gap with the heavy box.
# =========================================================================== #
def scenario_doors() -> Scenario:
    steps, prep, seed = 205, 96, 19
    spec: List[Tuple[str, int, bool]] = [
        ("hider", schema.TEAM_HIDER, False),    # 0
        ("hider", schema.TEAM_HIDER, False),    # 1
        ("seeker", schema.TEAM_SEEKER, False),  # 2
        ("seeker", schema.TEAM_SEEKER, False),  # 3
        ("box_heavy", schema.TEAM_NONE, False),  # 4 heavy box -> jams the door
        ("box_light", schema.TEAM_NONE, False),  # 5 light box (a little extra cover)
        ("wall", schema.TEAM_NONE, False),       # 6 border N
        ("wall", schema.TEAM_NONE, False),       # 7 border E
        ("wall", schema.TEAM_NONE, False),       # 8 border S
        ("door", schema.TEAM_NONE, False),       # 9 door (blocked the WHOLE episode)
        ("wall", schema.TEAM_NONE, False),       # 10 border W-top
        ("wall", schema.TEAM_NONE, False),       # 11 border W-bottom
    ]
    HA, HB, SA, SB = 0, 1, 2, 3
    BOX_HEAVY, BOX_L1 = 4, 5
    WALL_N, WALL_E, WALL_S, DOOR, WALL_WT, WALL_WB = 6, 7, 8, 9, 10, 11
    E = 14

    walls = border_walls(WALL_N, WALL_E, WALL_S, WALL_WT, WALL_WB)

    # Door sits at the west gap (-6, 0). The heavy box ends centered just inside it.
    DOOR_X, DOOR_Y = DOOR_GEOM[0], DOOR_GEOM[1]
    HEAVY_START = (-1.0, 0.6)
    HEAVY_END = (DOOR_X + 1.15, DOOR_Y)          # centered on the door, just inside
    LIGHT_START = (-1.6, -2.6)
    LIGHT_END = (-4.4, -0.9)                     # tucked beside the jammed door
    SEEKER_START = {SA: (4.6, -1.2), SB: (4.6, 1.2)}   # they will end up OUTSIDE west
    HIDER_START = {HA: (-2.6, 1.2), HB: (-2.0, -0.6)}

    pa, pb = AgentPlan(HIDER_START[HA]), AgentPlan(HIDER_START[HB])
    # Both hiders flank the heavy box and coop-shove it into the doorway, then lock it.
    pa.add(6, 20, (HEAVY_START[0], HEAVY_START[1] + 1.0))
    pb.add(6, 20, (HEAVY_START[0], HEAVY_START[1] - 1.0))
    pa.add(20, 64, (HEAVY_END[0] + 1.0, HEAVY_END[1] + 0.9), sprint=True)
    pb.add(20, 64, (HEAVY_END[0] + 1.0, HEAVY_END[1] - 0.9), sprint=True)
    # Hider B also nudges a light box in as extra cover.
    pb.add(64, 74, LIGHT_START)
    pb.add(74, 86, LIGHT_END, carry=BOX_L1)
    pa.add(64, prep, (-3.4, 1.4))
    pb.add(86, prep, (-3.6, -1.6))
    # MAIN: hiders relax safely behind the blocked door (small idle shuffles).
    pa.add(prep, prep + 60, (-2.6, 2.2))
    pa.add(prep + 60, steps, (-3.0, 1.0))
    pb.add(prep, prep + 70, (-2.4, -2.2))
    pb.add(prep + 70, steps, (-3.2, -1.0))

    # Seekers come from the east, loop around to the west door, and pile up outside.
    sa, sb = AgentPlan(SEEKER_START[SA]), AgentPlan(SEEKER_START[SB])
    sa.add(0, prep, SEEKER_START[SA])
    sb.add(0, prep, SEEKER_START[SB])
    # Hug the south / north wall to the west gap (clears the hiders' idle zone).
    sa.add(prep, prep + 30, (-1.0, -4.2), sprint=True)
    sa.add(prep + 30, prep + 54, (-4.6, -4.2), sprint=True)
    sa.add(prep + 54, prep + 78, (-7.2, -1.1), sprint=True)   # arrive outside the door
    sa.add(prep + 78, steps, (-7.1, -0.6))                    # mill at the blocked door
    sb.add(prep, prep + 32, (-1.0, 4.2), sprint=True)
    sb.add(prep + 32, prep + 56, (-4.6, 4.2), sprint=True)
    sb.add(prep + 56, prep + 80, (-7.2, 1.1), sprint=True)    # arrive outside the door
    sb.add(prep + 80, steps, (-7.1, 0.6))                     # mill at the blocked door
    plans = {HA: pa, HB: pb, SA: sa, SB: sb}

    box_pos = {BOX_HEAVY: HEAVY_START, BOX_L1: LIGHT_START}
    heavy_lock_step = 66   # once jammed in the gap, the heavy box locks (lk=1).

    def props(ctx: Dict) -> Dict[int, PropState]:
        t = ctx["t"]
        ast = ctx["agent_state"]
        carried_by = ctx["carried_by"]
        out: Dict[int, PropState] = {}
        # Heavy box: coop-shoved 20..64, then locked centered on the door.
        if 20 <= t < 64:
            box_pos[BOX_HEAVY] = ((ast[HA]["x"] + ast[HB]["x"]) / 2.0 - 1.0,
                                  (ast[HA]["y"] + ast[HB]["y"]) / 2.0)
        elif t >= 64:
            box_pos[BOX_HEAVY] = HEAVY_END
        hx, hy = box_pos[BOX_HEAVY]
        out[BOX_HEAVY] = PropState(hx, hy, lk=1 if t >= heavy_lock_step else 0)
        # Light box.
        lholder = carried_by.get(BOX_L1, -1)
        if lholder >= 0:
            box_pos[BOX_L1] = (ast[lholder]["x"], ast[lholder]["y"])
        elif t >= 86:
            box_pos[BOX_L1] = LIGHT_END
        lx, ly = box_pos[BOX_L1]
        out[BOX_L1] = PropState(lx, ly, lk=1 if t >= 88 else 0,
                                hd=1 if lholder >= 0 else 0, hb=lholder)
        return out

    # Door stays CLOSED/blocked (a=1) the WHOLE episode; hiders win (never spotted).
    return Scenario(
        sid="doors", title="Door Blocking", steps=steps, prep=prep, seed=seed,
        entity_spec=spec, walls=walls, door_id=DOOR, plans=plans,
        hider_ids=(HA, HB), seeker_ids=(SA, SB), total_entities=E,
        prop_dynamics=props, score_fn=hiders_win_score,
        seen_fn=lambda eid, ctx: 0,
        door_active=lambda ctx: 1,   # blocked door never opens
    )


# =========================================================================== #
# SCENARIO 6: decoy -- a decoy pulls the seekers the wrong way; hiders escape.
# =========================================================================== #
def scenario_decoy() -> Scenario:
    steps, prep, seed = 200, 90, 23
    spec: List[Tuple[str, int, bool]] = [
        ("hider", schema.TEAM_HIDER, False),    # 0
        ("hider", schema.TEAM_HIDER, False),    # 1
        ("seeker", schema.TEAM_SEEKER, False),  # 2
        ("seeker", schema.TEAM_SEEKER, False),  # 3
        ("decoy", schema.TEAM_NONE, True),       # 4 decoy (true identity, god-view)
        ("box_light", schema.TEAM_NONE, False),  # 5 light box (a bit of scenery/cover)
        ("wall", schema.TEAM_NONE, False),       # 6 border N
        ("wall", schema.TEAM_NONE, False),       # 7 border E
        ("wall", schema.TEAM_NONE, False),       # 8 border S
        ("door", schema.TEAM_NONE, False),       # 9 door
        ("wall", schema.TEAM_NONE, False),       # 10 border W-top
        ("wall", schema.TEAM_NONE, False),       # 11 border W-bottom
    ]
    HA, HB, SA, SB = 0, 1, 2, 3
    DECOY, BOX_L1 = 4, 5
    WALL_N, WALL_E, WALL_S, DOOR, WALL_WT, WALL_WB = 6, 7, 8, 9, 10, 11
    E = 14

    walls = border_walls(WALL_N, WALL_E, WALL_S, WALL_WT, WALL_WB)

    # Hiders flee to the SOUTH-EAST; the decoy pulses in the NORTH-WEST (opposite).
    DECOY_POS = (-3.8, 3.8)
    BOX_L1_POS = (1.2, -1.0)
    # SA starts to the WEST, SB to the EAST so their veer-to-decoy lanes never cross.
    SEEKER_START = {SA: (-1.4, 0.6), SB: (1.4, 1.0)}   # start mid-arena
    # HA starts in the EAST/upper row, HB in the SOUTH/lower row (lanes never cross).
    HIDER_START = {HA: (-1.2, -0.6), HB: (-2.4, -2.2)}

    decoy_on = prep - 18      # decoy lights up just before release
    veer_t0 = prep            # seekers commit toward the decoy at release
    veer_t1 = prep + 60       # ...reach it here, having abandoned the hiders

    pa, pb = AgentPlan(HIDER_START[HA]), AgentPlan(HIDER_START[HB])
    # Prep: hiders edge toward the SE escape lane in already-separated rows so their
    # main flee legs never cross -- A holds the EAST lane, B the SOUTH lane.
    pa.add(0, 40, (1.2, -0.8))
    pa.add(40, prep, (3.4, -1.6))
    pb.add(0, 40, (0.6, -3.0))
    pb.add(40, prep, (2.0, -3.6))
    # MAIN: hiders sprint away while the seekers chase the decoy -- A hugs the EAST
    # edge, B hugs the SOUTH edge, so the two of them never bunch up.
    pa.add(prep, prep + 50, (4.4, -1.2), sprint=True)
    pa.add(prep + 50, steps, (4.6, 1.4))
    pb.add(prep, prep + 54, (2.0, -4.2), sprint=True)
    pb.add(prep + 54, steps, (-0.6, -4.4))

    # Seekers: in prep drift toward the hiders; at release their paths BEND to the
    # decoy (NW) instead, then they search around the decoy -- pulled the wrong way.
    sa, sb = AgentPlan(SEEKER_START[SA]), AgentPlan(SEEKER_START[SB])
    sa.add(0, prep, (-1.2, -0.2))            # had been closing on the hiders
    sb.add(0, prep, (0.8, -0.4))
    # SA = the WESTERN seeker the whole way; SB = the EASTERN one (lanes never cross).
    sa.add(veer_t0, veer_t1, (DECOY_POS[0] - 1.1, DECOY_POS[1] - 0.6), sprint=True)
    sa.add(veer_t1, steps, (-4.8, 2.6))      # casting about, WEST of the decoy
    sb.add(veer_t0, veer_t1, (DECOY_POS[0] + 1.3, DECOY_POS[1] - 1.0), sprint=True)
    sb.add(veer_t1, steps, (-2.0, 4.2))      # casting about, EAST of the decoy
    plans = {HA: pa, HB: pb, SA: sa, SB: sb}

    def props(ctx: Dict) -> Dict[int, PropState]:
        t = ctx["t"]
        out: Dict[int, PropState] = {}
        if decoy_on <= t:
            out[DECOY] = PropState(DECOY_POS[0], DECOY_POS[1],
                                   no=decoy_noise_at(t - decoy_on), dc=1)
        else:
            out[DECOY] = PropState(DECOY_POS[0], DECOY_POS[1], no=0.0, dc=0)
        out[BOX_L1] = PropState(BOX_L1_POS[0], BOX_L1_POS[1])
        return out

    # Deception works: no hider ever spotted; hiders bank the score.
    return Scenario(
        sid="decoy", title="Sensory Deception", steps=steps, prep=prep, seed=seed,
        entity_spec=spec, walls=walls, door_id=DOOR, plans=plans,
        hider_ids=(HA, HB), seeker_ids=(SA, SB), total_entities=E,
        prop_dynamics=props, score_fn=hiders_win_score,
        seen_fn=lambda eid, ctx: 0,
    )


# --------------------------------------------------------------------------- #
# Scenario registry + manifest.
# --------------------------------------------------------------------------- #
SCENARIO_BUILDERS: Dict[str, Callable[[], Scenario]] = {
    "showcase": scenario_showcase,
    "fort": scenario_fort,
    "ramp": scenario_ramp,
    "chase": scenario_chase,
    "doors": scenario_doors,
    "decoy": scenario_decoy,
}

# Manifest entries (id/title/description/file) -- the hard contract the viewer reads.
MANIFEST_ENTRIES: List[Dict[str, str]] = [
    {"id": "showcase", "title": "Synthetic Showcase",
     "description": "All 2.0 mechanics in one run", "file": "showcase.json"},
    {"id": "fort", "title": "Fort Building",
     "description": "Two hiders gather and lock boxes into a barricade", "file": "fort.json"},
    {"id": "ramp", "title": "Ramp Use",
     "description": "A seeker drags a ramp to the fort and climbs in", "file": "ramp.json"},
    {"id": "chase", "title": "Running & Chasing",
     "description": "Open-arena pursuit and dodging", "file": "chase.json"},
    {"id": "doors", "title": "Door Blocking",
     "description": "Hiders jam the doorway with a heavy box", "file": "doors.json"},
    {"id": "decoy", "title": "Sensory Deception",
     "description": "A decoy lures the seekers the wrong way", "file": "decoy.json"},
]
DEFAULT_SCENARIO = "showcase"


def manifest_doc() -> Dict:
    return {
        "format": "hns2-manifest",
        "version": 1,
        "default": DEFAULT_SCENARIO,
        "scenarios": [dict(e) for e in MANIFEST_ENTRIES],
    }


# --------------------------------------------------------------------------- #
# Output paths / writing.
# --------------------------------------------------------------------------- #
TRAJ_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "trajectories")


def _write_json(path: str, doc: Dict) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    return os.path.getsize(path)


def build_and_save(sid: str, out_dir: str = TRAJ_DIR) -> Tuple[Dict, str, int, int]:
    """Build one scenario, validate-on-save, return (doc, path, frames, bytes)."""
    scn = SCENARIO_BUILDERS[sid]()
    doc = build_trajectory(scn)
    path = os.path.join(out_dir, f"{sid}.json")
    os.makedirs(out_dir, exist_ok=True)
    schema.save_trajectory(doc, path)  # validates first; raises on problems
    return doc, path, len(doc["frames"]), os.path.getsize(path)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate the synthetic Hide & Seek 2.0 scenario set + manifest "
                    "(pure stdlib; clean square arena; OpenAI-style named clips).",
    )
    parser.add_argument("--out-dir", default=TRAJ_DIR,
                        help="output directory (default: viz/web/trajectories/)")
    parser.add_argument("--scenario", default=None, metavar="ID",
                        help="(re)generate ONLY this scenario id "
                             f"(one of: {', '.join(SCENARIO_BUILDERS)})")
    parser.add_argument("--list", action="store_true",
                        help="list scenario ids + titles and exit")
    args = parser.parse_args(argv)

    if args.list:
        print("scenarios:")
        for e in MANIFEST_ENTRIES:
            mark = "  (default)" if e["id"] == DEFAULT_SCENARIO else ""
            print(f"  {e['id']:<9} {e['title']}{mark}")
            print(f"            {e['description']}")
        return 0

    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if args.scenario is not None:
        sid = args.scenario
        if sid not in SCENARIO_BUILDERS:
            parser.error(f"unknown scenario {sid!r}; choose from {', '.join(SCENARIO_BUILDERS)}")
        doc, path, nf, nbytes = build_and_save(sid, out_dir)
        problems = schema.validate_trajectory(doc)
        print(f"wrote {path}")
        print(f"  frames={nf}  entities={len(doc['entities'])}  bytes={nbytes}")
        print(f"  validate_trajectory -> {len(problems)} problem(s)"
              + ("" if not problems else ": " + "; ".join(problems)))
        if sid == DEFAULT_SCENARIO:  # keep the legacy demo file in sync
            demo = os.path.join(out_dir, "demo_trajectory.json")
            _write_json(demo, doc)
            print(f"wrote {demo}  (copy of {sid}.json)")
        return 0 if not problems else 1

    # Default: build EVERY scenario + the legacy demo copy + the manifest.
    rc = 0
    showcase_doc: Dict | None = None
    print(f"out-dir: {out_dir}")
    for sid in SCENARIO_BUILDERS:
        doc, path, nf, nbytes = build_and_save(sid, out_dir)
        problems = schema.validate_trajectory(doc)
        if sid == DEFAULT_SCENARIO:
            showcase_doc = doc
        status = f"{len(problems)} problem(s)" + ("" if not problems else ": " + "; ".join(problems))
        print(f"  {sid:<9} frames={nf:<4} bytes={nbytes:<7} entities={len(doc['entities']):<3} "
              f"validate -> {status}")
        if problems:
            rc = 1

    # Legacy demo_trajectory.json == showcase.json (keep old links/docs working).
    if showcase_doc is not None:
        demo = os.path.join(out_dir, "demo_trajectory.json")
        nbytes = _write_json(demo, showcase_doc)
        print(f"  {'demo':<9} (copy of showcase.json) bytes={nbytes}")

    # Manifest (exact contract the viewer depends on).
    man = manifest_doc()
    man_path = os.path.join(out_dir, "manifest.json")
    nbytes = _write_json(man_path, man)
    print(f"  {'manifest':<9} {man_path}  bytes={nbytes}  scenarios={len(man['scenarios'])}")
    return rc


if __name__ == "__main__":
    import sys

    sys.exit(main())
