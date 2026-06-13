"""
viz/make_demo_trajectory.py -- a SYNTHETIC Hide & Seek 2.0 episode generator.

PURE STDLIB ONLY (``math``, ``json``, ``random``, ``argparse``) -- importing or
running this module never touches jax or numpy, so the 3D viewer has something
clean to play the moment the repo is cloned, on any machine.

What it produces
----------------
A single ``hns2-traj`` v1 document (see :mod:`viz.schema`) describing a tidy,
~220-frame episode inside a **clean square arena** (continuous border walls with a
door gap, a small interior wall) that showcases the 2.0 mechanics:

PREP phase (``t < prep_steps``)
    * Seekers are held back / frozen at a start pad (``lk=1``) until the main phase.
    * Two hiders fetch the three light boxes (``hd=1``/``hb``) and stack them into a
      barricade, then cooperatively shove the heavy box together (coop physics).
    * Late in prep a decoy switches on (``dc=1``) with pulsing noise.

MAIN phase (``t >= prep_steps``)
    * Seekers released (``lk=0``) and chase; they keep a visible gap from the hiders.
    * One seeker climbs the ramp (``z`` rises, ``gr`` -> 0 on top: anti box-surf).
    * The fog patches drift; the door opens mid-main (``a`` -> 0); the interior wall
      breaks when a seeker rams it (``a`` -> 0).
    * A hider is spotted late (``sn`` -> 1, ``seen_any`` -> true) and the score swings.

Arena & entities
----------------
A 12x12 arena with four continuous border walls (north/south/east + a split west
wall with a central door), one breakable interior wall, plus 2 hiders, 2 seekers,
3 light boxes, 1 heavy box, 1 ramp and 1 decoy -- padded to a fixed ``E`` so the
slot layout is stable. Walls carry a real ``heading`` so the viewer orients them.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from typing import Dict, List, Tuple

from viz import schema

# --------------------------------------------------------------------------- #
# Fixed episode geometry / tuning (kept independent of config.py: stdlib promise)
# --------------------------------------------------------------------------- #
ARENA_SIZE = 12.0          # arena spans [-6, +6] in x and y
HALF = ARENA_SIZE / 2.0
DT = 0.1                   # seconds per control step (playback speed)
DEFAULT_STEPS = 220
DEFAULT_PREP = 96
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

TOTAL_ENTITIES = 18


# --------------------------------------------------------------------------- #
# Small stdlib math helpers (no numpy).
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


# --------------------------------------------------------------------------- #
# Entity table (agents-first, then props, then walls/door), padded to E.
# --------------------------------------------------------------------------- #
_ENTITY_SPEC: List[Tuple[str, int, bool]] = [
    ("hider", schema.TEAM_HIDER, False),    # 0  hider A
    ("hider", schema.TEAM_HIDER, False),    # 1  hider B
    ("seeker", schema.TEAM_SEEKER, False),  # 2  seeker A
    ("seeker", schema.TEAM_SEEKER, False),  # 3  seeker B
    ("box_light", schema.TEAM_NONE, False),  # 4  light box 1
    ("box_light", schema.TEAM_NONE, False),  # 5  light box 2
    ("box_light", schema.TEAM_NONE, False),  # 6  light box 3
    ("box_heavy", schema.TEAM_NONE, False),  # 7  heavy box (coop)
    ("ramp", schema.TEAM_NONE, False),       # 8  ramp
    ("decoy", schema.TEAM_NONE, True),       # 9  decoy (TRUE decoy identity)
    ("wall", schema.TEAM_NONE, False),       # 10 border wall N
    ("wall", schema.TEAM_NONE, False),       # 11 border wall E
    ("wall", schema.TEAM_NONE, False),       # 12 border wall S
    ("wall", schema.TEAM_NONE, False),       # 13 interior fragile wall (breaks)
    ("door", schema.TEAM_NONE, False),       # 14 door (west gap; opens mid-main)
    ("wall", schema.TEAM_NONE, False),       # 15 border wall W-top
    ("wall", schema.TEAM_NONE, False),       # 16 border wall W-bottom
]

HIDER_A, HIDER_B = 0, 1
SEEKER_A, SEEKER_B = 2, 3
BOX_L1, BOX_L2, BOX_L3 = 4, 5, 6
BOX_HEAVY = 7
RAMP = 8
DECOY = 9
WALL_N, WALL_E, WALL_S, WALL_FRAGILE = 10, 11, 12, 13
DOOR = 14
WALL_WT, WALL_WB = 15, 16

# Walls / door geometry: id -> (x, y, heading, half_len). Length rendered = half_len*2.
# Border half-lengths slightly exceed HALF so the four borders overlap at the corners.
WALL_DEF: Dict[int, Tuple[float, float, float, float]] = {
    WALL_N:       (0.0,  6.0, 0.0, 6.25),
    WALL_S:       (0.0, -6.0, 0.0, 6.25),
    WALL_E:       (6.0,  0.0, PI2, 6.25),
    WALL_WT:      (-6.0,  3.4, PI2, 2.45),   # west wall, upper segment
    WALL_WB:      (-6.0, -3.4, PI2, 2.45),   # west wall, lower segment
    WALL_FRAGILE: (1.9,  1.1, PI2, 1.70),    # interior wall a seeker rams
}
DOOR_DEF: Tuple[float, float, float, float] = (-6.0, 0.0, PI2, 1.15)  # fills west gap

# --------------------------------------------------------------------------- #
# Prop placements (arena coords), all comfortably inside the border walls.
# --------------------------------------------------------------------------- #
RAMP_POS = (3.9, 2.7)
DECOY_POS = (-3.0, -3.7)

BARRICADE = {                       # where hiders stack the light boxes (NW corner)
    BOX_L1: (-4.6, 3.0),
    BOX_L2: (-3.6, 3.5),
    BOX_L3: (-2.6, 3.0),
}
BOX_START = {                       # light boxes start scattered; hiders fetch them
    BOX_L1: (-4.4, -3.4),
    BOX_L2: (-1.2, -4.2),
    BOX_L3: (2.6, -3.2),
}
HEAVY_START = (0.6, -0.4)
HEAVY_END = (-2.6, 4.2)            # shoved up near the barricade fort

SEEKER_START = {SEEKER_A: (4.7, -3.6), SEEKER_B: (4.7, 3.6)}  # east pad, frozen in prep
HIDER_START = {HIDER_A: (-2.4, -1.2), HIDER_B: (-3.6, 0.6)}


# --------------------------------------------------------------------------- #
# Per-agent scripted, eased waypoint plan.
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


def build_hider_plans(prep: int, steps: int) -> Tuple[AgentPlan, AgentPlan]:
    """Two hiders: fetch boxes -> barricade -> coop-shove heavy box -> flee."""
    pa = AgentPlan(HIDER_START[HIDER_A])
    pb = AgentPlan(HIDER_START[HIDER_B])

    # PREP: hider A hauls boxes 1 & 2; hider B hauls box 3.
    pa.add(4, 16, BOX_START[BOX_L1])
    pa.add(16, 30, BARRICADE[BOX_L1], carry=BOX_L1)
    pa.add(30, 42, BOX_START[BOX_L2])
    pa.add(42, 56, BARRICADE[BOX_L2], carry=BOX_L2)
    pb.add(6, 24, BOX_START[BOX_L3])
    pb.add(24, 40, BARRICADE[BOX_L3], carry=BOX_L3)

    # Both flank and cooperatively shove the heavy box.
    pa.add(56, 66, (HEAVY_START[0] - 1.0, HEAVY_START[1]))
    pb.add(40, 54, (HEAVY_START[0] + 1.0, HEAVY_START[1]))
    pa.add(66, 88, (HEAVY_END[0] - 1.0, HEAVY_END[1]))
    pb.add(66, 88, (HEAVY_END[0] + 1.0, HEAVY_END[1]))
    # Settle behind the barricade for the rest of prep.
    pa.add(88, prep, (-2.6, 2.2))
    pb.add(88, prep, (-1.2, 1.8))

    # MAIN: flee. Hider A heads for the (opening) west door; hider B breaks east.
    pa.add(prep, prep + 44, (-3.4, -0.6), sprint=True)
    pa.add(prep + 44, prep + 92, (-4.6, -2.2), sprint=True)
    pa.add(prep + 92, steps, (-2.8, -3.0))
    pb.add(prep, prep + 38, (1.8, 1.6), sprint=True)
    pb.add(prep + 38, prep + 84, (3.2, -0.6), sprint=True)
    pb.add(prep + 84, steps, (1.0, -2.2), sprint=True)
    return pa, pb


def build_seeker_plans(prep: int, steps: int) -> Tuple[AgentPlan, AgentPlan]:
    """Two seekers: frozen in prep; then a wall-rammer and a ramp-climber that
    chase but keep a visible trailing gap from their target."""
    pa = AgentPlan(SEEKER_START[SEEKER_A])
    pb = AgentPlan(SEEKER_START[SEEKER_B])
    pa.add(0, prep, SEEKER_START[SEEKER_A])
    pb.add(0, prep, SEEKER_START[SEEKER_B])

    # Seeker A: ram the fragile interior wall, then pursue hider A (trailing).
    frag = (WALL_DEF[WALL_FRAGILE][0], WALL_DEF[WALL_FRAGILE][1])
    pa.add(prep, prep + 18, frag, sprint=True)
    pa.add(prep + 18, prep + 62, (-2.0, -0.2), sprint=True)
    pa.add(prep + 62, steps, (-3.4, -2.4), sprint=True)

    # Seeker B: climb the ramp (z rises), then pursue hider B (trailing gap).
    pb.add(prep, prep + 20, RAMP_POS, sprint=True)
    pb.add(prep + 20, prep + 40, (RAMP_POS[0] - 0.5, RAMP_POS[1] - 0.7))
    pb.add(prep + 40, prep + 82, (3.0, 0.6), sprint=True)
    pb.add(prep + 82, steps, (1.9, -1.4), sprint=True)
    return pa, pb


# --------------------------------------------------------------------------- #
# Fog patches that slowly drift.
# --------------------------------------------------------------------------- #
def fog_at(t: int, steps: int) -> List[List[float]]:
    """Return the 3 drifting fog ``[x, y, r]`` triples at step ``t`` (deterministic)."""
    frac = t / max(1, steps - 1)
    specs = [
        (-3.0, 1.0, 0.9, 0.6, 0.0),
        (2.5, 2.0, -0.8, 0.7, 1.7),
        (0.5, -3.0, 0.7, -0.9, 3.1),
    ]
    base_r = 2.3
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
# Static entity table.
# --------------------------------------------------------------------------- #
def build_entities() -> List[Dict]:
    """Build the STATIC per-slot ``entities`` table (ids == index), padded to E."""
    entities: List[Dict] = []
    for i, (etype, team, is_decoy) in enumerate(_ENTITY_SPEC):
        size = SIZE[etype]
        if i in WALL_DEF:
            size = WALL_DEF[i][3]
        elif i == DOOR:
            size = DOOR_DEF[3]
        entities.append(schema.make_entity_meta(
            id=i, type=etype, team=team, size=size, mass=MASS[etype], is_decoy=is_decoy,
        ))
    for i in range(len(_ENTITY_SPEC), TOTAL_ENTITIES):  # inert padding walls
        entities.append(schema.make_entity_meta(
            id=i, type="wall", team=schema.TEAM_NONE,
            size=SIZE["wall"], mass=MASS["wall"], is_decoy=False,
        ))
    return entities


# --------------------------------------------------------------------------- #
# Episode builder.
# --------------------------------------------------------------------------- #
def build_trajectory(seed: int, steps: int, prep: int) -> Dict:
    """Build and return a validated-on-save ``hns2-traj`` v1 document."""
    random.Random(seed)  # seed reserved for future jitter; choreography is fixed
    entities = build_entities()
    E = len(entities)

    plans = {
        HIDER_A: build_hider_plans(prep, steps)[0],
        HIDER_B: build_hider_plans(prep, steps)[1],
        SEEKER_A: build_seeker_plans(prep, steps)[0],
        SEEKER_B: build_seeker_plans(prep, steps)[1],
    }

    decoy_on_step = prep - 22
    wall_break_step = prep + 16
    door_open_step = prep + (steps - prep) // 2
    seen_step = prep + int((steps - prep) * 0.62)
    ramp_up_t0, ramp_up_t1 = prep + 20, prep + 40

    headings = {HIDER_A: 0.0, HIDER_B: math.pi, SEEKER_A: math.pi, SEEKER_B: math.pi}
    prev_pos = {a: SEEKER_START.get(a, HIDER_START.get(a)) for a in plans}
    prev_pos[HIDER_A], prev_pos[HIDER_B] = HIDER_START[HIDER_A], HIDER_START[HIDER_B]
    stamina = {a: 1.0 for a in plans}
    sh = ss = 0.0
    box_pos = {
        BOX_L1: BOX_START[BOX_L1], BOX_L2: BOX_START[BOX_L2],
        BOX_L3: BOX_START[BOX_L3], BOX_HEAVY: HEAVY_START,
    }

    frames: List[Dict] = []
    for t in range(steps):
        phase = phase_for(t, prep)
        is_main = phase == "main"

        agent_state: Dict[int, Dict] = {}
        carried_by: Dict[int, int] = {}
        for aid, plan in plans.items():
            x, y, carry, sprint, moving = plan.sample(t)
            x, y = clamp_arena(x, y)
            px, py = prev_pos[aid]
            headings[aid] = heading_from(x - px, y - py, headings[aid])
            prev_pos[aid] = (x, y)
            if sprint and moving:
                stamina[aid] = clamp(stamina[aid] - 0.020, 0.05, 1.0)
            else:
                stamina[aid] = clamp(stamina[aid] + 0.010, 0.05, 1.0)
            agent_state[aid] = {"x": x, "y": y, "h": headings[aid],
                                "sprint": sprint, "moving": moving, "st": stamina[aid]}
            if carry >= 0:
                carried_by[carry] = aid

        seekers_frozen = not is_main

        # Seeker B ramp climb (z up, grounded off on top).
        sb_z, sb_grounded = 0.0, 1
        if ramp_up_t0 <= t < ramp_up_t1:
            mid = (ramp_up_t0 + ramp_up_t1) / 2.0
            climb = ((t - ramp_up_t0) / max(1, mid - ramp_up_t0) if t <= mid
                     else 1.0 - (t - mid) / max(1, ramp_up_t1 - mid))
            sb_z = round(1.2 * clamp(climb, 0.0, 1.0), 4)
            sb_grounded = 0 if sb_z > 0.05 else 1

        # Decoy activation + pulsing noise.
        decoy_active = 1 if decoy_on_step <= t else 0
        if decoy_active:
            since = t - decoy_on_step
            decoy_noise = clamp(0.6 * clamp(since / 12.0, 0.0, 1.0)
                                + 0.15 * (0.5 + 0.5 * math.sin(since * 0.7)), 0.0, 1.0)
        else:
            decoy_noise = 0.0

        wall_fragile_active = 0 if (is_main and t >= wall_break_step) else 1
        door_active = 0 if (is_main and t >= door_open_step) else 1
        seen_this_frame = is_main and t >= seen_step

        if is_main:
            if seen_this_frame:
                ss += 1.0
                sh -= 1.0
            else:
                sh += 1.0
                ss -= 1.0

        for box_id in (BOX_L1, BOX_L2, BOX_L3):
            if box_id in carried_by:
                hid = carried_by[box_id]
                box_pos[box_id] = (agent_state[hid]["x"], agent_state[hid]["y"])
        if 66 <= t < 88:
            box_pos[BOX_HEAVY] = ((agent_state[HIDER_A]["x"] + agent_state[HIDER_B]["x"]) / 2.0,
                                  (agent_state[HIDER_A]["y"] + agent_state[HIDER_B]["y"]) / 2.0)
        elif t >= 88:
            box_pos[BOX_HEAVY] = HEAVY_END

        fog = fog_at(t, steps)

        ent: List[Dict] = []
        for eid in range(E):
            etype = entities[eid]["type"]

            if eid in (HIDER_A, HIDER_B, SEEKER_A, SEEKER_B):
                st = agent_state[eid]
                is_seeker = eid in (SEEKER_A, SEEKER_B)
                z, grounded, locked = 0.0, 1, 0
                if is_seeker:
                    if seekers_frozen:
                        locked = 1
                    if eid == SEEKER_B:
                        z, grounded = sb_z, sb_grounded
                noise = 0.25 if (st["sprint"] and st["moving"]) else 0.05
                sn = 1 if (eid == HIDER_B and seen_this_frame) else 0
                ent.append(schema.make_frame_ent(
                    id=eid, x=st["x"], y=st["y"], z=z, h=st["h"],
                    a=1, lk=locked, hd=0, hb=-1, no=noise, dc=0, gr=grounded,
                    st=st["st"], sn=sn,
                ))
                continue

            if eid in (BOX_L1, BOX_L2, BOX_L3):
                bx, by = box_pos[eid]
                holder = carried_by.get(eid, -1)
                ent.append(schema.make_frame_ent(
                    id=eid, x=bx, y=by, z=0.0, h=0.0,
                    a=1, lk=0, hd=1 if holder >= 0 else 0, hb=holder,
                    no=0.0, dc=0, gr=1, st=-1.0, sn=0,
                ))
                continue

            if eid == BOX_HEAVY:
                bx, by = box_pos[eid]
                ent.append(schema.make_frame_ent(
                    id=eid, x=bx, y=by, z=0.0, h=0.0,
                    a=1, lk=0, hd=0, hb=-1, no=0.0, dc=0, gr=1, st=-1.0, sn=0,
                ))
                continue

            if eid == RAMP:
                ent.append(schema.make_frame_ent(
                    id=eid, x=RAMP_POS[0], y=RAMP_POS[1], z=0.0, h=0.0,
                    a=1, lk=0, hd=0, hb=-1, no=0.0, dc=0, gr=1, st=-1.0, sn=0,
                ))
                continue

            if eid == DECOY:
                ent.append(schema.make_frame_ent(
                    id=eid, x=DECOY_POS[0], y=DECOY_POS[1], z=0.0, h=0.0,
                    a=1, lk=0, hd=0, hb=-1, no=decoy_noise, dc=decoy_active,
                    gr=1, st=-1.0, sn=0,
                ))
                continue

            if eid in WALL_DEF:
                wx, wy, wh, _ = WALL_DEF[eid]
                active = wall_fragile_active if eid == WALL_FRAGILE else 1
                ent.append(schema.make_frame_ent(
                    id=eid, x=wx, y=wy, z=0.0, h=wh,
                    a=active, lk=1, hd=0, hb=-1, no=0.0, dc=0, gr=1, st=-1.0, sn=0,
                ))
                continue

            if eid == DOOR:
                dx, dy, dh, _ = DOOR_DEF
                ent.append(schema.make_frame_ent(
                    id=eid, x=dx, y=dy, z=0.0, h=dh,
                    a=door_active, lk=1, hd=0, hb=-1, no=0.0, dc=0, gr=1, st=-1.0, sn=0,
                ))
                continue

            # Inert padding walls: parked outside the arena, never active.
            ent.append(schema.make_frame_ent(
                id=eid, x=HALF + 4.0, y=HALF + 4.0, z=0.0, h=0.0,
                a=0, lk=1, hd=0, hb=-1, no=0.0, dc=0, gr=1, st=-1.0, sn=0,
            ))

        frames.append(schema.make_frame(
            t=t, phase=phase, sh=sh, ss=ss,
            seen_any=bool(seen_this_frame), fog=fog, ent=ent,
        ))

    meta = {
        "title": "Hide & Seek 2.0 -- synthetic showcase",
        "seed": int(seed),
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
# CLI
# --------------------------------------------------------------------------- #
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "web", "trajectories", "demo_trajectory.json",
)


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Hide & Seek 2.0 demo trajectory "
                    "(pure stdlib; clean square arena; showcases the 2.0 mechanics).",
    )
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help="output JSON path (default: viz/web/trajectories/demo_trajectory.json)")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed (default 7)")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                        help=f"total control steps / frames (default {DEFAULT_STEPS})")
    parser.add_argument("--prep", type=int, default=DEFAULT_PREP,
                        help=f"prep-phase length in steps (default {DEFAULT_PREP})")
    args = parser.parse_args(argv)

    prep = min(args.prep, max(1, args.steps - 1))
    doc = build_trajectory(seed=args.seed, steps=args.steps, prep=prep)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    schema.save_trajectory(doc, out)  # validates first

    size = os.path.getsize(out)
    print(f"wrote {out}")
    print(f"  frames={len(doc['frames'])}  entities={len(doc['entities'])}  bytes={size}")
    problems = schema.validate_trajectory(doc)
    print(f"  validate_trajectory -> {len(problems)} problem(s)"
          + ("" if not problems else ": " + "; ".join(problems)))
    return 0 if not problems else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
