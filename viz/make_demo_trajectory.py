"""
viz/make_demo_trajectory.py -- a SYNTHETIC Hide & Seek 2.0 episode generator.

PURE STDLIB ONLY (``math``, ``json``, ``random``, ``argparse``) -- importing or
running this module never touches jax or numpy, so the 3D viewer has something
fun to play the moment the repo is cloned, on any machine.

What it produces
----------------
A single ``hns2-traj`` v1 document (see :mod:`viz.schema`) describing a believable
~220-frame episode that deliberately SHOWCASES every 2.0 mechanic so the viewer is
entertaining to watch:

PREP phase (``t < prep_steps``)
    * Seekers are *held back and frozen* at a start pad (``lk=1``, no motion):
      they cannot act until the main phase, mirroring the env's prep gating.
    * Hiders move smoothly (eased lerp toward waypoints, heading from velocity)
      to grab **light boxes** (``hd=1`` / ``hb=<hider id>`` on the box, which then
      follows the hider) and shove them into a **barricade** line.
    * BOTH hiders then **cooperatively push the heavy box** together -- the two
      agents sit adjacent to ``box_heavy`` and move it as one, illustrating the
      cooperative-physics requirement (a single agent could not budge it).
    * Near the end of prep a **decoy switches on** (``dc=1``), its emitted noise
      ``no`` ramps up and *pulses*, spoofing a sensory target.

MAIN phase (``t >= prep_steps``)
    * Seekers are released (``lk=0``) and **chase** the hiders.
    * One seeker **climbs the ramp**: its ``z`` rises and ``gr`` flips to 0 on top
      (the anti box-surfing cosmetic -- airborne / not grounded).
    * The three **fog patches drift** slowly across the arena the whole episode.
    * Around mid-main the **door opens** (its ``a`` -> 0, "opened") creating a path.
    * One **wall breaks** (``a`` -> 0) the moment a seeker rams it at high speed.
    * At a dramatic moment a hider becomes **visible**: that hider's ``sn`` -> 1,
      the frame's ``seen_any`` -> true, and the score swings (hiders lose, seekers
      gain) for every remaining step.

All motion is clamped to the arena bounds, headings follow velocity, and agent
stamina (``st``) drains while sprinting and regenerates otherwise.

Entities (padded to a fixed ``E``)
----------------------------------
2 hiders, 2 seekers, 3 box_light, 1 box_heavy, 1 ramp, 1 decoy, 4 walls (one of
which breaks mid-episode), 1 door -- then padded with inactive ``wall`` slots so
``E`` is fixed and id-indexing is stable. Three fog patches drift.

CLI
---
::

    py -3 -m viz.make_demo_trajectory --out <path> --seed 7 --steps 220

Defaults write to ``viz/web/trajectories/demo_trajectory.json`` next to the
viewer. The document is built exclusively through the :mod:`viz.schema` builders
and saved via :func:`viz.schema.save_trajectory`, which validates before writing.
"""
from __future__ import annotations

import argparse
import math
import os
import random
from typing import Dict, List, Tuple

from viz import schema

# --------------------------------------------------------------------------- #
# Fixed episode geometry / tuning. Mirrors config.EnvConfig defaults closely so
# the synthetic episode looks like a "real" one, but we hard-code here to keep
# this module import-free of the project config (pure stdlib promise).
# --------------------------------------------------------------------------- #
ARENA_SIZE = 12.0          # arena spans [-6, +6] in x and y
HALF = ARENA_SIZE / 2.0
DT = 0.1                   # seconds per control step (playback speed)
DEFAULT_STEPS = 220        # ~220-frame episode
DEFAULT_PREP = 96          # prep-phase length (steps)

# Per-type collision radius / half-extent and mass (cosmetic, drives the viewer).
SIZE = {
    "hider": 0.40, "seeker": 0.40,
    "box_light": 0.55, "box_heavy": 0.85,
    "ramp": 0.90, "decoy": 0.35,
    "wall": 1.20, "door": 1.10,
}
MASS = {
    "hider": 1.0, "seeker": 1.0,
    "box_light": 1.0, "box_heavy": 6.0,
    "ramp": 3.0, "decoy": 0.5,
    "wall": 1.0e6, "door": 1.0e6,
}

# Fixed total entity count. We declare the "interesting" entities explicitly and
# then pad with inactive walls up to this E so id-indexing stays stable and the
# viewer always sees the same slot layout.
TOTAL_ENTITIES = 18


# --------------------------------------------------------------------------- #
# Small stdlib math helpers (no numpy).
# --------------------------------------------------------------------------- #
def clamp(v: float, lo: float, hi: float) -> float:
    """Clamp ``v`` into ``[lo, hi]``."""
    return lo if v < lo else (hi if v > hi else v)


def clamp_arena(x: float, y: float, margin: float = 0.2) -> Tuple[float, float]:
    """Clamp an (x, y) point to the playable arena, keeping a small wall margin."""
    lim = HALF - margin
    return clamp(x, -lim, lim), clamp(y, -lim, lim)


def smoothstep(a: float, b: float, t: float) -> float:
    """Eased interpolation from ``a`` to ``b`` for ``t`` in [0, 1] (smooth ends)."""
    t = clamp(t, 0.0, 1.0)
    s = t * t * (3.0 - 2.0 * t)
    return a + (b - a) * s


def lerp(a: float, b: float, t: float) -> float:
    """Plain linear interpolation from ``a`` to ``b`` for ``t`` in [0, 1]."""
    return a + (b - a) * clamp(t, 0.0, 1.0)


def heading_from(dx: float, dy: float, fallback: float) -> float:
    """Heading (radians) implied by a velocity vector; ``fallback`` if ~zero."""
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return fallback
    return math.atan2(dy, dx)


def phase_for(t: int, prep_steps: int) -> str:
    """Return the schema phase string for control step ``t``."""
    return "prep" if t < prep_steps else "main"


# --------------------------------------------------------------------------- #
# Entity table. Order is agents-first (hiders, seekers) then props, mirroring the
# env's slot layout so an agent's id is identical across index spaces. Slots are
# padded with inactive walls up to TOTAL_ENTITIES.
# --------------------------------------------------------------------------- #
# Each tuple: (type, team, is_decoy_true_identity). ``is_decoy`` is the GOD-VIEW
# truth -- the real decoy carries True; everything else False (even though the
# decoy *spoofs* other types in the agents' local view, that spoofing is not
# represented in this god-view trajectory).
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
    ("wall", schema.TEAM_NONE, False),       # 10 wall N
    ("wall", schema.TEAM_NONE, False),       # 11 wall E
    ("wall", schema.TEAM_NONE, False),       # 12 wall S
    ("wall", schema.TEAM_NONE, False),       # 13 wall (fragile -- breaks)
    ("door", schema.TEAM_NONE, False),       # 14 door (opens mid-main)
]

# Named id aliases for readability below.
HIDER_A, HIDER_B = 0, 1
SEEKER_A, SEEKER_B = 2, 3
BOX_L1, BOX_L2, BOX_L3 = 4, 5, 6
BOX_HEAVY = 7
RAMP = 8
DECOY = 9
WALL_N, WALL_E, WALL_S, WALL_FRAGILE = 10, 11, 12, 13
DOOR = 14


def build_entities() -> List[Dict]:
    """Build the STATIC per-slot ``entities`` table, padded to TOTAL_ENTITIES.

    Returns a list of length :data:`TOTAL_ENTITIES`; ids equal their index (the
    schema validator enforces this). Padding slots are inactive ``wall`` entities
    (they are simply never marked active in any frame).
    """
    entities: List[Dict] = []
    for i, (etype, team, is_decoy) in enumerate(_ENTITY_SPEC):
        entities.append(
            schema.make_entity_meta(
                id=i, type=etype, team=team,
                size=SIZE[etype], mass=MASS[etype], is_decoy=is_decoy,
            )
        )
    # Pad remaining slots with inert walls so E is fixed at TOTAL_ENTITIES.
    for i in range(len(_ENTITY_SPEC), TOTAL_ENTITIES):
        entities.append(
            schema.make_entity_meta(
                id=i, type="wall", team=schema.TEAM_NONE,
                size=SIZE["wall"], mass=MASS["wall"], is_decoy=False,
            )
        )
    return entities


# --------------------------------------------------------------------------- #
# Fixed static prop placements (arena coords). Walls form three sides of a rough
# enclosure plus a fragile inner wall; the door sits in the remaining gap.
# --------------------------------------------------------------------------- #
WALL_POS = {
    WALL_N: (0.0, 4.6),
    WALL_E: (4.6, 0.0),
    WALL_S: (0.0, -4.6),
    WALL_FRAGILE: (-1.8, 1.4),   # interior fragile wall a seeker will ram
}
DOOR_POS = (-4.6, 0.0)
RAMP_POS = (3.4, 3.0)
DECOY_POS = (-3.2, -3.2)

# Barricade target line (where hiders stack the light boxes during prep).
BARRICADE = {
    BOX_L1: (-3.6, 2.6),
    BOX_L2: (-2.6, 2.9),
    BOX_L3: (-1.6, 2.6),
}
# Light boxes start scattered; hiders fetch them.
BOX_START = {
    BOX_L1: (-4.4, -2.0),
    BOX_L2: (-3.4, -3.6),
    BOX_L3: (1.2, -3.0),
}
# Heavy box: starts center-ish, gets cooperatively shoved aside during prep.
HEAVY_START = (0.6, 0.4)
HEAVY_END = (-3.0, 3.4)

# Seeker start pad (frozen during prep).
SEEKER_START = {SEEKER_A: (4.6, -3.6), SEEKER_B: (5.0, -2.4)}
# Hider spawn.
HIDER_START = {HIDER_A: (-2.0, -1.5), HIDER_B: (-3.0, -0.5)}


# --------------------------------------------------------------------------- #
# Per-agent motion planner.
#
# We script each hider as a sequence of timed "legs". A leg moves the agent from
# its current position toward a target between two step indices using a smooth
# ease. While a hider is on a "carry" leg for a given box, that box rides along
# (held). This keeps the demo deterministic and readable without any physics.
# --------------------------------------------------------------------------- #
class AgentPlan:
    """A scripted, eased waypoint plan for a single agent.

    Each leg is ``(t0, t1, (x, y), carry_id, sprint)``:

    * ``t0, t1``     -- inclusive/exclusive step window the leg is active over.
    * ``(x, y)``     -- world target the agent eases toward across the window.
    * ``carry_id``   -- entity id the agent holds during this leg (or ``-1``).
    * ``sprint``     -- whether the agent is sprinting (drains stamina) this leg.

    Between/after legs the agent simply holds its last position. Position is the
    eased interpolation from the position at ``t0`` to the target at ``t1``.
    """

    def __init__(self, start: Tuple[float, float]):
        self.start = start
        self.legs: List[Tuple[int, int, Tuple[float, float], int, bool]] = []

    def add(self, t0: int, t1: int, target: Tuple[float, float],
            carry: int = -1, sprint: bool = False) -> "AgentPlan":
        """Append a leg; returns self for chaining."""
        self.legs.append((t0, t1, target, carry, sprint))
        return self

    def _leg_origin(self, idx: int) -> Tuple[float, float]:
        """World position at the start of leg ``idx`` (end of the previous leg)."""
        if idx == 0:
            return self.start
        return self.legs[idx - 1][2]

    def sample(self, t: int) -> Tuple[float, float, int, bool, bool]:
        """Resolve the plan at step ``t``.

        Returns ``(x, y, carry_id, sprint, moving)`` where ``moving`` indicates
        the agent is actively translating this step (used for heading + stamina).
        """
        # Before the first leg: sit at start.
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

        # After the last leg: hold final target (carry persists if it was a
        # "park" leg, but by convention we drop the carry once parked).
        last = self.legs[-1]
        return last[2][0], last[2][1], -1, False, False


def build_hider_plans(prep: int, steps: int) -> Tuple[AgentPlan, AgentPlan]:
    """Script the two hiders' prep-phase choreography and main-phase fleeing.

    Hider A fetches light boxes 1 & 2 to the barricade; hider B fetches light box
    3; both then converge to cooperatively push the heavy box; in the main phase
    they flee from the released seekers (B is the one that gets spotted).
    """
    pa = AgentPlan(HIDER_START[HIDER_A])
    pb = AgentPlan(HIDER_START[HIDER_B])

    # --- PREP choreography ------------------------------------------------- #
    # Hider A: grab box 1, haul to barricade, then box 2, haul to barricade.
    pa.add(4, 16, BOX_START[BOX_L1])                       # go to box 1
    pa.add(16, 30, BARRICADE[BOX_L1], carry=BOX_L1)        # haul box 1 (held)
    pa.add(30, 42, BOX_START[BOX_L2])                      # go to box 2
    pa.add(42, 56, BARRICADE[BOX_L2], carry=BOX_L2)        # haul box 2 (held)

    # Hider B: grab box 3, haul to barricade.
    pb.add(6, 24, BOX_START[BOX_L3])                       # go to box 3
    pb.add(24, 40, BARRICADE[BOX_L3], carry=BOX_L3)        # haul box 3 (held)

    # Both converge on the heavy box and cooperatively push it (adjacent).
    # They approach from two sides so the viewer reads "two agents on the box".
    heavy_ax = HEAVY_START[0] - 0.9
    heavy_bx = HEAVY_START[0] + 0.9
    pa.add(56, 66, (heavy_ax, HEAVY_START[1]))             # A to heavy box (left)
    pb.add(40, 54, (heavy_bx, HEAVY_START[1]))             # B to heavy box (right)
    # Cooperative shove: both move in lockstep, flanking the heavy box.
    pa.add(66, 88, (HEAVY_END[0] - 0.9, HEAVY_END[1]), carry=-1)
    pb.add(66, 88, (HEAVY_END[0] + 0.9, HEAVY_END[1]), carry=-1)
    # Settle behind the barricade for the rest of prep.
    pa.add(88, prep, (-2.4, 2.0))
    pb.add(88, prep, (-1.2, 1.6))

    # --- MAIN phase: flee ------------------------------------------------- #
    # Hider A weaves toward the (soon-to-open) door, sprinting.
    pa.add(prep, prep + 40, (-3.6, -0.4), sprint=True)
    pa.add(prep + 40, prep + 90, (-4.8, -1.8), sprint=True)
    pa.add(prep + 90, steps, (-3.2, -2.6))
    # Hider B breaks for open arena and is the one that gets spotted.
    pb.add(prep, prep + 36, (1.6, 1.2), sprint=True)
    pb.add(prep + 36, prep + 80, (2.8, -1.0), sprint=True)
    pb.add(prep + 80, steps, (0.4, -2.4), sprint=True)

    return pa, pb


def build_seeker_plans(prep: int, steps: int) -> Tuple[AgentPlan, AgentPlan]:
    """Script the two seekers: frozen in prep, then a ramp-climber and a rammer.

    Seeker A charges the fragile interior wall (rams it) then chases. Seeker B
    climbs the ramp (z rises) then pursues hider B.
    """
    pa = AgentPlan(SEEKER_START[SEEKER_A])
    pb = AgentPlan(SEEKER_START[SEEKER_B])

    # Frozen during prep: a single trivial leg keeps them parked at the pad.
    pa.add(0, prep, SEEKER_START[SEEKER_A])
    pb.add(0, prep, SEEKER_START[SEEKER_B])

    # --- MAIN: seeker A rams the fragile wall, then hunts hider A ---------- #
    # Charge straight at the fragile wall at high speed (short leg => fast).
    pa.add(prep, prep + 18, WALL_POS[WALL_FRAGILE], sprint=True)
    pa.add(prep + 18, prep + 60, (-3.2, 0.2), sprint=True)
    pa.add(prep + 60, steps, (-3.6, -2.0), sprint=True)

    # --- MAIN: seeker B climbs the ramp, then chases hider B -------------- #
    pb.add(prep, prep + 20, RAMP_POS, sprint=True)          # to ramp foot
    pb.add(prep + 20, prep + 40, (RAMP_POS[0] - 0.4, RAMP_POS[1] - 0.6))  # up & over
    pb.add(prep + 40, prep + 78, (2.4, 0.2), sprint=True)   # down, pursue
    pb.add(prep + 78, steps, (1.0, -2.0), sprint=True)      # close on hider B

    return pa, pb


# --------------------------------------------------------------------------- #
# Fog patches that slowly drift.
# --------------------------------------------------------------------------- #
def fog_at(t: int, steps: int, rng: random.Random) -> List[List[float]]:
    """Return the 3 fog patch ``[x, y, r]`` triples at step ``t``.

    Each patch drifts along a slow looping path so the cover keeps shifting; radii
    breathe gently. Clamped to the arena. Deterministic given the seeded ``rng``
    used only to fix the per-patch phase offsets (computed once by the caller and
    captured here via closure-free recomputation -- so we instead pass fixed
    params). For determinism we derive phases from fixed constants below.
    """
    frac = t / max(1, steps - 1)
    patches: List[List[float]] = []
    # Three patches with distinct centers, drift directions and breathing phases.
    specs = [
        (-3.0, 1.0, 0.9, 0.6, 0.0),    # cx, cy, ax, ay, phase
        (2.5, 2.0, -0.8, 0.7, 1.7),
        (0.5, -3.0, 0.7, -0.9, 3.1),
    ]
    base_r = 2.5
    for (cx, cy, ax, ay, ph) in specs:
        ang = 2.0 * math.pi * frac + ph
        x = cx + ax * math.sin(ang)
        y = cy + ay * math.cos(ang * 0.8 + ph)
        r = base_r + 0.4 * math.sin(ang * 1.3 + ph)
        x, y = clamp_arena(x, y, margin=0.0)
        patches.append([x, y, max(1.2, r)])
    return patches


# --------------------------------------------------------------------------- #
# The main episode builder.
# --------------------------------------------------------------------------- #
def build_trajectory(seed: int, steps: int, prep: int) -> Dict:
    """Build the full synthetic trajectory document.

    Parameters
    ----------
    seed:
        RNG seed (drives only the tiny cosmetic jitter; the choreography itself
        is deterministic so the showcased mechanics always fire on cue).
    steps:
        Total number of control steps / frames.
    prep:
        Prep-phase length (steps with ``phase == "prep"``).

    Returns
    -------
    A validated-on-save ``hns2-traj`` v1 document (dict).
    """
    rng = random.Random(seed)
    entities = build_entities()
    E = len(entities)

    hider_a_plan, hider_b_plan = build_hider_plans(prep, steps)
    seeker_a_plan, seeker_b_plan = build_seeker_plans(prep, steps)
    plans = {
        HIDER_A: hider_a_plan, HIDER_B: hider_b_plan,
        SEEKER_A: seeker_a_plan, SEEKER_B: seeker_b_plan,
    }

    # --- Scripted event timings (main-phase, in absolute steps). ----------- #
    decoy_on_step = prep - 22            # decoy lights up late in prep
    wall_break_step = prep + 16          # seeker A's ram connects ~here
    door_open_step = prep + (steps - prep) // 2   # door opens mid-main
    seen_step = prep + int((steps - prep) * 0.62)  # hider B spotted late-main
    ramp_up_t0 = prep + 20               # seeker B on top of ramp window
    ramp_up_t1 = prep + 40

    # --- Per-agent running state we must carry across frames. -------------- #
    headings = {HIDER_A: 0.0, HIDER_B: math.pi, SEEKER_A: math.pi, SEEKER_B: math.pi}
    prev_pos = {
        HIDER_A: HIDER_START[HIDER_A], HIDER_B: HIDER_START[HIDER_B],
        SEEKER_A: SEEKER_START[SEEKER_A], SEEKER_B: SEEKER_START[SEEKER_B],
    }
    stamina = {HIDER_A: 1.0, HIDER_B: 1.0, SEEKER_A: 1.0, SEEKER_B: 1.0}

    # Cumulative scores. During prep both stay 0 (env zeroes prep reward). In main
    # they tick: hiders gain (+) while unseen; once a hider is spotted seekers gain.
    sh = 0.0  # cumulative hider score
    ss = 0.0  # cumulative seeker score

    # Light-box "parked" positions become permanent once hauled (so a box that
    # was carried to the barricade stays there afterwards). We track the last
    # known box position so a released box freezes in place.
    box_pos = {
        BOX_L1: BOX_START[BOX_L1], BOX_L2: BOX_START[BOX_L2],
        BOX_L3: BOX_START[BOX_L3], BOX_HEAVY: HEAVY_START,
    }

    frames: List[Dict] = []

    for t in range(steps):
        phase = phase_for(t, prep)
        is_main = phase == "main"

        # ---- Resolve agent kinematics for this frame. -------------------- #
        agent_state: Dict[int, Dict] = {}
        # Which box (if any) each hider is currently carrying this frame.
        carried_by: Dict[int, int] = {}   # box_id -> hider_id

        for aid, plan in plans.items():
            x, y, carry, sprint, moving = plan.sample(t)
            x, y = clamp_arena(x, y)

            # Heading from velocity; keep previous heading when stationary.
            px, py = prev_pos[aid]
            dx, dy = x - px, y - py
            headings[aid] = heading_from(dx, dy, headings[aid])
            prev_pos[aid] = (x, y)

            # Stamina: drain while sprinting, regenerate otherwise (per-step,
            # scaled so a long sprint visibly empties the bar but never < 0).
            if sprint and moving:
                stamina[aid] = clamp(stamina[aid] - 0.020, 0.05, 1.0)
            else:
                stamina[aid] = clamp(stamina[aid] + 0.010, 0.05, 1.0)

            agent_state[aid] = {
                "x": x, "y": y, "h": headings[aid],
                "sprint": sprint, "moving": moving, "st": stamina[aid],
            }
            if carry >= 0:
                carried_by[carry] = aid

        # ---- Seeker freeze during prep (locked at the start pad). -------- #
        seekers_frozen = not is_main

        # ---- Seeker B ramp climb: z rises, grounded flips off on top. ---- #
        seeker_b = agent_state[SEEKER_B]
        sb_z = 0.0
        sb_grounded = 1
        if ramp_up_t0 <= t < ramp_up_t1:
            # Triangular elevation profile: rise to the apex then descend.
            mid = (ramp_up_t0 + ramp_up_t1) / 2.0
            if t <= mid:
                climb = (t - ramp_up_t0) / max(1, (mid - ramp_up_t0))
            else:
                climb = 1.0 - (t - mid) / max(1, (ramp_up_t1 - mid))
            sb_z = round(1.2 * clamp(climb, 0.0, 1.0), 4)
            sb_grounded = 0 if sb_z > 0.05 else 1

        # ---- Decoy activation + pulsing noise. --------------------------- #
        decoy_active = 1 if (decoy_on_step <= t) else 0
        if decoy_active:
            # Ramp the noise up over ~12 steps then pulse around a high baseline.
            since = t - decoy_on_step
            ramp = clamp(since / 12.0, 0.0, 1.0)
            pulse = 0.15 * (0.5 + 0.5 * math.sin(since * 0.7))
            decoy_noise = clamp(0.6 * ramp + pulse, 0.0, 1.0)
        else:
            decoy_noise = 0.0

        # ---- Destructible fragile wall: active until the ram connects. --- #
        wall_fragile_active = 0 if (is_main and t >= wall_break_step) else 1

        # ---- Door: active(closed) until it opens mid-main. --------------- #
        door_active = 0 if (is_main and t >= door_open_step) else 1
        # Door-progress cosmetic: nudge the door leaf aside once opened.
        door_x, door_y = DOOR_POS
        if door_active == 0:
            door_y = DOOR_POS[1] + 1.6   # leaf swung aside, opening the gap

        # ---- "Seen" event: hider B spotted late-main -> score swing. ----- #
        seen_this_frame = is_main and t >= seen_step
        hider_b_seen = 1 if seen_this_frame else 0

        # ---- Score accounting (cumulative). ------------------------------ #
        if is_main:
            if seen_this_frame:
                # A hider is exposed: seekers gain, hiders bleed.
                ss += 1.0
                sh -= 1.0
            else:
                # Hiders survive unseen this step: they gain.
                sh += 1.0
                ss -= 1.0

        # ---- Update carried light-box positions to follow their hider. --- #
        for box_id in (BOX_L1, BOX_L2, BOX_L3):
            if box_id in carried_by:
                hid = carried_by[box_id]
                # Box rides just in front of / on the hider.
                box_pos[box_id] = (agent_state[hid]["x"], agent_state[hid]["y"])
        # Heavy box: cooperatively pushed during the joint shove window (66..88).
        if 66 <= t < 88:
            # Sits at the midpoint of the two flanking hiders -> reads as coop.
            mx = (agent_state[HIDER_A]["x"] + agent_state[HIDER_B]["x"]) / 2.0
            my = (agent_state[HIDER_A]["y"] + agent_state[HIDER_B]["y"]) / 2.0
            box_pos[BOX_HEAVY] = (mx, my)
        elif t >= 88:
            box_pos[BOX_HEAVY] = HEAVY_END

        # ---- Fog. -------------------------------------------------------- #
        fog = fog_at(t, steps, rng)

        # ---------------------------------------------------------------- #
        # Assemble the per-entity records, id-aligned with ``entities``.
        # ---------------------------------------------------------------- #
        ent: List[Dict] = []
        for eid in range(E):
            etype = entities[eid]["type"]

            # ---- Agents (hiders 0-1, seekers 2-3). ---------------------- #
            if eid in (HIDER_A, HIDER_B, SEEKER_A, SEEKER_B):
                st = agent_state[eid]
                is_seeker = eid in (SEEKER_A, SEEKER_B)

                z = 0.0
                grounded = 1
                locked = 0
                if is_seeker:
                    if seekers_frozen:
                        locked = 1   # held back / frozen at start pad
                    if eid == SEEKER_B:
                        z = sb_z
                        grounded = sb_grounded

                # Held flags: a hider holding a light box this frame.
                holding_box = any(
                    carried_by.get(b) == eid for b in (BOX_L1, BOX_L2, BOX_L3)
                )
                # Noise: agents emit a little noise while sprinting (movement).
                noise = 0.25 if (st["sprint"] and st["moving"]) else 0.05

                sn = hider_b_seen if eid == HIDER_B else 0

                ent.append(schema.make_frame_ent(
                    id=eid, x=st["x"], y=st["y"], z=z, h=st["h"],
                    a=1, lk=locked, hd=0, hb=-1,
                    no=noise, dc=0, gr=grounded,
                    st=st["st"], sn=sn,
                ))
                continue

            # ---- Light boxes (held while a hider carries them). --------- #
            if eid in (BOX_L1, BOX_L2, BOX_L3):
                bx, by = box_pos[eid]
                holder = carried_by.get(eid, -1)
                held = 1 if holder >= 0 else 0
                ent.append(schema.make_frame_ent(
                    id=eid, x=bx, y=by, z=0.0, h=0.0,
                    a=1, lk=0, hd=held, hb=holder,
                    no=0.0, dc=0, gr=1, st=-1.0, sn=0,
                ))
                continue

            # ---- Heavy box (coop-pushed; never "held" by a single agent). #
            if eid == BOX_HEAVY:
                bx, by = box_pos[eid]
                ent.append(schema.make_frame_ent(
                    id=eid, x=bx, y=by, z=0.0, h=0.0,
                    a=1, lk=0, hd=0, hb=-1,
                    no=0.0, dc=0, gr=1, st=-1.0, sn=0,
                ))
                continue

            # ---- Ramp (static). ----------------------------------------- #
            if eid == RAMP:
                ent.append(schema.make_frame_ent(
                    id=eid, x=RAMP_POS[0], y=RAMP_POS[1], z=0.0, h=0.0,
                    a=1, lk=0, hd=0, hb=-1,
                    no=0.0, dc=0, gr=1, st=-1.0, sn=0,
                ))
                continue

            # ---- Decoy (activates + pulses noise late prep). ------------ #
            if eid == DECOY:
                ent.append(schema.make_frame_ent(
                    id=eid, x=DECOY_POS[0], y=DECOY_POS[1], z=0.0, h=0.0,
                    a=1, lk=0, hd=0, hb=-1,
                    no=decoy_noise, dc=decoy_active, gr=1, st=-1.0, sn=0,
                ))
                continue

            # ---- Walls (one fragile breaks; padding walls stay inactive). #
            if eid in WALL_POS:
                wx, wy = WALL_POS[eid]
                active = 1
                if eid == WALL_FRAGILE:
                    active = wall_fragile_active
                ent.append(schema.make_frame_ent(
                    id=eid, x=wx, y=wy, z=0.0, h=0.0,
                    a=active, lk=1, hd=0, hb=-1,
                    no=0.0, dc=0, gr=1, st=-1.0, sn=0,
                ))
                continue

            # ---- Door (opens mid-main). --------------------------------- #
            if eid == DOOR:
                ent.append(schema.make_frame_ent(
                    id=eid, x=door_x, y=door_y, z=0.0, h=0.0,
                    a=door_active, lk=1, hd=0, hb=-1,
                    no=0.0, dc=0, gr=1, st=-1.0, sn=0,
                ))
                continue

            # ---- Padding slots (inactive walls parked off to the side). - #
            ent.append(schema.make_frame_ent(
                id=eid, x=HALF - 0.3, y=-(HALF - 0.3) + 0.6 * (eid - len(_ENTITY_SPEC)),
                z=0.0, h=0.0,
                a=0, lk=1, hd=0, hb=-1,
                no=0.0, dc=0, gr=1, st=-1.0, sn=0,
            ))

        frames.append(schema.make_frame(
            t=t, phase=phase, sh=sh, ss=ss,
            seen_any=bool(seen_this_frame), fog=fog, ent=ent,
        ))

    # --- Assemble the document. ------------------------------------------- #
    meta = {
        "title": "Hide & Seek 2.0 -- synthetic showcase",
        "seed": int(seed),
        "arena_size": ARENA_SIZE,
        "dt": DT,
        "max_steps": int(steps),
        "prep_steps": int(prep),
        "entity_types": list(schema.ENTITY_TYPES),
        "max_agents": 6,            # mirrors config default (3 hiders + 3 seekers)
        "max_entities": E,
    }
    return schema.make_trajectory(meta, entities, frames)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
DEFAULT_OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "web", "trajectories", "demo_trajectory.json",
)


def main(argv: List[str] | None = None) -> int:
    """Generate the demo trajectory and write it to disk (validated on save).

    Parameters
    ----------
    argv:
        Optional argument list (defaults to ``sys.argv[1:]`` via argparse).

    Returns
    -------
    Process exit code (0 on success).
    """
    parser = argparse.ArgumentParser(
        description="Generate a synthetic Hide & Seek 2.0 demo trajectory "
                    "(pure stdlib; showcases the 2.0 mechanics).",
    )
    parser.add_argument(
        "--out", default=DEFAULT_OUT,
        help="output JSON path (default: viz/web/trajectories/demo_trajectory.json)",
    )
    parser.add_argument("--seed", type=int, default=7, help="RNG seed (default 7)")
    parser.add_argument(
        "--steps", type=int, default=DEFAULT_STEPS,
        help=f"total control steps / frames (default {DEFAULT_STEPS})",
    )
    parser.add_argument(
        "--prep", type=int, default=DEFAULT_PREP,
        help=f"prep-phase length in steps (default {DEFAULT_PREP})",
    )
    args = parser.parse_args(argv)

    prep = min(args.prep, max(1, args.steps - 1))
    doc = build_trajectory(seed=args.seed, steps=args.steps, prep=prep)

    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    # save_trajectory validates first and raises ValueError on any problem.
    schema.save_trajectory(doc, out)

    size = os.path.getsize(out)
    print(f"wrote {out}")
    print(f"  frames={len(doc['frames'])}  entities={len(doc['entities'])}  "
          f"bytes={size}")
    # Re-validate explicitly and report (belt and braces).
    problems = schema.validate_trajectory(doc)
    print(f"  validate_trajectory -> {len(problems)} problem(s)"
          + ("" if not problems else ": " + "; ".join(problems)))
    return 0 if not problems else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
