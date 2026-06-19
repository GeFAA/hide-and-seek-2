"""
viz/make_demo_trajectory.py -- a REAL, deterministic micro-simulation of Hide &
Seek that produces the viewer's demo trajectories. PURE STDLIB (math/json/os/
argparse/random); never imports jax/numpy/config.

Why a simulation (not scripted waypoints)?
------------------------------------------
The old generator moved agents/boxes along hand-scripted eased waypoints with NO
collisions, so boxes clipped into each other, agents walked through walls and each
other, the motion looked fake, nobody actually "saw" anyone, and there was no clear
winner. This module instead runs a tiny but real 2-D simulation:

* **Collision resolution** (circle/circle + circle/AABB-wall) every step -> nothing
  ever overlaps or passes through a wall. Box mass makes the heavy box need two
  pushers (cooperation is emergent, not scripted).
* **Steering movement** (smoothed velocity toward a goal, capped speed) -> smooth,
  natural motion; agents face their direction of travel.
* **Perception** -> a seeker *sees* a hider only within range, inside its vision
  cone, and with line-of-sight not blocked by a wall or box. Seekers chase what
  they see; hiders flee the nearest seeker.
* **A clear outcome** -> seekers win by tagging a hider they can see; otherwise the
  hiders win when time runs out. The result is stored in the trajectory.

The output conforms to ``viz/schema.py`` (hns2-traj v1) plus a top-level
``outcome`` object the viewer uses to show a win banner. Every produced trajectory
is checked by :func:`validate_simulation` (no overlaps, no wall penetration, in
bounds, smooth, decided) before it is written.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Tuple

from viz import schema

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
ARENA = 12.0
HALF = ARENA / 2.0
DT = 0.1
WALL_T = 0.6                      # wall thickness (full)
VISION_RANGE = 5.2               # local perception so cover/distance matter
VISION_HALF_CONE = math.radians(58.0)
TAG_DIST = 1.2                   # seeker within this of a seen hider -> caught
AGENT_SPEED = 2.75               # units / second (hiders)
SEEKER_SPEED = 2.9               # seekers a touch faster so an open chase resolves

RADIUS = {
    "hider": 0.40, "seeker": 0.40,
    "box_light": 0.55, "box_heavy": 0.80,
    "ramp": 0.95, "decoy": 0.38,
}
MASS = {
    "hider": 1.0, "seeker": 1.0,
    "box_light": 1.7, "box_heavy": 7.5,   # heavy box ~ needs two agents
}


# --------------------------------------------------------------------------- #
# Small vector helpers
# --------------------------------------------------------------------------- #
def clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else (hi if v > hi else v)


def vlen(x: float, y: float) -> float:
    return math.hypot(x, y)


def vnorm(x: float, y: float) -> Tuple[float, float]:
    l = math.hypot(x, y)
    return (0.0, 0.0) if l < 1e-9 else (x / l, y / l)


def seg_circle_blocked(ax: float, ay: float, bx: float, by: float,
                       cx: float, cy: float, r: float) -> bool:
    """True if segment a->b passes within ``r`` of point c (circle blocks LOS)."""
    dx, dy = bx - ax, by - ay
    l2 = dx * dx + dy * dy
    if l2 < 1e-9:
        return vlen(cx - ax, cy - ay) < r
    t = clamp(((cx - ax) * dx + (cy - ay) * dy) / l2, 0.0, 1.0)
    px, py = ax + t * dx, ay + t * dy
    return vlen(cx - px, cy - py) < r


def seg_aabb_blocked(ax: float, ay: float, bx: float, by: float,
                     x0: float, y0: float, x1: float, y1: float) -> bool:
    """True if segment a->b intersects the axis-aligned box [x0,x1]x[y0,y1]
    (slab method)."""
    dx, dy = bx - ax, by - ay
    tmin, tmax = 0.0, 1.0
    for p, d, lo, hi in ((ax, dx, x0, x1), (ay, dy, y0, y1)):
        if abs(d) < 1e-9:
            if p < lo or p > hi:
                return False
        else:
            t1 = (lo - p) / d
            t2 = (hi - p) / d
            if t1 > t2:
                t1, t2 = t2, t1
            tmin = max(tmin, t1)
            tmax = min(tmax, t2)
            if tmin > tmax:
                return False
    return True


# --------------------------------------------------------------------------- #
# Entities & walls
# --------------------------------------------------------------------------- #
class Ent:
    __slots__ = ("id", "type", "team", "x", "y", "vx", "vy", "r", "mass",
                 "movable", "heading", "held_by", "locked", "noise", "dc",
                 "stam", "z", "grounded", "sn", "active")

    def __init__(self, id, type, team, x, y):
        self.id = id
        self.type = type
        self.team = team
        self.x = x
        self.y = y
        self.vx = 0.0
        self.vy = 0.0
        self.r = RADIUS.get(type, 0.4)
        self.mass = MASS.get(type, 1e6)      # walls/ramp/decoy effectively static
        self.movable = type in ("hider", "seeker", "box_light", "box_heavy")
        self.heading = 0.0
        self.held_by = -1
        self.locked = 0
        self.noise = 0.0
        self.dc = 0
        self.stam = 1.0
        self.z = 0.0
        self.grounded = 1
        self.sn = 0
        self.active = 1


class Wall:
    """An axis-aligned wall slab. Carries both an AABB (for collision) and the
    viewer entity params (center, half-length, heading)."""
    __slots__ = ("id", "x0", "y0", "x1", "y1", "active", "is_door", "cx", "cy",
                 "size", "heading", "_ent")

    def __init__(self, id, cx, cy, length, horizontal, is_door=False):
        self.id = id
        self.is_door = is_door
        self.cx = cx
        self.cy = cy
        self.heading = 0.0 if horizontal else math.pi / 2
        self.size = length / 2.0                 # viewer renders length = size*2
        hw = (length / 2.0) if horizontal else (WALL_T / 2.0)
        hh = (WALL_T / 2.0) if horizontal else (length / 2.0)
        self.x0, self.x1 = cx - hw, cx + hw
        self.y0, self.y1 = cy - hh, cy + hh
        self.active = 1


def border_walls(start_id: int, door: bool = True) -> List[Wall]:
    """Four border walls around the arena; the west side has a central door gap."""
    w = []
    w.append(Wall(start_id + 0, 0.0, HALF, ARENA + WALL_T, True))      # N
    w.append(Wall(start_id + 1, 0.0, -HALF, ARENA + WALL_T, True))     # S
    w.append(Wall(start_id + 2, HALF, 0.0, ARENA + WALL_T, False))     # E
    if door:
        w.append(Wall(start_id + 3, -HALF, 3.25, 5.5, False))          # W upper
        w.append(Wall(start_id + 4, -HALF, -3.25, 5.5, False))         # W lower
        w.append(Wall(start_id + 5, -HALF, 0.0, 2.0, False, is_door=True))  # door
    else:
        w.append(Wall(start_id + 3, -HALF, 0.0, ARENA + WALL_T, False))  # W full
    return w


# --------------------------------------------------------------------------- #
# Collision resolution
# --------------------------------------------------------------------------- #
def _push_out_wall(e: Ent, w: Wall) -> None:
    if not w.active:
        return
    cx = clamp(e.x, w.x0, w.x1)
    cy = clamp(e.y, w.y0, w.y1)
    dx, dy = e.x - cx, e.y - cy
    d2 = dx * dx + dy * dy
    if d2 >= e.r * e.r:
        return
    if d2 > 1e-12:
        d = math.sqrt(d2)
        nx, ny = dx / d, dy / d
        push = e.r - d
    else:
        # center inside the slab: eject along the nearest face
        left, right = e.x - w.x0, w.x1 - e.x
        bottom, top = e.y - w.y0, w.y1 - e.y
        m = min(left, right, bottom, top)
        if m == left:
            nx, ny, push = -1.0, 0.0, left + e.r
        elif m == right:
            nx, ny, push = 1.0, 0.0, right + e.r
        elif m == bottom:
            nx, ny, push = 0.0, -1.0, bottom + e.r
        else:
            nx, ny, push = 0.0, 1.0, top + e.r
    e.x += nx * push
    e.y += ny * push
    vn = e.vx * nx + e.vy * ny           # kill velocity into the wall
    if vn < 0:
        e.vx -= vn * nx
        e.vy -= vn * ny


def _separate(a: Ent, b: Ent) -> None:
    dx, dy = b.x - a.x, b.y - a.y
    d = math.hypot(dx, dy)
    mind = a.r + b.r
    if d >= mind:
        return
    ima = (1.0 / a.mass) if a.movable else 0.0
    imb = (1.0 / b.mass) if b.movable else 0.0
    s = ima + imb
    if s <= 0:
        return
    if d > 1e-9:
        nx, ny = dx / d, dy / d
        overlap = mind - d
    else:
        nx, ny, overlap = 1.0, 0.0, mind     # exact overlap -> deterministic split
    a.x -= nx * overlap * (ima / s)
    a.y -= ny * overlap * (ima / s)
    b.x += nx * overlap * (imb / s)
    b.y += ny * overlap * (imb / s)


def resolve_collisions(movers: List[Ent], walls: List[Wall], iters: int = 5) -> None:
    for _ in range(iters):
        for i in range(len(movers)):
            for j in range(i + 1, len(movers)):
                _separate(movers[i], movers[j])
        for e in movers:
            for w in walls:
                _push_out_wall(e, w)


# --------------------------------------------------------------------------- #
# Steering & perception
# --------------------------------------------------------------------------- #
def steer_to(e: Ent, tx: float, ty: float, maxspeed: float, gain: float = 0.3) -> None:
    nx, ny = vnorm(tx - e.x, ty - e.y)
    dvx, dvy = nx * maxspeed, ny * maxspeed
    e.vx += (dvx - e.vx) * gain
    e.vy += (dvy - e.vy) * gain
    sp = math.hypot(e.vx, e.vy)
    if sp > maxspeed:
        e.vx *= maxspeed / sp
        e.vy *= maxspeed / sp


def can_see(s: Ent, h: Ent, walls: List[Wall], boxes: List[Ent]) -> bool:
    dx, dy = h.x - s.x, h.y - s.y
    dist = math.hypot(dx, dy)
    if dist > VISION_RANGE:
        return False
    ang = math.atan2(dy, dx)
    da = (ang - s.heading + math.pi) % (2 * math.pi) - math.pi
    if abs(da) > VISION_HALF_CONE:
        return False
    for w in walls:
        if w.active and seg_aabb_blocked(s.x, s.y, h.x, h.y, w.x0, w.y0, w.x1, w.y1):
            return False
    for b in boxes:
        if seg_circle_blocked(s.x, s.y, h.x, h.y, b.x, b.y, b.r * 0.9):
            return False
    return True


def face_towards(e: Ent, tx: float, ty: float, rate: float = 0.35) -> None:
    want = math.atan2(ty - e.y, tx - e.x)
    da = (want - e.heading + math.pi) % (2 * math.pi) - math.pi
    e.heading += da * rate


# --------------------------------------------------------------------------- #
# Scenario configs
# --------------------------------------------------------------------------- #
# Each scenario lists agents, boxes, optional ramp/decoy, optional interior walls,
# whether the west door exists, prep/steps, and which "flavour" of behaviour.
SCENARIOS: Dict[str, Dict] = {
    "showcase": {
        "title": "Synthetic Showcase", "door": True, "prep": 60, "steps": 240,
        "hiders": [(-2.3, -1.2), (-3.4, 0.6)],
        "seekers": [(4.7, -3.4), (4.7, 3.4)],
        "boxes": [("box_light", -4.2, -3.2), ("box_light", -1.3, -4.2),
                  ("box_light", 2.4, -3.0), ("box_heavy", 0.6, -0.2)],
        "ramp": (3.9, 2.7), "decoy": (-3.0, -3.6),
        "interior": [(1.8, 1.0, 3.4, False)],     # (cx,cy,length,horizontal)
        "fort": (-4.4, 3.4), "safe": (-4.6, 3.6),
    },
    "chase": {
        "title": "Running & Chasing", "door": True, "prep": 35, "steps": 210,
        "hiders": [(-2.0, -1.0), (-3.2, 1.4)],
        "seekers": [(4.6, -3.4), (4.6, 3.4)],
        "boxes": [], "ramp": None, "decoy": None, "interior": [], "fort": None,
    },
    "fort": {
        "title": "Fort Building", "door": True, "prep": 80, "steps": 230,
        "hiders": [(-2.2, -1.0), (-3.4, 0.8)],
        "seekers": [(4.7, -3.5), (4.7, 3.5)],
        "boxes": [("box_light", -3.8, -3.2), ("box_light", -1.4, -3.8),
                  ("box_light", -4.4, -1.0), ("box_heavy", 0.4, 0.6)],
        "ramp": None, "decoy": None, "interior": [], "fort": (-4.3, 3.4),
        "safe": (-4.6, 3.6),
    },
    "doors": {
        "title": "Door Blocking", "door": True, "prep": 70, "steps": 220,
        "hiders": [(-2.4, 0.2), (-3.4, -1.0)],
        "seekers": [(4.7, -1.0), (4.7, 1.0)],
        "boxes": [("box_heavy", -3.6, 0.0), ("box_light", -2.6, 1.6)],
        "ramp": None, "decoy": None, "interior": [], "fort": None,
        "block_door": True, "safe": (-5.0, 2.6),
    },
}
MANIFEST_ORDER = ["showcase", "chase", "fort", "doors"]
DESCRIPTIONS = {
    "showcase": "A full game: hiders build cover, seekers hunt, a clear winner.",
    "chase": "Open-arena pursuit — seekers see, chase and tag a hider.",
    "fort": "Hiders push boxes into a barricade, then try to survive.",
    "doors": "Hiders jam the doorway with a heavy box to hold the seekers out.",
}


# --------------------------------------------------------------------------- #
# The simulation
# --------------------------------------------------------------------------- #
def simulate(cfg: Dict, seed: int) -> Tuple[List[Dict], List[Dict], Dict, int]:
    """Run one scenario. Returns (entity_meta, frames, outcome, E)."""
    rng = random.Random(seed)
    prep, steps = cfg["prep"], cfg["steps"]
    fort = cfg.get("fort")
    HS = cfg.get("hspeed", 2.75)     # hider top speed (per scenario)
    SS = cfg.get("sspeed", 2.9)      # seeker top speed (per scenario)

    ents: List[Ent] = []

    def add(t, team, x, y):
        e = Ent(len(ents), t, team, x + rng.uniform(-0.05, 0.05),
                y + rng.uniform(-0.05, 0.05))
        ents.append(e)
        return e

    hiders = [add("hider", 0, x, y) for (x, y) in cfg["hiders"]]
    seekers = [add("seeker", 1, x, y) for (x, y) in cfg["seekers"]]
    boxes = [add(t, -1, x, y) for (t, x, y) in cfg["boxes"]]
    light = [b for b in boxes if b.type == "box_light"]
    heavy = [b for b in boxes if b.type == "box_heavy"]
    ramp = add("ramp", -1, *cfg["ramp"]) if cfg.get("ramp") else None
    decoy = add("decoy", -1, *cfg["decoy"]) if cfg.get("decoy") else None
    for s in seekers:
        s.heading = math.pi                      # face into the arena

    # Walls (collision) + their viewer entity ids come after all the above.
    wall_start = len(ents)
    walls = border_walls(wall_start, door=cfg["door"])
    for (cx, cy, length, horiz) in cfg.get("interior", []):
        walls.append(Wall(wall_start + len(walls), cx, cy, length, horiz))
    # A viewer "entity" per wall (so they render); door is the is_door one.
    wall_ents: List[Ent] = []
    for w in walls:
        we = Ent(len(ents), "door" if w.is_door else "wall", -1, w.cx, w.cy)
        we.heading = w.heading
        we.locked = 1
        we.r = max(w.size, WALL_T)
        ents.append(we)
        wall_ents.append(we)
        w._ent = we  # type: ignore  (attach for size export)

    movers = hiders + seekers + boxes
    door_wall = next((w for w in walls if w.is_door), None)

    # ---- choreography targets ------------------------------------------- #
    # Hiders each "claim" a light box to push to the fort during prep.
    claims = {}
    for i, h in enumerate(hiders):
        if light and fort:
            claims[h.id] = light[i % len(light)]

    outcome: Optional[Dict] = None
    frames: List[Dict] = []
    decided_at = steps
    last_seen: Dict[int, Optional[Tuple[float, float]]] = {}
    lost: Dict[int, int] = {}

    for t in range(steps):
        is_main = t >= prep
        # ---------- decide goals + steer ----------
        for h in hiders:
            if outcome is not None:
                continue
            if not is_main:
                # prep: push your claimed box toward the fort (emergent pushing),
                # else move toward the fort area.
                box = claims.get(h.id)
                if cfg.get("block_door") and door_wall is not None and heavy:
                    box = heavy[0]
                    goal = (door_wall.cx + 1.0, door_wall.cy)
                    # press the heavy box from the arena side toward the door
                    bx, by = box.x, box.y
                    nx, ny = vnorm(bx - goal[0], by - goal[1])
                    steer_to(h, bx + nx * (box.r + h.r), by + ny * (box.r + h.r),
                             AGENT_SPEED)
                elif box is not None and fort is not None:
                    nx, ny = vnorm(box.x - fort[0], box.y - fort[1])
                    behind = (box.x + nx * (box.r + h.r), box.y + ny * (box.r + h.r))
                    if vlen(h.x - behind[0], h.y - behind[1]) > 0.25 and \
                       vlen(box.x - fort[0], box.y - fort[1]) > 0.9:
                        steer_to(h, behind[0], behind[1], HS)
                    else:
                        steer_to(h, fort[0], fort[1], HS)
                else:
                    steer_to(h, (fort or (-3.5, 2.5))[0], (fort or (-3.5, 2.5))[1],
                             HS *0.7)
            else:
                # main: head for cover (the safe pocket behind the fort / door
                # block) unless a seeker is close, then flee directly. Reaching
                # cover breaks line-of-sight, so the fort/door actually protects
                # the hiders -> the game has real stakes and a non-obvious winner.
                ns = min(seekers, key=lambda s: vlen(s.x - h.x, s.y - h.y))
                dseek = vlen(ns.x - h.x, ns.y - h.y)
                safe = cfg.get("safe")
                if safe is not None and dseek > 3.0:
                    steer_to(h, safe[0], safe[1], HS)
                else:
                    fx, fy = vnorm(h.x - ns.x, h.y - ns.y)
                    tx = clamp(h.x + fx * 4.0, -HALF + 0.8, HALF - 0.8)
                    ty = clamp(h.y + fy * 4.0, -HALF + 0.8, HALF - 0.8)
                    steer_to(h, tx, ty, HS)
            if vlen(h.vx, h.vy) > 0.05:
                face_towards(h, h.x + h.vx, h.y + h.vy, 0.5)

        for s in seekers:
            if outcome is not None:
                continue
            if not is_main:
                s.vx = s.vy = 0.0                # frozen during prep
                s.locked = 1
                continue
            s.locked = 0
            seen = [h for h in hiders if can_see(s, h, walls, boxes)]
            if seen:
                # chase what you SEE.
                tgt = min(seen, key=lambda h: vlen(h.x - s.x, h.y - s.y))
                last_seen[s.id] = (tgt.x, tgt.y)
                lost[s.id] = 0
                steer_to(s, tgt.x, tgt.y, SS)
                face_towards(s, tgt.x, tgt.y, 0.5)
            else:
                # NOT seeing anyone: go to where the hider was LAST seen (never the
                # true position -- that is the whole point of hiding); then patrol
                # and scan. So breaking line-of-sight actually loses the seeker.
                lost[s.id] = lost.get(s.id, 0) + 1
                ls = last_seen.get(s.id)
                if ls is not None and lost[s.id] < 40 and vlen(s.x - ls[0], s.y - ls[1]) > 0.7:
                    steer_to(s, ls[0], ls[1], SS *0.85)
                    face_towards(s, ls[0], ls[1], 0.4)
                else:
                    last_seen[s.id] = None
                    ang = t * 0.035 + s.id * 2.3
                    steer_to(s, 3.2 * math.cos(ang), 3.2 * math.sin(ang),
                             SS *0.6, gain=0.15)
                    s.heading += 0.05            # sweep the cone while searching

        # ---------- integrate (agents only; boxes move via separation) ----------
        for e in movers:
            if e.type in ("box_light", "box_heavy"):
                e.vx = e.vy = 0.0
                continue
            e.x += e.vx * DT
            e.y += e.vy * DT
            moving = vlen(e.vx, e.vy)
            if e.type in ("hider", "seeker"):
                if moving > 1.5:
                    e.stam = clamp(e.stam - 0.012, 0.05, 1.0)
                else:
                    e.stam = clamp(e.stam + 0.008, 0.05, 1.0)

        # ---------- collisions ----------
        resolve_collisions(movers, walls)

        # lock light boxes once parked at the fort (so they form a stable wall)
        if fort is not None:
            for b in light:
                if not b.locked and vlen(b.x - fort[0], b.y - fort[1]) < 1.4 and is_main:
                    b.locked = 1

        # ---------- perception / "spotted" ----------
        seen_any = False
        for h in hiders:
            h.sn = 0
        for s in seekers:
            for h in hiders:
                if is_main and outcome is None and can_see(s, h, walls, boxes):
                    h.sn = 1
                    seen_any = True
                    if vlen(s.x - h.x, s.y - h.y) < TAG_DIST:
                        outcome = {"winner": "seekers",
                                   "reason": "A seeker tagged a hider.",
                                   "step": t}
                        decided_at = t

        # ---------- decoy ----------
        if decoy is not None:
            since = t - (prep - 18)
            if since >= 0:
                decoy.dc = 1
                decoy.noise = clamp(0.55 + 0.2 * math.sin(since * 0.6), 0.0, 1.0)

        # ---------- scores ----------
        if is_main and outcome is None:
            sh = (t - prep) * 1.0
            ss = -(t - prep) * 1.0
        elif outcome is not None:
            sh, ss = -50.0, 50.0
        else:
            sh = ss = 0.0

        frames.append(_record(t, prep, ents, hiders, seekers, boxes, claims,
                              ramp, decoy, wall_ents, walls, sh, ss, seen_any))

    if outcome is None:
        outcome = {"winner": "hiders",
                   "reason": "The hiders stayed unseen until time ran out.",
                   "step": steps - 1}

    # static entity table
    meta = []
    for e in ents:
        size = e.r
        if e.type in ("wall", "door"):
            # find the wall to export its half-length as size
            wl = next((w for w in walls if getattr(w, "_ent", None) is e), None)
            size = wl.size if wl else e.r
        meta.append(schema.make_entity_meta(
            id=e.id, type=e.type, team=e.team, size=round(size, 3),
            mass=MASS.get(e.type, 1.0e6), is_decoy=(e.type == "decoy")))
    return meta, frames, outcome, len(ents)


def _record(t, prep, ents, hiders, seekers, boxes, claims, ramp, decoy,
            wall_ents, walls, sh, ss, seen_any) -> Dict:
    phase = "prep" if t < prep else "main"
    held = {}
    for h in hiders:
        b = claims.get(h.id)
        if b is not None and t < prep and vlen(h.x - b.x, h.y - b.y) < (h.r + b.r + 0.25):
            held[b.id] = h.id
    ent = []
    for e in ents:
        if e.type in ("wall", "door"):
            wl = next((w for w in walls if getattr(w, "_ent", None) is e), None)
            active = 1 if (wl is None or wl.active) else 0
            ent.append(schema.make_frame_ent(
                id=e.id, x=e.x, y=e.y, z=0.0, h=e.heading, a=active, lk=1,
                hd=0, hb=-1, no=0.0, dc=0, gr=1, st=-1.0, sn=0))
            continue
        is_agent = e.type in ("hider", "seeker")
        ent.append(schema.make_frame_ent(
            id=e.id, x=e.x, y=e.y, z=0.0,
            h=(e.heading if is_agent else 0.0),
            a=1, lk=e.locked,
            hd=1 if e.id in held else 0, hb=held.get(e.id, -1),
            no=e.noise, dc=e.dc, gr=1,
            st=(round(e.stam, 3) if is_agent else -1.0), sn=e.sn))
    return schema.make_frame(t=t, phase=phase, sh=sh, ss=ss,
                             seen_any=bool(seen_any), fog=[], ent=ent)


# --------------------------------------------------------------------------- #
# Programmatic correctness gate
# --------------------------------------------------------------------------- #
def validate_simulation(doc: Dict) -> List[str]:
    """Hard checks: no overlaps, nothing in a wall, in bounds, smooth, decided.
    Returns a list of problems (empty == clean)."""
    p: List[str] = []
    p += schema.validate_trajectory(doc)
    meta = {m["id"]: m for m in doc["entities"]}
    frames = doc["frames"]
    movable_types = ("hider", "seeker", "box_light", "box_heavy")
    walls = [m for m in doc["entities"] if m["type"] in ("wall", "door")]

    def wall_aabb(m, fe):
        horiz = abs(fe["h"]) < 0.3
        L = m["size"] * 2
        hw = (L / 2) if horiz else (WALL_T / 2)
        hh = (WALL_T / 2) if horiz else (L / 2)
        return fe["x"] - hw, fe["y"] - hh, fe["x"] + hw, fe["y"] + hh

    prev = {}
    for fi, fr in enumerate(frames):
        ent = {e["id"]: e for e in fr["ent"]}
        mov = [e for e in fr["ent"] if meta[e["id"]]["type"] in movable_types and e["a"]]
        # bounds
        for e in mov:
            if abs(e["x"]) > HALF + 0.05 or abs(e["y"]) > HALF + 0.05:
                p.append(f"f{fi} ent{e['id']} out of bounds ({e['x']:.2f},{e['y']:.2f})")
        # pairwise overlap
        for i in range(len(mov)):
            for j in range(i + 1, len(mov)):
                a, b = mov[i], mov[j]
                rr = meta[a["id"]]["size"] + meta[b["id"]]["size"]
                d = math.hypot(b["x"] - a["x"], b["y"] - a["y"])
                if d < rr - 0.12:
                    p.append(f"f{fi} overlap ent{a['id']}&{b['id']} d={d:.2f}<{rr:.2f}")
        # wall penetration
        for e in mov:
            r = meta[e["id"]]["size"]
            for w in walls:
                we = ent.get(w["id"])
                if not we or not we["a"]:
                    continue
                x0, y0, x1, y1 = wall_aabb(w, we)
                cx = clamp(e["x"], x0, x1)
                cy = clamp(e["y"], y0, y1)
                if math.hypot(e["x"] - cx, e["y"] - cy) < r - 0.12:
                    p.append(f"f{fi} ent{e['id']} inside wall {w['id']}")
                    break
        # smoothness (no teleport)
        for e in mov:
            if e["id"] in prev:
                step = math.hypot(e["x"] - prev[e["id"]][0], e["y"] - prev[e["id"]][1])
                if step > 0.9:
                    p.append(f"f{fi} ent{e['id']} jumped {step:.2f} in one step")
        prev = {e["id"]: (e["x"], e["y"]) for e in mov}
        if len(p) > 40:
            p.append("... (truncated)")
            return p
    if "outcome" not in doc or doc["outcome"].get("winner") not in ("hiders", "seekers"):
        p.append("no clear outcome recorded")
    return p


# --------------------------------------------------------------------------- #
# Build all scenarios + manifest
# --------------------------------------------------------------------------- #
def build_doc(name: str, seed: int) -> Dict:
    cfg = SCENARIOS[name]
    meta, frames, outcome, E = simulate(cfg, seed)
    doc = schema.make_trajectory(
        meta={
            "title": cfg["title"], "seed": int(seed), "arena_size": ARENA,
            "dt": DT, "max_steps": cfg["steps"], "prep_steps": cfg["prep"],
            "entity_types": list(schema.ENTITY_TYPES), "max_agents": 6,
            "max_entities": E,
        },
        entities=meta, frames=frames)
    doc["outcome"] = outcome
    return doc


DEFAULT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "web", "trajectories")


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Generate simulated Hide & Seek demo "
                                             "trajectories (pure stdlib).")
    ap.add_argument("--out-dir", default=DEFAULT_DIR)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--scenario", default=None, help="build only this scenario id")
    args = ap.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    names = [args.scenario] if args.scenario else list(MANIFEST_ORDER)
    rc = 0
    manifest_scn = []
    for i, name in enumerate(names):
        doc = build_doc(name, args.seed + i)
        problems = validate_simulation(doc)
        status = "OK" if not problems else f"{len(problems)} PROBLEM(S)"
        path = os.path.join(args.out_dir, f"{name}.json")
        if problems:
            rc = 1
            print(f"  {name:9s} FAILED validation: {status}")
            for pr in problems[:8]:
                print(f"      - {pr}")
        else:
            schema.save_trajectory(doc, path)
            print(f"  {name:9s} frames={len(doc['frames'])} E={doc['meta']['max_entities']} "
                  f"winner={doc['outcome']['winner']:7s} -> {status}")
        manifest_scn.append({"id": name, "title": SCENARIOS[name]["title"],
                             "description": DESCRIPTIONS[name], "file": f"{name}.json"})

    if not args.scenario and rc == 0:
        # demo_trajectory.json == showcase (back-compat) + manifest
        show = os.path.join(args.out_dir, "showcase.json")
        if os.path.exists(show):
            with open(show, "r", encoding="utf-8") as f:
                data = f.read()
            with open(os.path.join(args.out_dir, "demo_trajectory.json"), "w",
                      encoding="utf-8") as f:
                f.write(data)
        manifest = {"format": "hns2-manifest", "version": 1, "default": "showcase",
                    "scenarios": manifest_scn}
        with open(os.path.join(args.out_dir, "manifest.json"), "w",
                  encoding="utf-8") as f:
            json.dump(manifest, f, separators=(",", ":"))
        print(f"  manifest  {len(manifest_scn)} scenarios + demo_trajectory.json")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
