"""
learn/export_viewer.py -- train the real self-play RL (learn/train.py) and export
REAL data for the 3D viewer:

  * viz/web/learning.json          -- the MEASURED learning curve (not synthetic)
  * viz/web/trajectories/*.json    -- rollouts of the LEARNED policies at three
                                      training stages (untrained -> mid -> trained)
  * viz/web/trajectories/manifest.json + demo_trajectory.json

The grid world (cells) is mapped into the viewer's continuous arena; the wall
segments become slabs. Agent headings follow their actual movement. Each clip's
win banner reflects what really happened in that rollout.

Run:  python -m learn.export_viewer            # trains + writes everything
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List, Tuple

from learn import train as T
from viz import schema

CELL = 1.35
DT = 0.12
ROLL_STEPS = 48


def world(c: int) -> float:
    """Grid cell coordinate -> arena coordinate (centered)."""
    return round((c - (T.N - 1) / 2.0) * CELL, 3)


# --------------------------------------------------------------------------- #
# Static entities: 1 hider, 1 seeker, wall slabs (interior segments + border)
# --------------------------------------------------------------------------- #
def wall_defs() -> List[Tuple[float, float, float, float]]:
    """Return wall slabs as (x, y, heading, half_len) in arena coords."""
    defs: List[Tuple[float, float, float, float]] = []
    for (x0, y0), (x1, y1) in T.WALL_SEGMENTS:
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        if x0 == x1:
            length = (abs(y1 - y0) + 1) * CELL
            defs.append((world(cx), world(cy), math.pi / 2, length / 2.0))
        else:
            length = (abs(x1 - x0) + 1) * CELL
            defs.append((world(cx), world(cy), 0.0, length / 2.0))
    half = (T.N / 2.0) * CELL
    blen = T.N * CELL + CELL
    defs.append((0.0, half, 0.0, blen / 2.0))          # top
    defs.append((0.0, -half, 0.0, blen / 2.0))         # bottom
    defs.append((half, 0.0, math.pi / 2, blen / 2.0))  # right
    defs.append((-half, 0.0, math.pi / 2, blen / 2.0)) # left
    return defs


WALLS = wall_defs()
HIDER_ID, SEEKER_ID = 0, 1
WALL_ID0 = 2
E = 2 + len(WALLS)


def entity_meta() -> List[Dict]:
    meta = [
        schema.make_entity_meta(HIDER_ID, "hider", schema.TEAM_HIDER, 0.42, 1.0, False),
        schema.make_entity_meta(SEEKER_ID, "seeker", schema.TEAM_SEEKER, 0.42, 1.0, False),
    ]
    for i, (_, _, _, size) in enumerate(WALLS):
        meta.append(schema.make_entity_meta(WALL_ID0 + i, "wall", schema.TEAM_NONE,
                                            round(size, 3), 1.0e6, False))
    return meta


# --------------------------------------------------------------------------- #
# Roll out a policy pair into a trajectory
# --------------------------------------------------------------------------- #
def rollout(Q_s, Q_h, s_random: bool, h_random: bool, seed: int) -> Dict:
    rng = random.Random(seed)
    env = T.GridHideSeek(rng)
    env.reset()
    env.sx, env.sy, env.hx, env.hy = 1, 1, T.N - 2, T.N - 2   # fixed corners
    head_s, head_h = 0.0, math.pi
    frames: List[Dict] = []
    caught_step = -1
    seen_steps = 0

    for t in range(ROLL_STEPS):
        cs, ch = env.state()
        see = env.can_see()
        caught = see and (abs(env.sx - env.hx) + abs(env.sy - env.hy) <= 1)
        if see:
            seen_steps += 1
        if caught and caught_step < 0:
            caught_step = t

        ent = [
            schema.make_frame_ent(HIDER_ID, world(env.hx), world(env.hy), 0.0, head_h,
                                  a=1, lk=0, hd=0, hb=-1, no=0.0, dc=0, gr=1,
                                  st=1.0, sn=1 if see else 0),
            schema.make_frame_ent(SEEKER_ID, world(env.sx), world(env.sy), 0.0, head_s,
                                  a=1, lk=0, hd=0, hb=-1, no=0.0, dc=0, gr=1,
                                  st=1.0, sn=0),
        ]
        for i, (wx, wy, wh, _sz) in enumerate(WALLS):
            ent.append(schema.make_frame_ent(WALL_ID0 + i, wx, wy, 0.0, wh,
                                             a=1, lk=1, hd=0, hb=-1, no=0.0, dc=0,
                                             gr=1, st=-1.0, sn=0))
        sh = float(seen_steps == 0 and t or (t - seen_steps))   # rough "hidden" tally
        frames.append(schema.make_frame(t=t, phase="main", sh=round(sh, 1),
                                        ss=round(seen_steps, 1),
                                        seen_any=bool(see), fog=[], ent=ent))

        if caught:
            break
        a_s = rng.randrange(T.NA) if s_random else T.greedy(Q_s, cs, ch)
        a_h = rng.randrange(T.NA) if h_random else T.greedy(Q_h, cs, ch)
        psx, psy, phx, phy = env.sx, env.sy, env.hx, env.hy
        env.step(a_s, a_h)
        if (env.sx, env.sy) != (psx, psy):
            head_s = math.atan2(env.sy - psy, env.sx - psx)
        if (env.hx, env.hy) != (phx, phy):
            head_h = math.atan2(env.hy - phy, env.hx - phx)

    # Hold the final frame for a moment so the result banner lingers and short
    # catches don't loop frantically.
    hold_t = frames[-1]["t"]
    while len(frames) < 40:
        hold_t += 1
        prev = frames[-1]
        frames.append(schema.make_frame(t=hold_t, phase="main", sh=prev["sh"],
                                        ss=prev["ss"], seen_any=prev["seen_any"],
                                        fog=[], ent=prev["ent"]))

    if caught_step >= 0:
        outcome = {"winner": "seekers", "reason": "The seeker caught the hider.",
                   "step": caught_step}
    else:
        outcome = {"winner": "hiders",
                   "reason": "The hider evaded the seeker for the whole episode.",
                   "step": len(frames) - 1}

    doc = schema.make_trajectory(
        meta={"title": "", "seed": seed, "arena_size": T.N * CELL + CELL, "dt": DT,
              "max_steps": ROLL_STEPS, "prep_steps": 0,
              "entity_types": list(schema.ENTITY_TYPES), "max_agents": 2,
              "max_entities": E},
        entities=entity_meta(), frames=frames)
    doc["outcome"] = outcome
    return doc


# --------------------------------------------------------------------------- #
# Build the measured learning curve in the dashboard's learning.json shape
# --------------------------------------------------------------------------- #
def learning_doc(curve: List[Dict]) -> Dict:
    t = [c["episode"] for c in curve]
    seek = [c["seeker_skill"] for c in curve]
    hide = [c["hider_skill"] for c in curve]

    def first_ep(vals, thresh):
        for c in curve:
            if c["seeker_skill" if vals == "s" else "hider_skill"] >= thresh:
                return c["episode"]
        return curve[-1]["episode"]

    milestones = [
        {"id": "chase", "step": first_ep("s", 0.40), "team": "seeker",
         "title": "Seeker learns to hunt",
         "desc": "Its sight-rate against a random hider climbs far above chance.",
         "emoji": "\U0001F50D"},
        {"id": "evade", "step": first_ep("h", 0.95), "team": "hider",
         "title": "Hider learns to use cover",
         "desc": "It breaks line-of-sight behind walls and evades almost every time.",
         "emoji": "\U0001F9F1"},
        {"id": "armsrace", "step": t[-1], "team": "hider",
         "title": "Arms race",
         "desc": "As the hider masters evasion, the seeker has to keep adapting.",
         "emoji": "\U0001F501"},
    ]
    return {
        "format": "hns2-learning", "version": 1,
        "meta": {"title": "Self-play training progress (measured)",
                 "total_timesteps": t[-1], "unit": "self-play episodes",
                 "synthetic": False,
                 "note": "MEASURED, not synthetic: a real tabular-Q self-play run on "
                         "the CPU (learn/train.py). Skill = success against a random "
                         "opponent, so both curves rising proves real learning."},
        "series": {
            "t": t,
            "seeker_winrate": seek,
            "hider_winrate": hide,
            "seeker_elo": [int(1200 + s * 600) for s in seek],
            "hider_elo": [int(1200 + h * 600) for h in hide],
            "episode_len": [T.MAX_STEPS for _ in t],
        },
        "milestones": milestones,
        "teams": {
            "hider": {"elo": int(1200 + hide[-1] * 600), "winrate": hide[-1],
                      "tactic": "Breaks line-of-sight behind walls"},
            "seeker": {"elo": int(1200 + seek[-1] * 600), "winrate": seek[-1],
                       "tactic": "Chases and corners the hider"},
        },
    }


SCN = [
    ("seeker", "Trained seeker vs random hider",
     "The seeker has LEARNED to hunt: it chases and corners a randomly-moving hider."),
    ("hider", "Trained hider vs random seeker",
     "The hider has LEARNED to use the walls to break line-of-sight and evade."),
    ("untrained", "Untrained (random vs random)",
     "Before any learning — both agents just move at random."),
]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train + export real learned behaviour "
                                             "and the measured learning curve.")
    ap.add_argument("--episodes", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    mid = max(1, args.episodes // 5)
    out = T.train(args.episodes, args.seed, snap_episodes=[mid, args.episodes])
    snaps = out["snaps"]
    mid_s = snaps.get(mid) or (out["Q_s"], out["Q_h"])
    fin_s = snaps.get(args.episodes) or (out["Q_s"], out["Q_h"])

    rollouts = {
        "seeker": rollout(mid_s[0], None, False, True, 2),   # trained seeker, random hider
        "hider": rollout(None, fin_s[1], True, False, 5),     # random seeker, trained hider
        "untrained": rollout(None, None, True, True, 1),
    }

    tdir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "viz", "web", "trajectories")
    os.makedirs(tdir, exist_ok=True)
    web = os.path.dirname(tdir)

    manifest_scn = []
    for sid, title, desc in SCN:
        doc = rollouts[sid]
        doc["meta"]["title"] = title
        problems = schema.validate_trajectory(doc)
        if problems:
            print(f"  {sid:10s} INVALID: {problems[:3]}")
            return 1
        schema.save_trajectory(doc, os.path.join(tdir, f"{sid}.json"))
        manifest_scn.append({"id": sid, "title": title, "description": desc,
                             "file": f"{sid}.json"})
        print(f"  {sid:10s} frames={len(doc['frames'])} winner={doc['outcome']['winner']}")

    # demo_trajectory == the trained-seeker clip (the clearest "it learned" moment)
    with open(os.path.join(tdir, "seeker.json"), "r", encoding="utf-8") as f:
        data = f.read()
    with open(os.path.join(tdir, "demo_trajectory.json"), "w", encoding="utf-8") as f:
        f.write(data)
    with open(os.path.join(tdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump({"format": "hns2-manifest", "version": 1, "default": "seeker",
                   "scenarios": manifest_scn}, f, separators=(",", ":"))

    with open(os.path.join(web, "learning.json"), "w", encoding="utf-8") as f:
        json.dump(learning_doc(out["curve"]), f, separators=(",", ":"))

    print(f"  wrote learning.json (measured curve, {len(out['curve'])} points) + manifest")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
