"""
learn/train.py -- a REAL self-play reinforcement learner for hide-and-seek that
actually trains on the CPU (no GPU, no JAX). Two independent **tabular
Q-learning** agents (a seeker and a hider) learn by playing each other on a small
grid with walls and line-of-sight perception. Their behaviour is LEARNED, not
scripted.

This is the small, honest, runs-anywhere counterpart to the GPU JAX/MAPPO stack
in the rest of the repo: same idea (competitive self-play autocurriculum), shrunk
to a tabular grid world so it genuinely converges in seconds on a laptop and you
can *prove* it learns.

Run:  python -m learn.train            # trains + prints the learning curve
The curve (seeker "see rate" vs a RANDOM hider, and hider "evasion" vs a RANDOM
seeker) is the proof: both climb well above their random baselines.
"""
from __future__ import annotations

import argparse
import random
from typing import Dict, List, Tuple

import numpy as np

# --------------------------------------------------------------------------- #
# Environment: a grid hide-and-seek MDP
# --------------------------------------------------------------------------- #
N = 9                                   # grid is N x N cells
MAX_STEPS = 40
SEE_RADIUS = 3                          # Chebyshev sight radius
ACTIONS = [(0, 0), (0, 1), (0, -1), (1, 0), (-1, 0)]   # stay, N, S, E, W
NA = len(ACTIONS)

# Interior walls as straight cell-runs (segments), so the SAME layout can be drawn
# as a few slabs in the 3D viewer. A bit of cover lets the hider break line-of-sight.
WALL_SEGMENTS = [
    ((4, 1), (4, 3)),    # vertical
    ((2, 5), (4, 5)),    # horizontal
    ((6, 4), (6, 6)),    # vertical
    ((1, 7), (2, 7)),    # horizontal
]


def _seg_cells(seg) -> List[Tuple[int, int]]:
    (x0, y0), (x1, y1) = seg
    if x0 == x1:
        return [(x0, y) for y in range(min(y0, y1), max(y0, y1) + 1)]
    return [(x, y0) for x in range(min(x0, x1), max(x0, x1) + 1)]


WALLS = frozenset(c for seg in WALL_SEGMENTS for c in _seg_cells(seg))


def in_bounds(x: int, y: int) -> bool:
    return 0 <= x < N and 0 <= y < N


def free(x: int, y: int) -> bool:
    return in_bounds(x, y) and (x, y) not in WALLS


FREE_CELLS = [(x, y) for x in range(N) for y in range(N) if free(x, y)]


def cell_id(x: int, y: int) -> int:
    return y * N + x


def los_clear(x0: int, y0: int, x1: int, y1: int) -> bool:
    """Bresenham line-of-sight: blocked if a wall lies strictly between."""
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    x, y = x0, y0
    while (x, y) != (x1, y1):
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy
        if (x, y) == (x1, y1):
            break
        if (x, y) in WALLS:
            return False
    return True


class GridHideSeek:
    def __init__(self, rng: random.Random):
        self.rng = rng
        self.sx = self.sy = self.hx = self.hy = 0
        self.t = 0

    def reset(self) -> Tuple[int, int]:
        while True:
            self.sx, self.sy = self.rng.choice(FREE_CELLS)
            self.hx, self.hy = self.rng.choice(FREE_CELLS)
            if max(abs(self.sx - self.hx), abs(self.sy - self.hy)) >= 3:
                break
        self.t = 0
        return self.state()

    def state(self) -> Tuple[int, int]:
        return cell_id(self.sx, self.sy), cell_id(self.hx, self.hy)

    def can_see(self) -> bool:
        if max(abs(self.sx - self.hx), abs(self.sy - self.hy)) > SEE_RADIUS:
            return False
        return los_clear(self.sx, self.sy, self.hx, self.hy)

    def step(self, a_s: int, a_h: int):
        dxs, dys = ACTIONS[a_s]
        if free(self.sx + dxs, self.sy + dys):
            self.sx += dxs
            self.sy += dys
        dxh, dyh = ACTIONS[a_h]
        if free(self.hx + dxh, self.hy + dyh):
            self.hx += dxh
            self.hy += dyh
        self.t += 1
        see = self.can_see()
        caught = see and (abs(self.sx - self.hx) + abs(self.sy - self.hy) <= 1)
        # zero-sum-ish: seeker wants to SEE / catch; hider wants to stay unseen.
        r_s = 3.0 if caught else (1.0 if see else -0.05)
        done = caught or self.t >= MAX_STEPS
        return self.state(), r_s, -r_s, see, caught, done


# --------------------------------------------------------------------------- #
# Tabular Q-learning self-play
# --------------------------------------------------------------------------- #
NS = N * N


def greedy(Q: np.ndarray, cs: int, ch: int) -> int:
    return int(np.argmax(Q[cs, ch]))


def eps_action(Q: np.ndarray, cs: int, ch: int, eps: float, rng: random.Random) -> int:
    if rng.random() < eps:
        return rng.randrange(NA)
    return greedy(Q, cs, ch)


def evaluate(Q_s: np.ndarray, Q_h: np.ndarray, rng: random.Random,
             episodes: int = 300) -> Tuple[float, float]:
    """Return (seeker see-rate vs RANDOM hider, hider evasion vs RANDOM seeker).
    Clean skill signals against a fixed random opponent -> a true learning curve."""
    env = GridHideSeek(rng)
    # seeker (greedy) vs random hider
    seen = 0
    tot = 0
    for _ in range(episodes):
        cs, ch = env.reset()
        for _ in range(MAX_STEPS):
            a_s = greedy(Q_s, cs, ch)
            a_h = rng.randrange(NA)
            (cs, ch), _, _, see, caught, done = env.step(a_s, a_h)
            seen += 1 if see else 0
            tot += 1
            if done:
                break
    seeker_skill = seen / max(1, tot)
    # hider (greedy) vs random seeker
    seen2 = 0
    tot2 = 0
    for _ in range(episodes):
        cs, ch = env.reset()
        for _ in range(MAX_STEPS):
            a_s = rng.randrange(NA)
            a_h = greedy(Q_h, cs, ch)
            (cs, ch), _, _, see, caught, done = env.step(a_s, a_h)
            seen2 += 1 if see else 0
            tot2 += 1
            if done:
                break
    hider_evasion = 1.0 - seen2 / max(1, tot2)
    return seeker_skill, hider_evasion


def train(episodes: int, seed: int, log_every: int = 2500, snap_episodes=()) -> Dict:
    rng = random.Random(seed)
    Q_s = np.zeros((NS, NS, NA), dtype=np.float32)
    Q_h = np.zeros((NS, NS, NA), dtype=np.float32)
    env = GridHideSeek(rng)
    alpha, gamma = 0.25, 0.95
    curve: List[Dict] = []
    snaps: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    snap_set = set(snap_episodes)

    print(f"training {episodes} self-play episodes on a {N}x{N} grid "
          f"({len(WALLS)} walls)...")
    # baseline (untrained == random policies)
    ss0, he0 = evaluate(Q_s, Q_h, random.Random(123))
    print(f"  ep {0:7d}   seeker see-rate={ss0:.3f}   hider evasion={he0:.3f}   (random baseline)")
    curve.append({"episode": 0, "seeker_skill": round(ss0, 4), "hider_skill": round(he0, 4)})

    for ep in range(1, episodes + 1):
        eps = max(0.05, 0.35 * (1.0 - ep / episodes))
        cs, ch = env.reset()
        for _ in range(MAX_STEPS):
            a_s = eps_action(Q_s, cs, ch, eps, rng)
            a_h = eps_action(Q_h, cs, ch, eps, rng)
            (cs2, ch2), r_s, r_h, see, caught, done = env.step(a_s, a_h)
            ns_s = 0.0 if done else float(np.max(Q_s[cs2, ch2]))
            ns_h = 0.0 if done else float(np.max(Q_h[cs2, ch2]))
            Q_s[cs, ch, a_s] += alpha * (r_s + gamma * ns_s - Q_s[cs, ch, a_s])
            Q_h[cs, ch, a_h] += alpha * (r_h + gamma * ns_h - Q_h[cs, ch, a_h])
            cs, ch = cs2, ch2
            if done:
                break
        if ep % log_every == 0:
            ss, he = evaluate(Q_s, Q_h, random.Random(123))
            print(f"  ep {ep:7d}   seeker see-rate={ss:.3f}   hider evasion={he:.3f}   (eps={eps:.2f})")
            curve.append({"episode": ep, "seeker_skill": round(ss, 4),
                          "hider_skill": round(he, 4)})
        if ep in snap_set:
            snaps[ep] = (Q_s.copy(), Q_h.copy())

    return {"Q_s": Q_s, "Q_h": Q_h, "curve": curve, "snaps": snaps}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train a self-play tabular-Q "
                                             "hide-and-seek (CPU, no GPU).")
    ap.add_argument("--episodes", type=int, default=40000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    out = train(args.episodes, args.seed)
    c0, c1 = out["curve"][0], out["curve"][-1]
    print("\nLEARNED:")
    print(f"  seeker see-rate  {c0['seeker_skill']:.3f} -> {c1['seeker_skill']:.3f}")
    print(f"  hider evasion    {c0['hider_skill']:.3f} -> {c1['hider_skill']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
