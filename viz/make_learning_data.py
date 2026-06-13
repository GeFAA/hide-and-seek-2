"""
viz/make_learning_data.py -- synthetic, illustrative TRAINING-PROGRESS data for
the viewer's "Learning" tab. PURE STDLIB. Writes ``viz/web/learning.json``.

This is **not** measured training output (the JAX trainer needs a GPU). It is a
hand-tuned, believable *autocurriculum* curve so that even a beginner can SEE
progress: an oscillating hider-vs-seeker "arms race" where new behaviours unlock
over training, mirroring the emergent phases of the original *Emergent Tool Use*
work, adapted to the 2.0 mechanics (cooperative physics, decoys, door blocking,
ramp locking).

learning.json schema (``format="hns2-learning"``, ``version=1``)
----------------------------------------------------------------
{
  "meta": { "title", "total_timesteps", "unit", "synthetic", "note" },
  "series": {                         # parallel arrays, x = "t"
    "t":              [int, ...],     # environment steps (x-axis)
    "hider_winrate":  [0..1, ...],   # rolling win rate of the hider team
    "seeker_winrate": [0..1, ...],   # == 1 - hider_winrate
    "hider_elo":      [int, ...],    # ELO rating (both climb as skill grows)
    "seeker_elo":     [int, ...],
    "episode_len":    [int, ...]     # avg surviving episode length (flavour)
  },
  "milestones": [                     # behaviours that emerge, in step order
    { "id", "step", "team":"hider"|"seeker", "title", "desc", "emoji" }, ...
  ],
  "teams": { "hider": {elo,winrate,tactic}, "seeker": {elo,winrate,tactic} }
}
"""
from __future__ import annotations

import json
import math
import os

TOTAL = 200_000_000      # total environment steps the curve spans
N = 41                   # number of sample points

# (step_millions, team, id, title, description, emoji)
MILESTONES = [
    (2,   "seeker", "chase",   "Running & Chasing",
     "Seekers learn to chase hiders down in the open arena.", "\U0001F3C3"),
    (6,   "hider",  "flee",    "Fleeing & Hiding",
     "Hiders learn to run and break the seekers' line of sight.", "\U0001F648"),
    (22,  "hider",  "fort",    "Fort Building",
     "Hiders grab boxes and barricade a shelter during the prep phase.", "\U0001F9F1"),
    (40,  "hider",  "coop",    "Cooperative Push",
     "Two hiders shove a heavy box together — real teamwork.", "\U0001F91D"),
    (88,  "seeker", "ramp",    "Ramp Use",
     "Seekers drag a ramp to the fort and climb in to breach it.", "\U0001F4D0"),
    (115, "hider",  "rampdef", "Ramp Defense",
     "Hiders lock ramps away before the seekers are even released.", "\U0001F512"),
    (150, "hider",  "decoy",   "Sensory Deception",
     "Hiders trigger decoys to send the seekers the wrong way.", "\U0001F3AD"),
    (172, "hider",  "door",    "Door Blocking",
     "Hiders jam a doorway with a heavy box to hold a chokepoint.", "\U0001F6AA"),
]


def _sigmoid(x: float) -> float:
    # numerically safe logistic
    if x < -60:
        return 0.0
    if x > 60:
        return 1.0
    return 1.0 / (1.0 + math.exp(-x))


def _hider_winrate(step: float) -> float:
    """Oscillating arms-race win rate: each milestone tilts the balance toward
    the team that discovered it, via a smooth sigmoid ramp."""
    wr = 0.5 - 0.16 * _sigmoid((4e6 - step) / 3e6)  # seekers dominate very early
    for (sm, team, *_rest) in MILESTONES:
        s = sm * 1e6
        d = 1.0 if team == "hider" else -1.0
        wr += d * 0.16 * _sigmoid((step - s) / 6e6)
    return max(0.18, min(0.82, wr))


def build() -> dict:
    ts = [int(round(i * TOTAL / (N - 1))) for i in range(N)]
    hw = [round(_hider_winrate(t), 4) for t in ts]
    sw = [round(1.0 - x, 4) for x in hw]

    hider_elo, seeker_elo = [], []
    for i, t in enumerate(ts):
        prog = t / TOTAL
        base = 1200.0 + 300.0 * prog          # both teams climb as skill grows
        gap = 160.0 * (hw[i] - 0.5) * 2.0      # current dominance -> rating gap
        hider_elo.append(int(round(base + gap * 0.5)))
        seeker_elo.append(int(round(base - gap * 0.5)))

    episode_len = [int(round(120 + 90 * (hw[i] - 0.3))) for i in range(N)]

    return {
        "format": "hns2-learning",
        "version": 1,
        "meta": {
            "title": "Training progress (illustrative)",
            "total_timesteps": TOTAL,
            "unit": "environment steps",
            "synthetic": True,
            "note": ("Hand-tuned illustrative autocurriculum — not measured "
                     "output. The JAX trainer needs a GPU; this curve exists so "
                     "the Learning tab tells the emergence story for newcomers."),
        },
        "series": {
            "t": ts,
            "hider_winrate": hw,
            "seeker_winrate": sw,
            "hider_elo": hider_elo,
            "seeker_elo": seeker_elo,
            "episode_len": episode_len,
        },
        "milestones": [
            {"id": mid, "step": int(sm * 1e6), "team": team,
             "title": title, "desc": desc, "emoji": emoji}
            for (sm, team, mid, title, desc, emoji) in MILESTONES
        ],
        "teams": {
            "hider": {"elo": hider_elo[-1], "winrate": hw[-1],
                      "tactic": "Fort, ramp-lock & decoys"},
            "seeker": {"elo": seeker_elo[-1], "winrate": sw[-1],
                       "tactic": "Ramp rush & door pressure"},
        },
    }


def main() -> int:
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "web", "learning.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    doc = build()
    with open(out, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))
    s = doc["series"]
    print(f"wrote {out}")
    print(f"  points={len(s['t'])}  milestones={len(doc['milestones'])}  "
          f"final hider_winrate={s['hider_winrate'][-1]}  "
          f"hider_elo={s['hider_elo'][-1]} seeker_elo={s['seeker_elo'][-1]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
