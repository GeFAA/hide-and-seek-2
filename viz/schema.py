"""
viz/schema.py -- The **trajectory file contract** for the Hide & Seek 2.0 viewer.

This is the single source of truth that ties together three things:

  * ``viz/recorder.py``            -- exports REAL rollouts from the JAX env (uses numpy).
  * ``viz/make_demo_trajectory.py`` -- a synthetic episode generator (pure stdlib,
                                       runs with no jax/numpy) so the 3D viewer works
                                       out of the box on any machine.
  * ``viz/web/*``                  -- the Three.js dark viewer that loads & plays a file.

The format is plain JSON. It is intentionally *static per-entity metadata* +
*lean per-frame dynamic state*, so a 240-step / 22-entity episode stays a few
hundred KB.

Stdlib only -- importable and runnable everywhere (no jax, no numpy).

-------------------------------------------------------------------------------
FILE FORMAT  (format="hns2-traj", version=1)
-------------------------------------------------------------------------------
{
  "format": "hns2-traj",
  "version": 1,
  "meta": {
    "title": str, "seed": int,
    "arena_size": float,           # arena spans [-arena_size/2, +arena_size/2] in x,y
    "dt": float,                   # seconds per step (for playback speed)
    "max_steps": int, "prep_steps": int,
    "entity_types": [str, ...],    # == ENTITY_TYPES below (one-hot order)
    "max_agents": int, "max_entities": int,
    "n_frames": int
  },
  "entities": [                    # STATIC per-slot metadata, id-indexed (length E)
    { "id": int, "type": str, "team": int,   # team: 0 hider, 1 seeker, -1 non-agent
      "size": float, "mass": float, "is_decoy": bool }   # is_decoy = TRUE identity (god-view)
  ],
  "frames": [                      # DYNAMIC state, one per recorded step
    { "t": int, "phase": "prep"|"main",
      "sh": float, "ss": float,    # cumulative hider / seeker score
      "seen_any": bool,            # any hider currently seen by any seeker
      "fog": [[x, y, r], ...],     # fog patch centers + radius
      "ent": [                     # length E, id-aligned with "entities"
        { "id": int,
          "x": float, "y": float, "z": float,   # z = elevation (climbing a box/ramp)
          "h": float,              # heading (radians); agents only, else 0
          "a": 0|1,                # active (exists this episode / not broken/opened)
          "lk": 0|1,               # locked in place
          "hd": 0|1,               # currently held by an agent
          "hb": int,               # held-by agent id, or -1
          "no": float,             # emitted noise intensity 0..1 (decoys spoof this)
          "dc": 0|1,               # decoy actively emitting
          "gr": 0|1,               # grounded (agents) -- 0 while airborne on top of a box
          "st": float,             # stamina 0..1 (agents), else -1
          "sn": 0|1 }              # seen by the OPPOSING team this frame (drives highlight)
      ]
    }
  ]
}
-------------------------------------------------------------------------------
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

FORMAT = "hns2-traj"
VERSION = 1

# Must match config.ENTITY_TYPES order exactly (one-hot index == list index).
ENTITY_TYPES = (
    "hider", "seeker", "box_light", "box_heavy",
    "ramp", "decoy", "wall", "door",
)

# Per-frame entity keys, in documentation order (the viewer reads by key, not position).
FRAME_ENT_KEYS = (
    "id", "x", "y", "z", "h", "a", "lk", "hd", "hb", "no", "dc", "gr", "st", "sn",
)

# Teams
TEAM_HIDER = 0
TEAM_SEEKER = 1
TEAM_NONE = -1


# --------------------------------------------------------------------------- builders
def make_entity_meta(
    id: int, type: str, team: int, size: float, mass: float, is_decoy: bool
) -> Dict[str, Any]:
    """Build one STATIC per-slot metadata record (goes in top-level ``entities``)."""
    assert type in ENTITY_TYPES, f"unknown entity type {type!r}"
    return {
        "id": int(id),
        "type": type,
        "team": int(team),
        "size": float(size),
        "mass": float(mass),
        "is_decoy": bool(is_decoy),
    }


def make_frame_ent(
    id: int,
    x: float, y: float, z: float = 0.0, h: float = 0.0,
    a: int = 1, lk: int = 0, hd: int = 0, hb: int = -1,
    no: float = 0.0, dc: int = 0, gr: int = 1, st: float = -1.0, sn: int = 0,
) -> Dict[str, Any]:
    """Build one DYNAMIC per-entity record for a single frame."""
    return {
        "id": int(id),
        "x": round(float(x), 4), "y": round(float(y), 4), "z": round(float(z), 4),
        "h": round(float(h), 4),
        "a": int(a), "lk": int(lk), "hd": int(hd), "hb": int(hb),
        "no": round(float(no), 4), "dc": int(dc), "gr": int(gr),
        "st": round(float(st), 4), "sn": int(sn),
    }


def make_frame(
    t: int, phase: str, sh: float, ss: float, seen_any: bool,
    fog: List[List[float]], ent: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assemble a single frame."""
    assert phase in ("prep", "main"), f"bad phase {phase!r}"
    return {
        "t": int(t),
        "phase": phase,
        "sh": round(float(sh), 4),
        "ss": round(float(ss), 4),
        "seen_any": bool(seen_any),
        "fog": [[round(float(c[0]), 4), round(float(c[1]), 4), round(float(c[2]), 4)] for c in fog],
        "ent": ent,
    }


def make_trajectory(
    meta: Dict[str, Any], entities: List[Dict[str, Any]], frames: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Wrap metadata + static entities + frames into the top-level document."""
    doc = {
        "format": FORMAT,
        "version": VERSION,
        "meta": dict(meta),
        "entities": entities,
        "frames": frames,
    }
    doc["meta"]["n_frames"] = len(frames)
    return doc


def save_trajectory(doc: Dict[str, Any], path: str) -> None:
    """Validate then write a trajectory document to ``path`` as JSON."""
    problems = validate_trajectory(doc)
    if problems:
        raise ValueError("invalid trajectory:\n  - " + "\n  - ".join(problems))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f, separators=(",", ":"))


# --------------------------------------------------------------------------- validator
def validate_trajectory(doc: Dict[str, Any]) -> List[str]:
    """Return a list of human-readable problems. Empty list == valid.

    Cheap, dependency-free structural validation -- enough to catch the mistakes
    that silently break the viewer (missing keys, id misalignment, wrong lengths).
    """
    p: List[str] = []
    if not isinstance(doc, dict):
        return ["top-level value is not an object"]
    if doc.get("format") != FORMAT:
        p.append(f"format must be {FORMAT!r}, got {doc.get('format')!r}")
    if doc.get("version") != VERSION:
        p.append(f"version must be {VERSION}, got {doc.get('version')!r}")

    meta = doc.get("meta", {})
    for k in ("arena_size", "dt", "max_steps", "prep_steps", "entity_types",
              "max_agents", "max_entities"):
        if k not in meta:
            p.append(f"meta.{k} missing")
    if meta.get("entity_types") and tuple(meta["entity_types"]) != ENTITY_TYPES:
        p.append("meta.entity_types does not match the canonical ENTITY_TYPES order")

    entities = doc.get("entities")
    if not isinstance(entities, list) or not entities:
        return p + ["entities must be a non-empty list"]
    E = len(entities)
    for i, e in enumerate(entities):
        if e.get("id") != i:
            p.append(f"entities[{i}].id must equal its index ({i}), got {e.get('id')!r}")
        if e.get("type") not in ENTITY_TYPES:
            p.append(f"entities[{i}].type invalid: {e.get('type')!r}")

    frames = doc.get("frames")
    if not isinstance(frames, list) or not frames:
        return p + ["frames must be a non-empty list"]
    for fi, fr in enumerate(frames):
        if fr.get("phase") not in ("prep", "main"):
            p.append(f"frames[{fi}].phase invalid: {fr.get('phase')!r}")
        ent = fr.get("ent")
        if not isinstance(ent, list) or len(ent) != E:
            p.append(f"frames[{fi}].ent must have length {E} (got {len(ent) if isinstance(ent, list) else 'n/a'})")
            continue
        for ei, en in enumerate(ent):
            if en.get("id") != ei:
                p.append(f"frames[{fi}].ent[{ei}].id misaligned (expected {ei}, got {en.get('id')!r})")
            missing = [k for k in FRAME_ENT_KEYS if k not in en]
            if missing:
                p.append(f"frames[{fi}].ent[{ei}] missing keys: {missing}")
                break  # one report per entity is enough
    return p


def load_trajectory(path: str) -> Dict[str, Any]:
    """Load + validate a trajectory file."""
    with open(path, "r", encoding="utf-8") as f:
        doc = json.load(f)
    problems = validate_trajectory(doc)
    if problems:
        raise ValueError("invalid trajectory:\n  - " + "\n  - ".join(problems))
    return doc


__all__ = [
    "FORMAT", "VERSION", "ENTITY_TYPES", "FRAME_ENT_KEYS",
    "TEAM_HIDER", "TEAM_SEEKER", "TEAM_NONE",
    "make_entity_meta", "make_frame_ent", "make_frame", "make_trajectory",
    "save_trajectory", "load_trajectory", "validate_trajectory",
]
