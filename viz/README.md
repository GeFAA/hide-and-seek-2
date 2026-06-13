# Hide & Seek 2.0 — 3D Replay Viewer

A **clean, browser-based 3D replay viewer** for Hide & Seek 2.0 episodes,
built with [Three.js](https://threejs.org/). It loads a self-contained
**trajectory file** (plain JSON) and plays the episode back as an interactive
god-view scene: agents, boxes, ramps, walls, doors and decoys move through the
arena while you scrub, change speed, and toggle overlays such as vision cones,
fog of war, movement trails and a follow-cam.

The viewer is **decoupled from training**. It never imports JAX, numpy, or the
environment. It reads only the trajectory contract in
[`viz/schema.py`](schema.py), so you can record an episode on a GPU box and
watch it on a laptop with nothing installed but a web browser.

There is **no build step** and nothing to `npm install`: the page is plain ES
modules that pull Three.js from a CDN at runtime.

---

## Quick look (no training needed)

You do **not** need JAX, a GPU, or a trained policy to see the viewer working.
A pure-stdlib script generates a synthetic but schema-valid demo episode, and a
tiny stdlib HTTP server serves the page. From the repository root:

```bash
# 1. Generate the demo trajectory (pure Python stdlib — no jax/numpy needed).
#    Writes viz/web/trajectories/demo_trajectory.json
python viz/make_demo_trajectory.py

# 2. Serve viz/web on http://localhost:8000 (stdlib http.server, no deps).
python -m viz.serve

# 3. Open the viewer in a browser:
#    http://localhost:8000
```

> [!IMPORTANT]
> Open the viewer over **`http://`**, not by double-clicking the HTML file
> (`file://`). The page is loaded as native **ES modules**, and browsers refuse
> to load ES modules over the `file://` protocol (CORS / module-origin rules).
> Always go through `python -m viz.serve` (or any static HTTP server) so the
> URL begins with `http://localhost:8000`.

On **Windows PowerShell** the same three steps are:

```powershell
python viz\make_demo_trajectory.py
python -m viz.serve
# then open http://localhost:8000 in your browser
```

Stop the server with `Ctrl+C`.

---

## Record a real rollout

Once you have a trained policy (or even a random one) and a working JAX install,
you can export an **actual** episode to the same trajectory format and watch it
in the viewer. Recording lives in [`viz/recorder.py`](recorder.py), which uses
numpy to pull device arrays back to the host and serialize them.

There are two entry points:

- **`rollout_to_trajectory(...)`** — a one-call convenience helper: step the
  environment for an episode (with your action function / policy), capture every
  frame, and return a schema-valid trajectory document.
- **`TrajectoryRecorder`** — a lower-level recorder you drive yourself if you
  already have a rollout loop. Construct it with the static entity metadata,
  call `record_frame(...)` (or the env-aware equivalent) once per step, then
  `finish()` to get the document.

Typical usage after a JAX/GPU run (sketch — see the docstrings in
[`viz/recorder.py`](recorder.py) for the exact, current signatures):

```python
import jax
from config import default_config
from envs import HideAndSeekEnv
from viz.recorder import rollout_to_trajectory
from viz.schema import save_trajectory

config = default_config()
env = HideAndSeekEnv(config)

# `action_fn(obs, rng) -> action_dict` — your trained policy, or a random one.
doc = rollout_to_trajectory(
    env,
    action_fn=my_policy,                 # or a random action sampler
    rng=jax.random.PRNGKey(0),
    title="my first rollout",
)

# Drop it next to the demo so the viewer's file picker / loader finds it.
save_trajectory(doc, "viz/web/trajectories/my_rollout.json")
```

Then serve and open as in the Quick look:

```bash
python -m viz.serve
# http://localhost:8000  (load my_rollout.json from the viewer's UI)
```

The curated scenarios in `viz/web/trajectories/` (the demo, the named scenarios,
and `manifest.json`) are **committed** — they ship with the repo and are
published to GitHub Pages. Real recorded rollouts can be large, so put them in
**`viz/web/recordings/`** instead, which is **gitignored** and stays local. Load
a recording into the viewer from its file picker, or move it into
`trajectories/` and add a manifest entry if you want to publish it.

---

## Trajectory format

The viewer and the recorder communicate **only** through the trajectory
contract. The single source of truth, including builders and a dependency-free
validator, is [`viz/schema.py`](schema.py) — read it before changing anything on
either side.

In brief (`format="hns2-traj"`, `version=1`):

```jsonc
{
  "format": "hns2-traj",
  "version": 1,
  "meta": { /* title, seed, arena_size, dt, max_steps, prep_steps,
               entity_types[], max_agents, max_entities, n_frames */ },
  "entities": [ /* STATIC per-slot metadata, id-indexed (length E) */ ],
  "frames":   [ /* DYNAMIC per-step state, each with an `ent` array of length E */ ]
}
```

- **`meta`** — episode-wide constants: `arena_size` (the arena spans
  `[-arena_size/2, +arena_size/2]` in both `x` and `y`), `dt` (seconds per step,
  used for playback speed), `max_steps`, `prep_steps` (the hiders-only
  preparation phase), the canonical `entity_types` order, and counts.
- **`entities[]`** — one static record per entity slot, **id-indexed** (record
  `i` has `id == i`). Fields: `id`, `type`, `team` (`0` hider, `1` seeker, `-1`
  non-agent), `size`, `mass`, and `is_decoy` — the **true** decoy identity, a
  god-view fact the local agents cannot see.
- **`frames[]`** — one record per recorded step: `t`, `phase`
  (`"prep"`/`"main"`), cumulative scores `sh`/`ss`, `seen_any`, the `fog` patch
  list (`[x, y, r]`), and `ent` — an array of length `E`, **id-aligned** with
  `entities`. Each per-entity record carries position (`x`, `y`, and `z`
  elevation), heading `h`, and the status flags `a` (active), `lk` (locked),
  `hd` (held), `hb` (held-by id / `-1`), `no` (noise `0..1`), `dc` (decoy
  actively emitting), `gr` (grounded), `st` (stamina `0..1`, or `-1` for
  non-agents), and `sn` (seen by the opposing team this frame).

The `entity_types` order is part of the public contract and matches
`config.ENTITY_TYPES`:

```
hider, seeker, box_light, box_heavy, ramp, decoy, wall, door
```

The format is intentionally *static metadata + lean per-frame state*, so a full
240-step / ~22-entity episode is only a few hundred KB of JSON.

---

## On-screen controls

The viewer's control bar and keyboard shortcuts:

| Control | What it does |
| --- | --- |
| **Scrubber** (timeline slider) | Jump to any frame; shows the current step `t` and phase. |
| **Play / Pause** | Start or stop playback. |
| **Speed** | Playback rate multiplier (e.g. 0.25× … 4×); honors `meta.dt`. |
| **Vision cones** | Toggle the seekers' line-of-sight / vision-cone wedges. |
| **Fog** | Toggle the fog-of-war patches (`frames[].fog`). |
| **Reveal decoys** | God-view toggle: highlight which entities are *truly* decoys (`entities[].is_decoy`) vs. what they spoof. |
| **Trails** | Toggle fading movement trails behind moving entities. |
| **Follow-cam** | Toggle a camera that follows a selected agent instead of the free orbit camera. |

**Keyboard shortcuts**

| Key | Action |
| --- | --- |
| **Spacebar** | Play / pause. |
| **→ (Right arrow)** | Step forward one frame. |
| **← (Left arrow)** | Step backward one frame. |

The camera itself uses standard orbit controls (drag to rotate, scroll to zoom,
right-drag to pan) when follow-cam is off.

---

## Palette legend

The viewer renders a bright, clean scene in the OpenAI *Emergent Tool Use*
visual style. The default color language:

| Color | Meaning |
| --- | --- |
| **Soft white** | Background, tiled arena floor, walls and the surrounding cube field — the clean OpenAI look. |
| **Cool blue** | **Hiders** (team `0`). |
| **Warm red / orange** | **Seekers** (team `1`). |
| **Bright highlight ring/outline** | An entity currently **seen by the opposing team** (`sn == 1`). |
| **Neutral grey** | Static props: **walls** and **doors** (doors fade/clear when opened, i.e. inactive). |
| **Wood / tan** | Movable **boxes** (`box_light`, `box_heavy`) and **ramps**. |
| **Magenta / purple accent** | **Decoys** — and, with *Reveal decoys* on, the god-view marker on the entity's true identity. |
| **Soft volumetric haze** | **Fog** patches (toggle with *Fog*). |

Exact hues are defined in the viewer's web sources under
[`viz/web/`](web/); the table above is the intended reading.

---

## Requirements & notes

- **A modern browser** with ES-module support (any current Chrome, Firefox,
  Edge, or Safari).
- **Internet access at view time.** Three.js is loaded from a public **CDN**, so
  the machine running the browser needs network access the first time it loads
  the page (the browser will then cache it). There is intentionally **no bundler
  and no `node_modules`** — nothing to build or install for the frontend.
- **Serve over HTTP, not `file://`.** As noted above, ES modules will not load
  from the filesystem; always go through `python -m viz.serve` (which serves the
  `viz/web/` directory) or another static HTTP server.
- **Python ≥ 3.10** for the helper scripts. `make_demo_trajectory.py` and
  `serve` are **stdlib-only**; only `recorder.py` (real rollouts) needs numpy and
  a working JAX environment.

---

## Files in this directory

| Path | Purpose |
| --- | --- |
| [`schema.py`](schema.py) | The trajectory contract: builders + validator (stdlib only). |
| [`make_demo_trajectory.py`](make_demo_trajectory.py) | Generates the synthetic `demo_trajectory.json` (stdlib only). |
| [`recorder.py`](recorder.py) | Exports **real** JAX rollouts to the trajectory format (uses numpy). |
| `serve.py` | Stdlib static server: `python -m viz.serve` serves `viz/web/`. |
| `web/` | The Three.js viewer (HTML / CSS / ES modules) — no build step. |
| `web/trajectories/` | Committed scenario files + `manifest.json` (the demo and named scenarios). |
| `web/recordings/` | Local-only (gitignored) home for large real rollouts. |

See the top-level [`README.md`](../README.md) for the project overview.
