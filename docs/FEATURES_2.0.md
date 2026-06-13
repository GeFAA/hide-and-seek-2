# Hide & Seek 2.0 — New Features ("What's new in 2.0")

This document describes each mechanic that Hide & Seek 2.0 adds on top of a
faithful recreation of OpenAI's 2019 environment. For every feature:

- **Motivation** — why it exists;
- **Targeted emergent behavior** — the strategy we hope it provokes;
- **Config knobs** — the exact field names in [`config.py`](../config.py)
  (`EnvConfig` unless noted);
- **Implementation** — the file and the `# 2.0:` (or `# FIX:`) grep tag.

All shared dimensions are derived in `config.py`; never hard-code them. The
canonical entity taxonomy `ENTITY_TYPES` (in `config.py`) defines the one-hot
order and includes the new `decoy`, `box_heavy`, `wall`, and `door` types.

---

## 1. Variable mass & cooperative physics

**Motivation.** In the original game, any single agent could move any box. By
introducing *heavy* boxes that no single agent can budge, we make cooperation a
physical necessity rather than a nicety.

**Targeted emergent behavior.** Hiders learning to **co-push** a heavy box
together to barricade a doorway; seekers coordinating to clear or topple one;
implicit role assignment and timing between teammates.

**Config knobs** (`EnvConfig`):
- `box_light_mass = 1.0`, `box_heavy_mass = 6.0` — the two box masses.
- `coop_required_agents = 2` — number of *distinct* simultaneous pushers a heavy
  box needs.
- `coop_force_threshold = 10.0` — combined applied force magnitude required to
  budge a heavy box.
- Entity types `box_light` (id 2) and `box_heavy` (id 3) come from
  `ENTITY_TYPES` in `config.py`.

**Implementation.** `envs/physics.py`, in `physics_step`. A box accelerates by
`F_net / mass`; for a heavy box the net *agent* force is gated to ~0 unless both
the combined pusher force ≥ `coop_force_threshold` **and** the number of distinct
pushers ≥ `coop_required_agents`. Grep tag: **`# 2.0: cooperative physics`**.

---

## 2. Strict Newtonian ground-contact (anti box-surfing fix)

**Motivation.** OpenAI's agents famously discovered "box surfing" — exploiting
the physics to ride a box around the arena in a way the designers never intended.
2.0 closes this exploit with a strict ground-contact rule.

**Targeted behavior.** Climbing onto boxes/ramps remains useful for *vision and
reach*, but an agent perched above the ground (`z > 0`) cannot self-propel —
removing the degenerate surfing strategy and keeping locomotion Newtonian.

**Config knobs** (`EnvConfig`):
- `ground_contact_required = True` — master switch for the gate.
- `agent_radius`, `agent_max_force`, `linear_damping` — the locomotion model the
  gate sits in front of.

**Implementation.** `envs/physics.py`. Each agent's `grounded` flag (in
`PhysicsState`) is `True` only when its elevation `z` is ~0; locomotion force is
multiplied by `grounded.astype(float32)` when `ground_contact_required`. Grep
tag: **`# FIX: strict Newtonian ground-contact (no box-surfing)`**.

> This is also surfaced to the critic as a privileged `grounded` feature
> (`global_entities[..., Fe+1]`, see `CONTRACT.md` §3.2) but withheld from the
> actor's local view.

---

## 3. Stamina

**Motivation.** Unlimited sprinting and infinite heavy-pushing trivialize chases
and constructions. A stamina budget adds resource management and pacing.

**Targeted behavior.** Seekers learning to **conserve** stamina and burst-sprint
at the right moment; hiders timing a heavy-box barricade before stamina runs
out; agents disengaging to regen rather than over-committing.

**Config knobs** (`EnvConfig`):
- `stamina_max = 100.0` — full tank.
- `stamina_regen = 8.0` /sec while not sprinting.
- `sprint_drain = 20.0` /sec at full sprint.
- `heavy_push_drain = 15.0` extra /sec while moving heavy objects.
- `sprint_force_mult = 1.8` — force multiplier when sprint is engaged.

**Implementation.** `envs/physics.py`. Sprint multiplies applied force by
`sprint_force_mult` and drains stamina; pushing heavy objects adds
`heavy_push_drain`; stamina regenerates when idle and clamps to
`[0, stamina_max]`; zero stamina disables sprint. Stamina lives in
`GameState.stamina` and is exposed (normalized) in the `self` vector
(`self[8]`, see `CONTRACT.md` §3.3). Grep tag: **`# 2.0: stamina`**.

---

## 4. Decoys & sensory manipulation

**Motivation.** Add deception as a first-class, *physical* tool: an object that
lies to the opponent's sensors.

**Targeted behavior.** Hiders deploying a decoy to **draw seekers away** from
their real hiding spot; seekers learning (via memory + the critic's privileged
`true_is_decoy`) to become skeptical of suspicious signals.

**Config knobs** (`EnvConfig`):
- `n_decoys_max = 2` — padding count for decoy slots.
- `decoy_noise_radius = 6.0` — how far the spoofed noise reaches.
- `decoy_active_steps = 40` — how long a decoy stays active once triggered.
- The `decoy` entity type (`ENTITY_TYPES` in `config.py`).
- Triggered via the discrete `interact` action `[2]` (decoy on/off), see
  `ModelConfig.action_discrete_nvec` and `CONTRACT.md` §4.

**Implementation.** Triggering sets `GameState.decoy_timer` and makes the decoy
broadcast spoofed `emitted_noise` and a **fake type one-hot** in the *local*
(`entities`) observation only. The ground-truth identity is `GameState.is_decoy`
and is exposed to the **critic only** as `true_is_decoy`
(`global_entities[..., Fe]`, `CONTRACT.md` §3.1–3.2). Grep tag:
**`# 2.0: deception`** in `envs/hide_and_seek.py` (decoy observation spoofing)
and the decoy timer update.

---

## 5. Fog of war & dynamic lighting

**Motivation.** The original sensing was essentially clean line-of-sight. Fog
patches create graded, dynamic partial observability the agents must reason
around.

**Targeted behavior.** Hiders **exploiting fog** to break line-of-sight and
escape; seekers learning to flush opponents out of or sweep around fog; both
teams leaning on GRU memory for object permanence when targets are obscured.

**Config knobs** (`EnvConfig`):
- `lidar_n_rays = 30`, `lidar_range = 8.0` — lidar sensing.
- `vision_range = 9.0`, `vision_cone_deg = 135.0` — forward vision cone.
- `n_fog_patches = 3`, `fog_radius = 2.5` — fog geometry (centers in
  `GameState.fog_pos`).
- `fog_vision_mult = 0.45` — range multiplier for rays passing through fog.

**Implementation.** `utils/visibility.py` performs GPU ray-casting (LOS, lidar,
vision cone) and attenuates a ray's effective range when it crosses a fog patch,
producing the per-agent `entity_mask`. The same *true* visibility drives the
reward (`any_hider_seen`). Grep tag: **`# 2.0: fog of war`**.

---

## 6. Destructible walls & doors

**Motivation.** Make the arena layout itself something agents can reshape —
breaking through fragile walls or opening doors to create or deny chokepoints.

**Targeted behavior.** Seekers **ramming** a fragile wall at speed to breach a
hiders' fort; teams accumulating contact to **open a door** into a sealed room;
hiders defending doorways with heavy boxes (ties back to feature 1).

**Config knobs** (`EnvConfig`):
- `n_walls_max = 6`, `n_doors_max = 2` — padding counts.
- `wall_break_speed = 6.0` — ram speed above which a fragile wall breaks.
- `door_open_steps = 25` — contact-steps required to open a door.
- `wall` and `door` entity types (`ENTITY_TYPES` in `config.py`).
- Per-entity state: `GameState.wall_hp`, `GameState.door_progress`.

**Implementation.** `envs/physics.py` decrements `wall_hp` on high-speed contact
(`>= wall_break_speed`); at `wall_hp <= 0` the wall's `active` flag flips off.
Doors accumulate `door_progress` (0→1) under sustained contact and open past a
threshold derived from `door_open_steps`. Grep tag:
**`# 2.0: destructible env`** in `envs/physics.py` and `envs/hide_and_seek.py`.

---

## Cross-feature notes

- **Deception is asymmetric by construction.** The actor's local observation
  (`entities`, `Fe = entity_feat_dim = 20`) can be *spoofed* by decoys, while the
  critic's global observation (`global_entities`, `Fg = 22`) carries the
  privileged `true_is_decoy` and `grounded` flags. This is the CTDE backbone
  (see `ARCHITECTURE.md` §3) and is what lets the critic give honest value
  estimates while the actor must cope with lies.
- **Reward is unchanged in spirit.** The team game reward (`CONTRACT.md` §7)
  still hinges on whether *any* hider is truly seen by *any* seeker; the new
  mechanics change *how* agents achieve (or prevent) that, not the objective.
- **A tiny config for testing.** `debug_config()` in `config.py` shrinks
  `num_envs`/`num_steps` for fast CPU smoke tests; all the 2.0 knobs above keep
  their defaults so the mechanics are still exercised.
