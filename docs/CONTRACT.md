# Hide & Seek 2.0 — Internal Interface Contract

> **Authoritative spec.** Every module (`envs/`, `models/`, `trainers/`,
> `utils/`) MUST conform to the structures, names and dimensions defined here.
> All shared dimensions are *derived* in `config.py` — read them from there,
> never hard-code them. If you need to change an interface, change it here
> first, then in `config.py`, then in the code.

---

## 0. Tech stack & conventions

- **JAX + Flax (linen) + Optax + distrax.** End-to-end on device
  (PureJaxRL / JaxMARL style): env step, physics, and learning all run inside
  `jax.jit` / `jax.lax.scan` with **zero host↔device copies** in the hot loop.
- Env follows a **Brax-style functional API**: pure functions `reset(key) ->
  State` and `step(state, action) -> State`. One call == ONE environment; the
  trainer `jax.vmap`s over `num_envs`. No Python loops over environments.
- Randomness is explicit: every stochastic function takes a `jax.random.PRNGKey`.
- **No `Date.now`/wall-clock/`Math.random`** inside jitted code.
- Arrays are `jnp.float32` unless stated; masks are `bool`; ids are `int32`.
- Forbidden everywhere in the hot path: Python `if` on traced values
  (use `jnp.where`/`lax.select`), `.item()`, host callbacks.

---

## 1. Package layout

```
config.py                 # single source of truth for all dims/hyperparams (DONE)
docs/CONTRACT.md          # this file (DONE)

envs/
  __init__.py             # exports HideAndSeekEnv, State, GameState
  state.py                # PhysicsState, GameState, State (flax.struct.dataclass)
  physics.py              # vectorized 2.5D JAX physics + cooperative force + ground-contact
  procedural.py           # vectorized per-episode randomization (maps/teams/props/fog)
  hide_and_seek.py        # HideAndSeekEnv: reset/step/reward/observe + 2.0 mechanics

models/
  __init__.py             # exports EntityTransformer, ActorRNN, CriticRNN, ScannedGRU
  transformer.py          # permutation-invariant entity encoder (masked self-attention)
  memory.py               # ScannedGRU: PureJaxRL-style RNN with episode resets
  actor.py                # ActorRNN: encoder -> GRU -> hybrid heads (distrax dists)
  critic.py               # CriticRNN: CTDE privileged encoder -> GRU -> scalar value
  networks.py             # init helpers, param-count utils, hidden-state factories

trainers/
  __init__.py             # exports make_train, EloManager, OpponentPool
  rollout.py              # Transition pytree + batchify/step glue
  selfplay.py             # OpponentPool + ELO-based opponent sampling + snapshots
  mappo.py                # make_train(config) -> jitted train fn (scan rollout, GAE, update)

utils/
  __init__.py
  visibility.py           # GPU ray-casting: LOS, lidar, vision cone, fog attenuation -> masks
  elo.py                  # expected_score, update_elo
  spaces.py               # lightweight Box/Discrete/Dict space descriptors
  pytree.py               # batchify/unbatchify multi-agent dicts, tree utils
  logging.py              # metric aggregation, optional wandb/tensorboard

tests/                    # pytest sanity tests (shapes, jit, vmap, elo, visibility)
train.py                  # CLI entrypoint: build config -> make_train -> run
examples/quickstart.py    # minimal random-rollout demo (guards missing jax)
```

Run from repo root with `PYTHONPATH=.`. Imports are absolute from root, e.g.
`from config import Config`, `from envs import HideAndSeekEnv`,
`from utils.pytree import batchify`.

---

## 2. State structures (`envs/state.py`)

All are `flax.struct.dataclass` (registered pytrees, jit/vmap-safe). Shapes are
**per single environment**; the trainer adds a leading `num_envs` axis via vmap.

Let `A = cfg.env.max_agents`, `E = cfg.env.max_entities`.

```python
@flax.struct.dataclass
class PhysicsState:
    pos:        jnp.ndarray  # (E, 3) float32   x, y, z(height/elevation)
    vel:        jnp.ndarray  # (E, 3) float32
    heading:    jnp.ndarray  # (E,)   float32   facing angle (radians), agents only meaningful
    ang_vel:    jnp.ndarray  # (E,)   float32
    mass:       jnp.ndarray  # (E,)   float32   (heavy boxes large; static walls = inf-proxy)
    size:       jnp.ndarray  # (E,)   float32   radius/half-extent for collision
    type_id:    jnp.ndarray  # (E,)   int32     index into ENTITY_TYPES
    grounded:   jnp.ndarray  # (E,)   bool      agent in contact with ground (anti-surf gate)
    active:     jnp.ndarray  # (E,)   bool      entity exists this episode (padding mask)

@flax.struct.dataclass
class GameState:
    team:        jnp.ndarray  # (A,)  int32   0=hider 1=seeker, -1 for non-agents/pad
    stamina:     jnp.ndarray  # (A,)  float32
    holding:     jnp.ndarray  # (A,)  int32   entity id being held, -1 if none
    locked:      jnp.ndarray  # (E,)  bool    object locked in place by its team
    locked_by:   jnp.ndarray  # (E,)  int32   team that locked it, -1 if none
    is_decoy:    jnp.ndarray  # (E,)  bool    TRUE decoy identity (privileged!)
    decoy_timer: jnp.ndarray  # (E,)  int32   >0 while a decoy is actively emitting
    emitted_noise: jnp.ndarray # (E,) float32 noise intensity each entity broadcasts
    wall_hp:     jnp.ndarray  # (E,)  float32 destructible wall health (<=0 => broken/inactive)
    door_progress: jnp.ndarray # (E,) float32 0..1 door-open progress
    fog_pos:     jnp.ndarray  # (n_fog_patches, 2) float32 fog patch centers
    phase:       jnp.ndarray  # ()    int32   0=prep, 1=main
    step:        jnp.ndarray  # ()    int32

@flax.struct.dataclass
class State:
    physics: PhysicsState
    game:    GameState
    obs:     dict            # see §3 (the observation contract)
    reward:  jnp.ndarray     # (A,) float32  per-agent reward (team reward broadcast)
    done:    jnp.ndarray     # () bool       episode terminated/truncated
    info:    dict            # metrics: {"hiders_reward","seekers_reward","seen_frac",...}
    key:     jnp.ndarray     # PRNGKey carried for stochastic dynamics
```

`HideAndSeekEnv.reset/step` return a fresh `State`. `done=True` auto-resets are
handled by the **trainer** wrapper (so `step` itself stays branch-free), per
PureJaxRL convention.

---

## 3. Observation contract (`State.obs` dict)

`obs` is a **dict of stacked arrays**, leading axis = agent index `A`.
Decentralized actor inputs are masked; the critic gets privileged global inputs.

Let `Fe = cfg.model.entity_feat_dim`, `Fg = cfg.model.global_entity_feat_dim`,
`Fs = cfg.model.self_feat_dim`.

| key                | shape          | who   | description |
|--------------------|----------------|-------|-------------|
| `entities`         | `(A, E, Fe)`   | actor | per-agent **local** entity tokens (relative coords) |
| `entity_mask`      | `(A, E)` bool  | actor | visibility mask: active AND in LOS/lidar/vision/fog |
| `self`             | `(A, Fs)`      | actor | proprioception of the observing agent |
| `global_entities`  | `(E, Fg)`      | critic| **absolute** entity tokens + privileged extras |
| `global_mask`      | `(E,)` bool    | critic| `active` (existence) mask only — NOT visibility |
| `agent_active`     | `(A,)` bool    | both  | which agent slots are real this episode |

### 3.1 Local entity vector layout (`Fe = 12 + N_ENTITY_TYPES = 20`)
Index order is fixed:
```
[0:3]   rel_pos      (entity_pos - observer_pos), zeroed when not visible
[3:6]   rel_vel      (entity_vel - observer_vel)
[6]     dist         euclidean distance observer->entity (normalized by arena_size)
[7]     mass         normalized
[8:8+T] type_onehot  one-hot over ENTITY_TYPES (T = N_ENTITY_TYPES = 8)
[8+T]   locked       1.0 if locked
[9+T]   emitted_noise intensity the entity APPEARS to emit (decoys spoof this!)
[10+T]  is_held      1.0 if currently grabbed by some agent
[11+T]  size
```
> **Deception note:** in `entities`/local view, a *decoy* is one-hot encoded as
> whatever it is mimicking and carries spoofed `emitted_noise`; the actor cannot
> tell it from a real hider. Only the critic sees `true_is_decoy` (below).

### 3.2 Global entity vector layout (`Fg = Fe + 2 = 22`)
Same first `Fe` features but in **absolute** coords and unmasked, plus:
```
[Fe]    true_is_decoy  1.0 if the entity is really a decoy
[Fe+1]  grounded       1.0 if agent is ground-contacting (privileged for critic)
```

### 3.3 Self vector layout (`Fs = 14`)
```
[0:3]   pos (normalized)        [3:6] vel
[6:8]   facing (sin, cos)       [8]   stamina (normalized 0..1)
[9:11]  team one-hot (hider,seeker)
[11]    prep_flag (1 during prep phase)
[12]    holding_flag            [13]  grounded_flag
```

---

## 4. Action contract

Per single env, `action` is a dict (leading axis `A`):

```python
action = {
    "move":     jnp.ndarray,  # (A, action_move_dim=3) float in [-1,1]: fx, fy, torque
    "interact": jnp.ndarray,  # (A, n_discrete=3) int: [grab, lock, decoy] categoricals
}
```

`env.step` consumes this dict. Inactive agent slots (`~agent_active`) have their
actions ignored internally. **Strict ground-contact gate:** `move` force is
multiplied by `physics.grounded` (anti box-surfing) — implemented in
`physics.py`, see §6.

---

## 5. Model API (`models/`)

All are `flax.linen.Module`. Hidden state for the GRU is
`(*batch, gru_hidden)` and is reset on episode boundaries via a `reset` mask
(PureJaxRL `ScannedGRU` convention: inputs are `(obs, done)` tuples over time).

```python
class EntityTransformer(nn.Module):
    cfg: ModelConfig
    # __call__(entities, entity_mask, self_feat) -> embedding (..., d_model)
    # Permutation-invariant: self-attention over E tokens with key_padding mask,
    # then masked mean-pool; `self_feat` injected as a learned query / extra token.

class ScannedGRU(nn.Module):
    hidden_size: int
    # __call__(carry, x) where x=(inputs, resets); applies nn.GRUCell over a
    # leading TIME axis with lax.scan, zeroing carry where resets==True.
    # staticmethod initialize_carry(batch_dims, hidden_size) -> carry

class ActorRNN(nn.Module):
    cfg: ModelConfig
    # __call__(carry, (obs, dones)) -> (new_carry, pi) where
    #   obs uses keys: entities, entity_mask, self, agent_active
    #   pi = (move_dist: distrax dist over R^3, interact_dists: list[distrax.Categorical])
    # Uses EntityTransformer -> ScannedGRU -> heads. LOCAL inputs only (decentralized).

class CriticRNN(nn.Module):
    cfg: ModelConfig
    # __call__(carry, (global_obs, dones)) -> (new_carry, value (...,))
    #   global_obs uses keys: global_entities, global_mask  (PRIVILEGED, CTDE).
```

`models/networks.py` provides `init_actor`, `init_critic`, `initialize_carries`,
and an `ActorCritic` convenience bundling both (separate params).

Action (log)probs: `move` via the Gaussian's `log_prob` summed over the 3 dims;
each discrete head contributes its categorical `log_prob`; total logprob is the
sum. Entropy is the sum of component entropies. Provide a single
`sample_and_logprob(pi, key)` and `eval_logprob(pi, action)` helper in
`models/actor.py`.

---

## 6. Physics & 2.0 mechanics (`envs/physics.py`, `envs/hide_and_seek.py`)

Vectorized 2.5D: planar (x,y) dynamics + scalar elevation `z` used only to model
climbing/surfing. Single function `physics_step(physics, game, forces, cfg) ->
(physics, contact_info)` doing `physics_substeps` semi-implicit Euler updates.

**MUST implement explicitly (these are graded deliverables):**

1. **Variable mass & cooperative push.** A box accelerates by
   `F_net / mass`. For a *heavy* box, only count agent push forces whose
   **combined magnitude** ≥ `coop_force_threshold` AND number of distinct
   pushers ≥ `coop_required_agents`; otherwise net agent force on it is clamped
   to ~0 (it "won't budge"). Comment this block with `# 2.0: cooperative physics`.
2. **Strict ground-contact gate (anti box-surfing).** Compute `grounded` per
   agent (an agent is *not* grounded while its elevation `z > 0`, i.e. it has
   climbed onto a box/ramp/wall). Locomotion force is multiplied by
   `grounded.astype(f32)` when `cfg.ground_contact_required`. Comment with
   `# FIX: strict Newtonian ground-contact (no box-surfing)`.
3. **Stamina.** Sprint multiplies force by `sprint_force_mult` and drains
   stamina; pushing heavy objects adds drain; regen when idle. Zero stamina
   disables sprint. (`# 2.0: stamina`)
4. **Decoys.** Activating a decoy sets `decoy_timer`, makes it broadcast spoofed
   `emitted_noise` and a fake type signature in *local* observations only.
5. **Fog / lighting.** `utils.visibility` attenuates vision/lidar range for rays
   passing through fog patches. (`# 2.0: fog of war`)
6. **Destructible walls / doors.** Wall `wall_hp` decreases on high-speed ramming
   (`>= wall_break_speed`); at `<=0` the wall becomes inactive. Doors accumulate
   `door_progress` under sustained contact and open past a threshold.

Grabbing/locking: an agent within reach may `grab` the nearest grabbable entity
(rigidly attaching it) and `lock` it (immovable, only same team can unlock).

---

## 7. Reward (`hide_and_seek.py`)

Team game reward, broadcast to each team's agents (per-agent vector `(A,)`):

- During **prep** (`step < prep_steps`): reward 0 for both teams (or small shaping).
- During **main**: each step, `any_hider_seen = ∃ hider visible to ∃ seeker`
  (use the *true* visibility, computed via `utils.visibility`).
  - hiders: `+reward_scale` if NOT seen, else `-reward_scale`.
  - seekers: the negation.
- `info` carries `hiders_reward`, `seekers_reward`, `seen_frac`, plus 2.0 metrics
  (`n_ramps_used`, `n_decoys_active`, `n_heavy_moved`, `avg_stamina`).

---

## 8. Trainer API (`trainers/`)

```python
# rollout.py
@flax.struct.dataclass
class Transition:
    done; action; value; reward; log_prob; obs; global_obs; avail; info

# mappo.py
def make_train(cfg: Config):
    # returns train(rng) -> {"runner_state":..., "metrics":...}, fully jittable.
    # Pipeline (all inside lax.scan, no host transfer):
    #   1. vmap env over num_envs; auto-reset on done.
    #   2. ScannedGRU actor produces actions from LOCAL masked obs (decentralized).
    #   3. CriticRNN consumes GLOBAL privileged obs (centralized training).
    #   4. collect Transition; compute GAE (gamma, gae_lambda).
    #   5. MAPPO loss = clipped policy + vf_coef*value + ent_coef*entropy(- );
    #      update_epochs * num_minibatches SGD steps with optax (clip+adam).
    #   6. parameter sharing per team (shared_policy_per_team).
```

`batchify(dict_of_(A,...)) -> (A*B, ...)` and `unbatchify` live in
`utils/pytree.py` and are the ONLY sanctioned way to flatten the agent axis for
the network and restore it afterward.

### Self-play / ELO (`selfplay.py`)
`OpponentPool` holds up to `opponent_pool_size` frozen parameter snapshots with
ELO ratings. Each update, with prob `past_opponent_prob`, the *seeker* (or
hider) opponent params for a fraction of envs are sampled from the pool weighted
by ELO; otherwise self-play against current params. `EloManager` (uses
`utils/elo.py`) updates ratings from episode win/loss outcomes. Snapshots are
pushed every `snapshot_interval_updates`.

---

## 9. Naming / style rules

- Comment every 2.0 feature block with a `# 2.0: <feature>` tag and the
  anti-surf fix with `# FIX: ...` so graders can grep them.
- Docstrings on every public function/module; type hints throughout.
- Keep functions pure & jit-friendly; isolate any Python-side control flow in the
  trainer's non-jitted setup, not in env/model forward passes.
- Prefer `chex.assert_shape` in tests, not in hot paths.
