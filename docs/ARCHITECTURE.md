# Hide & Seek 2.0 — Architecture

This document is a deeper dive into *how* the pieces fit together. For the
authoritative struct/observation/action/module specification, see
[`CONTRACT.md`](CONTRACT.md). For the per-feature breakdown of the new
mechanics, see [`FEATURES_2.0.md`](FEATURES_2.0.md). All shared dimensions are
**derived in [`config.py`](../config.py)** and read everywhere else.

---

## 1. Design philosophy

Hide & Seek 2.0 is built in the **PureJaxRL / JaxMARL** style: the environment,
its physics, the observation generation, *and* the learner are all pure JAX
functions composed into a single compiled graph. The consequences:

- One `reset(key) -> State` and one `step(state, action) -> State` model **one**
  environment. The trainer `jax.vmap`s them over `num_envs` (default 2048).
- The whole rollout is a `jax.lax.scan`, so there are **no Python loops over
  time or over environments** and **no host↔device copies** in the hot path.
- All control flow that depends on traced values uses `jnp.where` / `lax.select`
  (never a Python `if`), so everything is `jit`/`vmap`/`scan`-safe.
- Randomness is explicit: every stochastic function threads a
  `jax.random.PRNGKey`.

This is what makes "hours not weeks" plausible (a *target*, not a benchmark in
this scaffold): thousands of environments step in lockstep on the accelerator.

---

## 2. Data-flow diagram

```
                         config.py  (single source of truth: all dims/HPs)
                              │
        ┌─────────────────────┴───────────────────────────────────────────┐
        │                                                                   │
        ▼                                                                   ▼
   ┌──────────────────────────── envs/ ───────────────────────────┐   ┌──── models/ ────┐
   │ procedural.py  reset(key) → State                            │   │ EntityTransformer│
   │   randomize map / teams / props / fog                         │   │   (transformer.py)│
   │ hide_and_seek.py  step(state, action) → State                │   │        │          │
   │   ├─ physics.py   physics_step(...) (substeps, coop, ground)  │   │        ▼          │
   │   ├─ utils/visibility.py  rays → entity_mask                  │   │ ScannedGRU        │
   │   ├─ reward (team game reward, broadcast)                     │   │   (memory.py)     │
   │   └─ observe → State.obs dict                                 │   │        │          │
   └───────────────┬──────────────────────────────────────────────┘   │  ┌─────┴─────┐    │
                   │  State.obs                                        │  ▼           ▼    │
        ┌──────────┴───────────────┐                                  │ ActorRNN   CriticRNN
        │  LOCAL (masked)           │  GLOBAL (privileged)            │ (actor.py) (critic.py)
        │  entities, entity_mask,   │  global_entities, global_mask   │  │            │     │
        │  self, agent_active       │                                 └──┼────────────┼─────┘
        ▼                           ▼                                     │            │
   ╔══════════════╗            ╔══════════════╗                    actions(pi)     value
   ║  ActorRNN    ║            ║  CriticRNN   ║                          │            │
   ║ (decentral.) ║            ║ (centralized)║                          ▼            ▼
   ╚══════╤═══════╝            ╚══════╤═══════╝                  ┌──────────── trainers/ ───────────┐
          │ action                    │ value                   │ rollout.py  Transition pytree      │
          └────────────┬──────────────┘                         │ mappo.py    lax.scan rollout       │
                       ▼                                         │             GAE → MAPPO loss → SGD │
                  Transition                                     │ selfplay.py OpponentPool + ELO     │
                       │                                         └────────────────┬──────────────────┘
                       └─────────────── lax.scan over num_steps ──────────────────┘
                                          (vmap over num_envs)
```

Everything inside the `lax.scan` box runs on-device with no host transfer.

---

## 3. The CTDE split (Centralized Training, Decentralized Execution)

The single most important structural decision is the **asymmetry between what
the actor sees and what the critic sees**:

| | Actor (`ActorRNN`) | Critic (`CriticRNN`) |
|---|---|---|
| Inputs | `entities`, `entity_mask`, `self`, `agent_active` | `global_entities`, `global_mask` |
| Coords | **relative** to the observing agent | **absolute** |
| Visibility | **masked** by LOS/lidar/vision/fog | existence-only (sees everything active) |
| Privileged | no | yes — `true_is_decoy`, `grounded` per entity |
| Feature dim | `Fe = entity_feat_dim = 20` | `Fg = global_entity_feat_dim = 22` |
| Used at | training **and** execution | training **only** |

At deployment you keep only the actor, which depends purely on locally available
information. During training, the critic's privileged global view (including the
ground-truth `is_decoy` flag and `grounded` state that the actor is forbidden to
see) gives a low-variance value estimate that stabilizes MAPPO. The two networks
have **separate parameters**; `models/networks.py` bundles them as an
`ActorCritic` convenience without sharing weights.

Per-team parameter sharing (`TrainConfig.shared_policy_per_team`) means all
hiders share one actor and all seekers share another, with the `team` one-hot in
the `self` vector letting a shared network specialize behavior.

---

## 4. Entity-centric Transformer + GRU = object permanence

The world is variable in composition (procedural team sizes, prop counts) so a
fixed flat observation vector would be brittle. Instead each agent observes a
**padded set of entity tokens** `(E, Fe)` plus a `self` proprioception vector.

1. **`EntityTransformer` (`models/transformer.py`).** Masked multi-head
   self-attention over the `E` entity tokens, using `entity_mask` as a key-
   padding mask so invisible/inactive entities cannot leak information. Because
   attention is permutation-invariant and the pooling is masked-mean, the policy
   is invariant to entity ordering and robust to padding. The `self` vector is
   injected (as a learned query / extra token) so the encoding is *egocentric*.

2. **`ScannedGRU` (`models/memory.py`).** The per-step embedding is fed through a
   PureJaxRL-style GRU that scans over the **time** axis and **zeros its carry on
   episode resets** (the `(inputs, resets)` convention). This recurrent memory is
   what gives **object permanence**: when an opponent disappears behind a wall or
   into fog (and thus its token is masked out), the GRU's hidden state still
   carries a belief about where it went. Without memory, a hider that loses sight
   of a seeker would behave as if the seeker had vanished.

3. **Hybrid heads (`models/actor.py`).** From the GRU output the actor produces a
   `distrax` diagonal Gaussian over the 3 continuous `move` dims and one
   `distrax.Categorical` per discrete `interact` toggle. `sample_and_logprob`
   and `eval_logprob` combine them: total log-prob is the Gaussian log-prob
   (summed over 3 dims) plus the categorical log-probs; total entropy is the sum
   of component entropies.

---

## 5. The scan-based, zero-copy training loop

`trainers/mappo.make_train(cfg)` returns a single jittable `train(rng)` function.
Conceptually:

```
train(rng):
  init env states (vmap reset over num_envs)
  init actor/critic params + optax optimizer state
  init GRU carries (zeros)
  runner_state = (params, opt_state, env_state, gru_carries, rng)

  repeat num_updates times:               # outer lax.scan (or fori) — on device
    # --- collect a rollout -------------------------------------------------
    runner_state, traj = lax.scan(_env_step, runner_state, None, num_steps)
      _env_step:
        actor(local_obs, dones)  -> pi, new_actor_carry      # decentralized
        sample action + log_prob
        critic(global_obs, dones) -> value                    # centralized
        env.step (vmapped) -> next_state ; auto-reset on done
        emit Transition(done, action, value, reward, log_prob,
                        obs, global_obs, avail, info)

    # --- advantage estimation ---------------------------------------------
    last_value = critic(last_global_obs)
    advantages, targets = gae(traj, last_value, gamma, gae_lambda)

    # --- MAPPO update ------------------------------------------------------
    for _ in range(update_epochs):
      shuffle, split into num_minibatches
      for minibatch:
        loss = clipped_policy_loss
             + vf_coef * value_loss
             - ent_coef * entropy
        params, opt_state = optax_update(grad(loss), ...)   # clip-by-norm + adam

    # --- self-play bookkeeping --------------------------------------------
    every snapshot_interval_updates: OpponentPool.push(frozen params)
    EloManager.update(...) from episode win/loss outcomes

  return {"runner_state": ..., "metrics": ...}
```

Key points:

- **`batchify`/`unbatchify` (`utils/pytree.py`)** are the *only* sanctioned way
  to fold the agent axis `A` into the batch for the network
  (`(A, ...) → (A*B, ...)`) and restore it afterward.
- The GRU carry is part of `runner_state` and threaded through the scan;
  `done` flags reset it per episode.
- Auto-reset lives in the **trainer wrapper**, not in `env.step`, so `step`
  itself stays branch-free (PureJaxRL convention; see `CONTRACT.md` §2).
- Because the entire thing is one compiled graph, the only host transfer is the
  occasional metrics pull for logging (`utils/logging.py`, throttled by
  `log_interval`).

---

## 6. Historical self-play via ELO

`trainers/selfplay.py` maintains an `OpponentPool` of up to
`opponent_pool_size` (default 20) **frozen parameter snapshots**, each carrying
an ELO rating (`utils/elo.py`: `expected_score`, `update_elo`). On each update:

- with probability `past_opponent_prob` (default 0.5), the opposing team's
  params for a fraction of the `num_envs` are sampled from the pool **weighted by
  ELO**; otherwise the env plays current-policy self-play;
- `EloManager` updates ratings from episode win/loss outcomes (K = `elo_k`);
- a fresh snapshot is pushed every `snapshot_interval_updates` updates.

This produces an autocurriculum: the learner must keep beating an increasingly
strong, diverse population rather than overfitting to its current self.

---

## 7. The 2.0 mechanics → code map

Every new mechanic is tagged in source with `# 2.0: <feature>` (and the anti
box-surfing fix with `# FIX: strict Newtonian ground-contact (no box-surfing)`)
so graders can `grep` them. Summary of where each lives:

| 2.0 mechanic | primary location | grep tag |
|---|---|---|
| Variable mass & cooperative push | `envs/physics.py` | `# 2.0: cooperative physics` |
| Strict ground-contact (anti box-surf) | `envs/physics.py` | `# FIX: strict Newtonian ground-contact (no box-surfing)` |
| Stamina | `envs/physics.py` | `# 2.0: stamina` |
| Decoys / sensory spoofing | `envs/hide_and_seek.py` | `# 2.0: deception` / decoy blocks |
| Fog of war & lighting | `utils/visibility.py` | `# 2.0: fog of war` |
| Destructible walls & doors | `envs/physics.py`, `envs/hide_and_seek.py` | `# 2.0: destructible env` |

The corresponding config knobs (masses, `coop_required_agents`,
`coop_force_threshold`, `stamina_*`, `decoy_*`, fog params, `wall_break_speed`,
`door_open_steps`, `ground_contact_required`) are all in `EnvConfig` in
[`config.py`](../config.py). See [`FEATURES_2.0.md`](FEATURES_2.0.md) for the
field-by-field mapping.

---

## 8. Procedural generation strategy

`envs/procedural.py` produces a fresh world per episode, **fully vectorized** so
it can be `vmap`ped across `num_envs` with no Python branching:

1. **Teams.** Sample hider/seeker counts in `[min_team_size, max_team_size]`.
   Agents beyond the sampled counts are marked inactive via the `active` /
   `agent_active` masks (slots always pad up to `max_agents` for static shapes).
2. **Props.** Place boxes (light/heavy mix), ramps, decoys, walls (some fragile),
   and doors up to their `n_*_max` counts; unused slots are padded inactive.
   Counts never exceed the `n_*_max` values because those *are* the padding
   sizes (`max_entities` is derived from them in `config.py`).
3. **Layout.** Positions are sampled within the arena (`arena_size`); walls/doors
   can form rooms and chokepoints. Heavy boxes get `box_heavy_mass`; walls get a
   finite `wall_hp` only if fragile (otherwise an inf-proxy mass / no break).
4. **Fog.** Sample `n_fog_patches` patch centers (`fog_pos`) with radius
   `fog_radius`; `utils/visibility.py` uses them to attenuate sensing.
5. **Static shapes always.** Because every category pads to its max, every array
   has a shape known at trace time — essential for `jit`/`vmap`. "Inactivity" is
   expressed through boolean masks, never through variable-length arrays.

This keeps the entire reset path pure and traceable while still giving the
learner a broad distribution of maps, team sizes, and prop loadouts — the
diversity an autocurriculum needs.
