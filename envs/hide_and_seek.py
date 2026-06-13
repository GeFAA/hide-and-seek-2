"""
envs/hide_and_seek.py -- The **Hide & Seek 2.0** environment.

:class:`HideAndSeekEnv` exposes the Brax-style functional API mandated by
CONTRACT.md: pure ``reset(key) -> State`` and ``step(state, action) -> State``
for a **single** environment (the trainer ``vmap``s over ``num_envs``). Every
method is branch-free on traced values (``jnp.where`` / ``lax.select`` only) and
contains no host transfers, so the whole thing inlines into ``jax.jit`` /
``jax.lax.scan``.

This module owns the *game logic* on top of :mod:`envs.physics`:

* Observation building (CONTRACT §3): local masked actor view + privileged
  global critic view + decoy spoofing.
* Reward / phase logic (CONTRACT §7): prep vs main, hider-seen game reward.
* The interactable 2.0 mechanics driven by the discrete action channel:
  grab / lock / decoy activation, destructible walls, doors, and the kinematic
  2.5D elevation (climbing) update that feeds the anti box-surfing gate.

Visibility (LOS / lidar / vision cone / fog attenuation) is delegated to
:mod:`utils.visibility`, which is built in parallel and referenced by its
contract names:

* ``compute_visibility_batch(physics, game, observer_ids, cfg) -> (A, E)``
  returning boolean visibility for each observing agent (vmapped over observers).
* ``lidar_scan(physics, game, observer_ids, cfg) -> (A, lidar_n_rays)`` ranges.

We import them lazily inside methods so this module stays importable even before
``utils`` lands (a clearly-documented soft dependency).
"""
from __future__ import annotations

from typing import Dict

import jax
import jax.numpy as jnp

from config import Config, EnvConfig, ModelConfig, N_ENTITY_TYPES, TYPE_TO_ID
from envs.physics import physics_step
from envs.procedural import entity_slot_ranges, generate_episode
from envs.state import GameState, PhysicsState, State

__all__ = ["HideAndSeekEnv"]

# Resolved type ids (host-side constants).
_ID_HIDER = TYPE_TO_ID["hider"]
_ID_SEEKER = TYPE_TO_ID["seeker"]
_ID_DECOY = TYPE_TO_ID["decoy"]
_ID_WALL = TYPE_TO_ID["wall"]
_ID_DOOR = TYPE_TO_ID["door"]
_ID_BOX_LIGHT = TYPE_TO_ID["box_light"]
_ID_BOX_HEAVY = TYPE_TO_ID["box_heavy"]
_ID_RAMP = TYPE_TO_ID["ramp"]

# Discrete interact channel indices (CONTRACT §4 / ModelConfig.action_discrete_nvec).
_ACT_GRAB = 0
_ACT_LOCK = 1
_ACT_DECOY = 2

_EPS = 1e-8


class HideAndSeekEnv:
    """Functional multi-agent Hide & Seek 2.0 environment (single env).

    Parameters
    ----------
    cfg:
        Top-level :class:`config.Config`. Both ``cfg.env`` (:class:`EnvConfig`)
        and ``cfg.model`` (:class:`ModelConfig`, for feature dims) are used.
    """

    def __init__(self, cfg: Config) -> None:
        self.cfg: Config = cfg
        self.env: EnvConfig = cfg.env
        self.model: ModelConfig = cfg.model
        self.A: int = cfg.env.max_agents
        self.E: int = cfg.env.max_entities
        self.ranges = entity_slot_ranges(cfg.env)

    # ===================================================================== #
    # Reset
    # ===================================================================== #
    def reset(self, key: jnp.ndarray) -> State:
        """Generate a fresh episode and return the initial :class:`State`.

        Parameters
        ----------
        key:
            ``jax.random.PRNGKey``.

        Returns
        -------
        state:
            Initial environment state with observations populated and zero
            reward.
        """
        key, gen_key = jax.random.split(key)
        physics, game = generate_episode(gen_key, self.env)
        obs = self.observe(physics, game)
        A = self.A
        info = self._zero_info()
        return State(
            physics=physics,
            game=game,
            obs=obs,
            reward=jnp.zeros((A,), dtype=jnp.float32),
            done=jnp.array(False),
            info=info,
            key=key,
        )

    # ===================================================================== #
    # Step
    # ===================================================================== #
    def step(self, state: State, action: Dict[str, jnp.ndarray]) -> State:
        """Advance the environment one control step (branch-free).

        No auto-reset on ``done`` here -- the trainer wrapper handles that, per
        PureJaxRL convention, so ``step`` stays a pure transition function.

        Parameters
        ----------
        state:
            Current :class:`State`.
        action:
            Action dict per CONTRACT §4:
            ``{"move": (A, 3) float in [-1,1], "interact": (A, 3) int}``.

        Returns
        -------
        state:
            Next :class:`State`.
        """
        cfg = self.env
        A, E = self.A, self.E
        physics, game = state.physics, state.game
        agent_active = physics.active[:A]

        move = action["move"]            # (A, 3) in [-1, 1]
        interact = action["interact"]    # (A, 3) int categoricals
        grab_a = interact[:, _ACT_GRAB].astype(jnp.int32)
        lock_a = interact[:, _ACT_LOCK].astype(jnp.int32)
        decoy_a = interact[:, _ACT_DECOY].astype(jnp.int32)

        # Decode continuous action into arena forces. Inactive agents -> zero.
        # We interpret the (unit-bounded) move[:, :2] as a sprint-or-walk command
        # where a large magnitude implies sprint intent; sprint is the explicit
        # signal here derived from command magnitude exceeding
        # cfg.sprint_cmd_threshold (schematic -- the trainer may instead pass a
        # dedicated channel, but the contract action dict has no sprint field, so
        # we infer it).
        fxy = move[:, :2] * cfg.agent_max_force
        torque = move[:, 2] * cfg.agent_max_torque
        agent_forces = jnp.concatenate([fxy, torque[:, None]], axis=-1)  # (A,3)
        agent_forces = jnp.where(agent_active[:, None], agent_forces, 0.0)

        # 2.0: stamina -- infer sprint intent from command magnitude.
        cmd_mag = jnp.linalg.norm(move[:, :2], axis=-1)
        sprint = (cmd_mag > cfg.sprint_cmd_threshold) & agent_active

        # --- Interactions BEFORE physics (grab/lock alter who pushes what). ---
        game = self._apply_grab(physics, game, grab_a, agent_active)
        game = self._apply_lock(physics, game, lock_a, agent_active)
        game = self._apply_decoy(physics, game, decoy_a, agent_active)

        # --- Physics step. ---
        physics, contact = physics_step(physics, game, agent_forces, sprint, cfg)
        game = game.replace(stamina=contact["new_stamina"])

        # --- Kinematic 2.5D elevation update (climbing) -> feeds anti-surf gate.
        physics = self._update_elevation(physics, contact["agent_on_entity"])

        # --- Destructible walls & doors. ---
        physics, game = self._update_walls(physics, game, contact["impact_speed"])
        physics, game = self._update_doors(physics, game)

        # --- Decoy timers tick down; expired decoys stop emitting. ---
        game = self._tick_decoys(game)

        # --- Phase / step advance. ---
        new_step = game.step + 1
        phase = jnp.where(new_step >= cfg.prep_steps, 1, 0).astype(jnp.int32)
        game = game.replace(step=new_step, phase=phase)

        # --- Visibility, reward, observation, done. ---
        vis_mask, seeker_sees_hider = self._true_visibility(physics, game)
        reward, info = self._reward(physics, game, seeker_sees_hider, contact)
        obs = self.observe(physics, game, vis_mask=vis_mask)
        done = new_step >= cfg.max_steps

        return State(
            physics=physics,
            game=game,
            obs=obs,
            reward=reward,
            done=done,
            info=info,
            key=state.key,
        )

    # ===================================================================== #
    # Interactions
    # ===================================================================== #
    def _nearest_grabbable(
        self, physics: PhysicsState, agent_active: jnp.ndarray
    ) -> jnp.ndarray:
        """Return, per agent, the id of the nearest grabbable entity in reach.

        Grabbable = boxes / ramps / decoys that are active and within
        ``agent_radius + size + reach``. Returns ``-1`` if none.

        Parameters
        ----------
        physics:
            Current physics.
        agent_active:
            ``(A,)`` bool active agent mask.

        Returns
        -------
        nearest:
            ``(A,)`` int32 entity ids (or ``-1``).
        """
        A, E = self.A, self.E
        type_id = physics.type_id
        grabbable = (
            (type_id == _ID_BOX_LIGHT)
            | (type_id == _ID_BOX_HEAVY)
            | (type_id == _ID_RAMP)
            | (type_id == _ID_DECOY)
        ) & physics.active                                  # (E,)
        reach = self.env.agent_radius + 0.6
        agent_pos = physics.pos[:A, :2]
        ent_pos = physics.pos[:, :2]
        d = jnp.linalg.norm(agent_pos[:, None, :] - ent_pos[None, :, :], axis=-1)  # (A,E)
        in_reach = (d < (reach + physics.size[None, :])) & grabbable[None, :]
        in_reach = in_reach & agent_active[:, None]
        big = jnp.where(in_reach, d, jnp.inf)
        nearest = jnp.argmin(big, axis=1).astype(jnp.int32)
        return jnp.where(jnp.any(in_reach, axis=1), nearest, -1)

    def _apply_grab(
        self,
        physics: PhysicsState,
        game: GameState,
        grab_a: jnp.ndarray,
        agent_active: jnp.ndarray,
    ) -> GameState:
        """Toggle grab: grab nearest grabbable if requested, else release.

        ``grab_a == 1`` => grab (or keep) the nearest grabbable; ``grab_a == 0``
        => release. Branch-free via :func:`jnp.where`.

        Parameters
        ----------
        physics, game:
            Current state.
        grab_a:
            ``(A,)`` int32 grab toggle.
        agent_active:
            ``(A,)`` bool.

        Returns
        -------
        game:
            Updated game state (``holding`` modified).
        """
        nearest = self._nearest_grabbable(physics, agent_active)  # (A,)
        want_grab = (grab_a == 1) & agent_active
        # If already holding something, keep it; otherwise pick up nearest.
        currently = game.holding
        new_hold = jnp.where(
            want_grab,
            jnp.where(currently >= 0, currently, nearest),
            -1,
        )
        # Cannot hold a locked object: drop if target is locked.
        held_idx = jnp.clip(new_hold, 0, self.E - 1)
        target_locked = game.locked[held_idx] & (new_hold >= 0)
        new_hold = jnp.where(target_locked, -1, new_hold)
        return game.replace(holding=new_hold.astype(jnp.int32))

    def _apply_lock(
        self,
        physics: PhysicsState,
        game: GameState,
        lock_a: jnp.ndarray,
        agent_active: jnp.ndarray,
    ) -> GameState:
        """Toggle lock on the object an agent is holding / standing next to.

        ``lock_a == 1`` locks the agent's held object (immovable; tagged with the
        agent's team). ``lock_a == 0`` unlocks it but **only** if the agent's team
        matches ``locked_by`` (same-team-only unlock). Branch-free.

        Parameters
        ----------
        physics, game:
            Current state.
        lock_a:
            ``(A,)`` int32 lock toggle.
        agent_active:
            ``(A,)`` bool.

        Returns
        -------
        game:
            Updated game state (``locked`` / ``locked_by`` modified).
        """
        A, E = self.A, self.E
        holding = game.holding                              # (A,)
        team = game.team                                    # (A,)
        held_idx = jnp.clip(holding, 0, E - 1)
        valid = (holding >= 0) & agent_active

        # Scatter lock requests onto entities. Use max-reduction over agents so
        # any agent locking wins; encode "lock" as 1 and remember the team.
        lock_req = (lock_a == 1) & valid                    # (A,)
        unlock_req = (lock_a == 0) & valid                  # (A,)

        # Build per-entity lock set: an entity becomes locked if any holder locks.
        ent_lock = jnp.zeros((E,), dtype=bool)
        ent_lock = ent_lock.at[held_idx].max(lock_req)
        # Team that locked it (last writer among lockers).
        ent_team = jnp.full((E,), -1, dtype=jnp.int32)
        scatter_team = jnp.where(lock_req, team, -1)
        ent_team = ent_team.at[held_idx].max(scatter_team)

        # Unlock requests: only honored if requester team == locked_by team.
        same_team_unlock = unlock_req & (game.locked_by[held_idx] == team)
        ent_unlock = jnp.zeros((E,), dtype=bool)
        ent_unlock = ent_unlock.at[held_idx].max(same_team_unlock)

        new_locked = jnp.where(ent_lock, True, game.locked)
        new_locked = jnp.where(ent_unlock, False, new_locked)
        new_locked_by = jnp.where(ent_lock, ent_team, game.locked_by)
        new_locked_by = jnp.where(ent_unlock, -1, new_locked_by)
        return game.replace(locked=new_locked, locked_by=new_locked_by.astype(jnp.int32))

    def _apply_decoy(
        self,
        physics: PhysicsState,
        game: GameState,
        decoy_a: jnp.ndarray,
        agent_active: jnp.ndarray,
    ) -> GameState:
        """Activate decoys near agents that request it.

        # 2.0: deception
        ``decoy_a == 1`` activates the nearest inactive decoy in reach, setting
        its ``decoy_timer`` to ``decoy_active_steps`` and making it broadcast
        spoofed ``emitted_noise``. The *spoofed local appearance* (fake type +
        noise) is applied later in :meth:`observe`; here we only flip the timer /
        noise channels. Branch-free.

        Parameters
        ----------
        physics, game:
            Current state.
        decoy_a:
            ``(A,)`` int32 decoy toggle.
        agent_active:
            ``(A,)`` bool.

        Returns
        -------
        game:
            Updated game state (``decoy_timer`` / ``emitted_noise`` modified).
        """
        A, E = self.A, self.E
        dec_s, dec_e = self.ranges["decoys"]
        is_decoy = game.is_decoy & physics.active           # (E,)
        # An agent activates a decoy it is near and that is currently idle.
        agent_pos = physics.pos[:A, :2]
        ent_pos = physics.pos[:, :2]
        d = jnp.linalg.norm(agent_pos[:, None, :] - ent_pos[None, :, :], axis=-1)  # (A,E)
        near = d < self.env.decoy_noise_radius
        idle = game.decoy_timer <= 0
        can_activate = near & is_decoy[None, :] & idle[None, :]
        can_activate = can_activate & ((decoy_a == 1) & agent_active)[:, None]
        # Any agent activating a decoy flips it on.
        activate = jnp.any(can_activate, axis=0)            # (E,)

        new_timer = jnp.where(
            activate, self.env.decoy_active_steps, game.decoy_timer
        ).astype(jnp.int32)
        # Emitted noise = max intensity while active (decays in _tick_decoys).
        new_noise = jnp.where(activate, 1.0, game.emitted_noise)
        return game.replace(decoy_timer=new_timer, emitted_noise=new_noise)

    def _tick_decoys(self, game: GameState) -> GameState:
        """Decrement decoy timers; zero emitted noise when a decoy expires.

        Parameters
        ----------
        game:
            Current game state.

        Returns
        -------
        game:
            Updated game state.
        """
        active_now = game.decoy_timer > 0
        new_timer = jnp.maximum(game.decoy_timer - 1, 0)
        still_active = new_timer > 0
        # Noise fades to zero on the step a decoy expires, but ONLY for decoy
        # slots: non-decoy entities keep whatever emitted_noise they carry (this
        # tick must not clobber non-decoy noise channels).
        decoy_noise = jnp.where(active_now & still_active, game.emitted_noise, 0.0)
        new_noise = jnp.where(game.is_decoy, decoy_noise, game.emitted_noise)
        return game.replace(decoy_timer=new_timer.astype(jnp.int32), emitted_noise=new_noise)

    # ===================================================================== #
    # Destructible walls & doors
    # ===================================================================== #
    def _update_walls(
        self,
        physics: PhysicsState,
        game: GameState,
        impact_speed: jnp.ndarray,
    ) -> tuple[PhysicsState, GameState]:
        """Damage / break destructible walls hit above ``wall_break_speed``.

        # 2.0: destructible environment
        A wall takes damage proportional to the excess of the ram impact speed
        over ``wall_break_speed``. When ``wall_hp <= 0`` the wall is deactivated
        (``active = False``) so it no longer blocks movement or vision.

        Parameters
        ----------
        physics, game:
            Current state.
        impact_speed:
            ``(E,)`` max approach speed onto each entity (from physics).

        Returns
        -------
        physics, game:
            Updated states.
        """
        is_wall = physics.type_id == _ID_WALL
        over = jnp.maximum(impact_speed - self.env.wall_break_speed, 0.0)
        # Damage only fragile walls actually rammed hard; scale to make a single
        # high-speed ram meaningful.
        damage = jnp.where(is_wall, over * 25.0, 0.0)
        new_hp = jnp.where(is_wall, game.wall_hp - damage, game.wall_hp)
        broken = is_wall & (new_hp <= 0.0)
        new_active = jnp.where(broken, False, physics.active)
        return (
            physics.replace(active=new_active),
            game.replace(wall_hp=new_hp.astype(jnp.float32)),
        )

    def _update_doors(
        self, physics: PhysicsState, game: GameState
    ) -> tuple[PhysicsState, GameState]:
        """Accumulate door-open progress under sustained agent contact and open.

        # 2.0: interactable environment
        A door advances ``door_progress`` toward 1 while any agent stands in
        contact; once it reaches the open threshold (progress >= 1) the door is
        deactivated as a collider (``active = False``) so it stops blocking
        movement and vision (the chokepoint opens). Progress is normalized so
        ``door_open_steps`` of contact fully opens it.

        Parameters
        ----------
        physics, game:
            Current state.

        Returns
        -------
        physics, game:
            Updated states (``physics.active`` flips False on opened doors;
            ``game.door_progress`` advanced).
        """
        A = self.A
        is_door = (physics.type_id == _ID_DOOR) & physics.active   # (E,)
        agent_pos = physics.pos[:A, :2]
        ent_pos = physics.pos[:, :2]
        d = jnp.linalg.norm(agent_pos[:, None, :] - ent_pos[None, :, :], axis=-1)  # (A,E)
        contact = (d < (physics.size[None, :] + self.env.agent_radius + 0.2))
        any_contact = jnp.any(contact & is_door[None, :], axis=0)  # (E,)
        inc = jnp.where(any_contact, 1.0 / float(self.env.door_open_steps), 0.0)
        new_prog = jnp.clip(game.door_progress + inc, 0.0, 1.0)
        # A fully-progressed door OPENS: deactivate it so it no longer occludes
        # or collides. Only doors can be opened this way (gate on is_door).
        opened = is_door & (new_prog >= 1.0)                       # (E,)
        new_active = jnp.where(opened, False, physics.active)
        return (
            physics.replace(active=new_active),
            game.replace(door_progress=new_prog.astype(jnp.float32)),
        )

    # ===================================================================== #
    # 2.5D elevation (climbing) -- feeds the anti box-surfing ground gate.
    # ===================================================================== #
    def _update_elevation(
        self, physics: PhysicsState, agent_on_entity: jnp.ndarray
    ) -> PhysicsState:
        """Set agent elevation ``z`` kinematically when standing on a prop.

        Schematic 2.5D climbing model: an agent overlapping a ramp/box is lifted
        to that prop's "surface height" (proportional to its size); otherwise it
        descends to ``z = 0``. This drives ``grounded`` so a climbed agent loses
        its locomotion drive (anti box-surfing). Not a true 3D contact model --
        documented simplification.

        Parameters
        ----------
        physics:
            Current physics (post-step).
        agent_on_entity:
            ``(A,)`` int32 entity id each agent stands on (-1 none).

        Returns
        -------
        physics:
            Physics with updated ``pos[:A, 2]`` and ``grounded``.
        """
        A, E = self.A, self.E
        on = agent_on_entity                                # (A,)
        on_idx = jnp.clip(on, 0, E - 1)
        surf_h = physics.size[on_idx]                       # surface height proxy
        z = jnp.where(on >= 0, surf_h, 0.0)                 # (A,)
        new_pos = physics.pos.at[:A, 2].set(z)
        # grounded recomputed: agent grounded iff z ~ 0.
        new_grounded = physics.grounded.at[:A].set(z <= 1e-4)
        return physics.replace(pos=new_pos, grounded=new_grounded)

    # ===================================================================== #
    # Visibility (delegates to utils.visibility)
    # ===================================================================== #
    def _true_visibility(self, physics: PhysicsState, game: GameState):
        """Compute the per-agent visibility mask and the seeker->hider flag.

        Uses :func:`utils.visibility.compute_visibility_batch` for the LOS /
        vision-cone / fog-attenuated mask (CONTRACT §3 ``entity_mask``). From it we
        also derive the reward signal ``seeker_sees_hider`` (does *any* seeker see
        *any* hider), used by :meth:`_reward`.

        Parameters
        ----------
        physics, game:
            Current state.

        Returns
        -------
        vis_mask:
            ``(A, E)`` bool per-agent visibility mask.
        seeker_sees_hider:
            ``()`` bool -- any active seeker sees any active hider.
        """
        from utils.visibility import compute_visibility_batch  # soft dep (built in parallel)

        # Agents-first layout => agent row index == entity id (0..A-1).
        observer_ids = jnp.arange(self.A, dtype=jnp.int32)
        # Contract (P3): compute_visibility_batch returns (A, E) bool visibility.
        vis_mask = compute_visibility_batch(physics, game, observer_ids, self.env)

        type_id = physics.type_id
        is_hider = (type_id == _ID_HIDER) & physics.active          # (E,)
        agent_team = game.team                                       # (A,)
        is_seeker_agent = (agent_team == 1) & physics.active[: self.A]  # (A,)

        # seeker_sees_hider: any seeker row that sees any hider column.
        seeker_rows = vis_mask & is_seeker_agent[:, None]           # (A,E)
        sees_hider = seeker_rows & is_hider[None, :]
        seeker_sees_hider = jnp.any(sees_hider)
        return vis_mask, seeker_sees_hider

    # ===================================================================== #
    # Reward & phase (CONTRACT §7)
    # ===================================================================== #
    def _reward(
        self,
        physics: PhysicsState,
        game: GameState,
        seeker_sees_hider: jnp.ndarray,
        contact: Dict[str, jnp.ndarray],
    ):
        """Compute per-agent team reward + info metrics (CONTRACT §7).

        * **Prep** (``phase == 0``): reward 0 for both teams.
        * **Main** (``phase == 1``): if no hider is seen by any seeker, hiders get
          ``+reward_scale`` and seekers ``-reward_scale``; if a hider *is* seen,
          the signs flip.

        Parameters
        ----------
        physics, game:
            Current state.
        seeker_sees_hider:
            ``()`` bool from :meth:`_true_visibility`.
        contact:
            Physics diagnostics (for 2.0 metrics).

        Returns
        -------
        reward:
            ``(A,)`` float32 per-agent reward.
        info:
            Metrics dict.
        """
        A = self.A
        team = game.team                                    # (A,)
        agent_active = physics.active[:A]
        is_main = (game.phase == 1)

        # Hider reward in main phase: +1 if NOT seen, else -1.
        hider_r = jnp.where(seeker_sees_hider, -1.0, 1.0) * self.env.reward_scale
        seeker_r = -hider_r
        # Zero during prep.
        hider_r = jnp.where(is_main, hider_r, 0.0)
        seeker_r = jnp.where(is_main, seeker_r, 0.0)

        is_hider = team == 0
        is_seeker = team == 1
        reward = jnp.where(is_hider, hider_r, 0.0) + jnp.where(is_seeker, seeker_r, 0.0)
        reward = jnp.where(agent_active, reward, 0.0).astype(jnp.float32)

        # ---- info metrics ----
        type_id = physics.type_id
        n_decoys_active = jnp.sum((game.decoy_timer > 0).astype(jnp.int32))
        n_heavy_moved = jnp.sum(contact["heavy_moved"].astype(jnp.int32))
        # ramps "used": any agent elevated on a ramp (z>0 while over a ramp).
        elevated = physics.pos[:A, 2] > 1e-3
        n_ramps_used = jnp.sum(elevated.astype(jnp.int32))
        avg_stamina = jnp.sum(jnp.where(agent_active, game.stamina, 0.0)) / jnp.maximum(
            jnp.sum(agent_active.astype(jnp.float32)), 1.0
        )
        info = {
            "hiders_reward": hider_r,
            "seekers_reward": seeker_r,
            "seen_frac": seeker_sees_hider.astype(jnp.float32),
            "n_ramps_used": n_ramps_used,
            "n_decoys_active": n_decoys_active,
            "n_heavy_moved": n_heavy_moved,
            "avg_stamina": avg_stamina,
        }
        return reward, info

    def _zero_info(self) -> Dict[str, jnp.ndarray]:
        """Return a zeroed info dict matching :meth:`_reward`'s schema."""
        return {
            "hiders_reward": jnp.array(0.0, dtype=jnp.float32),
            "seekers_reward": jnp.array(0.0, dtype=jnp.float32),
            "seen_frac": jnp.array(0.0, dtype=jnp.float32),
            "n_ramps_used": jnp.array(0, dtype=jnp.int32),
            "n_decoys_active": jnp.array(0, dtype=jnp.int32),
            "n_heavy_moved": jnp.array(0, dtype=jnp.int32),
            "avg_stamina": jnp.array(0.0, dtype=jnp.float32),
        }

    # ===================================================================== #
    # Observation (CONTRACT §3)
    # ===================================================================== #
    def observe(
        self,
        physics: PhysicsState,
        game: GameState,
        vis_mask: jnp.ndarray | None = None,
    ) -> Dict[str, jnp.ndarray]:
        """Build the observation dict per CONTRACT §3.

        Produces:

        * ``entities`` ``(A, E, Fe)`` -- per-agent **local** entity tokens in
          relative coords, with decoy spoofing applied (a decoy looks like what
          it mimics + carries spoofed noise).
        * ``entity_mask`` ``(A, E)`` bool -- ``active`` AND visible.
        * ``self`` ``(A, Fs)`` -- proprioception.
        * ``global_entities`` ``(E, Fg)`` -- **absolute** tokens + privileged
          ``true_is_decoy`` and ``grounded`` extras (critic only).
        * ``global_mask`` ``(E,)`` bool -- existence (``active``) only.
        * ``agent_active`` ``(A,)`` bool.

        Parameters
        ----------
        physics, game:
            Current state.
        vis_mask:
            Optional precomputed ``(A, E)`` visibility mask (from
            :meth:`_true_visibility`). If ``None`` it is computed here.

        Returns
        -------
        obs:
            The observation dict.
        """
        A, E = self.A, self.E
        T = N_ENTITY_TYPES
        Fe = self.model.entity_feat_dim
        Fg = self.model.global_entity_feat_dim
        Fs = self.model.self_feat_dim
        arena = self.env.arena_size

        if vis_mask is None:
            vis_mask, _ = self._true_visibility(physics, game)

        active = physics.active                              # (E,)
        agent_active = active[:A]                            # (A,)

        # --- absolute per-entity features shared by local & global views. ---
        pos = physics.pos                                   # (E, 3)
        vel = physics.vel                                   # (E, 3)
        mass_n = (physics.mass / self.env.box_heavy_mass)[:, None]  # (E,1) normalized
        size_col = physics.size[:, None]                    # (E,1)
        locked_col = game.locked.astype(jnp.float32)[:, None]
        is_held = self._is_held_mask(game)[:, None]         # (E,1)

        # True one-hot type.
        true_type = physics.type_id                         # (E,)
        true_onehot = jax.nn.one_hot(jnp.clip(true_type, 0, T - 1), T)  # (E,T)

        # 2.0: deception -- spoofed local appearance for active decoys.
        # An active decoy mimics a HIDER and broadcasts spoofed noise; only the
        # critic's global view sees the true decoy flag.
        decoy_on = (game.decoy_timer > 0) & game.is_decoy & active   # (E,)
        spoof_onehot = jax.nn.one_hot(jnp.full((E,), _ID_HIDER), T)   # mimic hider
        local_onehot = jnp.where(decoy_on[:, None], spoof_onehot, true_onehot)
        # Spoofed noise the entity APPEARS to emit (decoys spoof high; others true).
        spoofed_noise = jnp.where(decoy_on, 1.0, game.emitted_noise)[:, None]
        true_noise = game.emitted_noise[:, None]

        # ----------------------------------------------------------------- #
        # LOCAL (actor) view: relative coords per observing agent.
        # entities[a, e] = features of entity e seen from agent a.
        # Layout (CONTRACT §3.1): rel_pos(3) rel_vel(3) dist(1) mass(1)
        #   type_onehot(T) locked(1) emitted_noise(1) is_held(1) size(1).
        # ----------------------------------------------------------------- #
        obs_pos = pos[:A]                                   # (A, 3)
        obs_vel = vel[:A]                                   # (A, 3)
        rel_pos = pos[None, :, :] - obs_pos[:, None, :]     # (A, E, 3)
        rel_vel = vel[None, :, :] - obs_vel[:, None, :]     # (A, E, 3)
        dist = jnp.linalg.norm(rel_pos, axis=-1, keepdims=True) / arena  # (A,E,1)

        # Broadcast per-entity scalar features to (A, E, ...).
        AE = (A, E)
        mass_b = jnp.broadcast_to(mass_n[None], (*AE, 1))
        size_b = jnp.broadcast_to(size_col[None], (*AE, 1))
        locked_b = jnp.broadcast_to(locked_col[None], (*AE, 1))
        held_b = jnp.broadcast_to(is_held[None], (*AE, 1))
        onehot_b = jnp.broadcast_to(local_onehot[None], (*AE, T))
        noise_b = jnp.broadcast_to(spoofed_noise[None], (*AE, 1))

        local = jnp.concatenate(
            [rel_pos, rel_vel, dist, mass_b, onehot_b, locked_b, noise_b, held_b, size_b],
            axis=-1,
        )  # (A, E, Fe)

        # Visibility mask: active AND visible. Zero out invisible entity rows so
        # the actor literally cannot read their relative coords (CONTRACT §3.1).
        entity_mask = vis_mask & active[None, :]            # (A, E)
        local = local * entity_mask[..., None].astype(jnp.float32)

        # ----------------------------------------------------------------- #
        # GLOBAL (critic) view: ABSOLUTE coords, unmasked, + privileged extras.
        # global_entities[e] = abs features + true_is_decoy(1) + grounded(1).
        # ----------------------------------------------------------------- #
        # In the absolute (critic) frame there is no single observer, so the
        # "dist" slot carries distance-from-arena-origin (a stable global feature
        # occupying the same index as the local observer->entity distance).
        g_dist = jnp.linalg.norm(pos, axis=-1, keepdims=True) / arena  # (E,1)
        g_base = jnp.concatenate(
            [
                pos,                                        # rel_pos := absolute
                vel,                                        # rel_vel := absolute
                g_dist,
                mass_n,
                true_onehot,                                # TRUE type (no spoof)
                locked_col,
                true_noise,                                 # TRUE noise (no spoof)
                is_held,
                size_col,
            ],
            axis=-1,
        )  # (E, Fe)
        true_is_decoy = (game.is_decoy & active).astype(jnp.float32)[:, None]  # (E,1)
        grounded_col = physics.grounded.astype(jnp.float32)[:, None]            # (E,1)
        global_entities = jnp.concatenate([g_base, true_is_decoy, grounded_col], axis=-1)  # (E,Fg)
        global_entities = global_entities * active[:, None].astype(jnp.float32)
        global_mask = active                                # existence only

        # ----------------------------------------------------------------- #
        # SELF proprioception (CONTRACT §3.3, Fs = 14).
        # ----------------------------------------------------------------- #
        self_pos = obs_pos / arena                          # (A,3)
        self_vel = obs_vel                                  # (A,3)
        head = physics.heading[:A]
        facing = jnp.stack([jnp.sin(head), jnp.cos(head)], axis=-1)  # (A,2)
        stamina_n = (game.stamina / self.env.stamina_max)[:, None]   # (A,1)
        team_onehot = jax.nn.one_hot(jnp.clip(game.team, 0, 1), 2)   # (A,2)
        team_onehot = team_onehot * (game.team >= 0)[:, None].astype(jnp.float32)
        prep_flag = jnp.broadcast_to((game.phase == 0).astype(jnp.float32), (A, 1))
        holding_flag = (game.holding >= 0).astype(jnp.float32)[:, None]
        grounded_flag = physics.grounded[:A].astype(jnp.float32)[:, None]
        self_feat = jnp.concatenate(
            [self_pos, self_vel, facing, stamina_n, team_onehot,
             prep_flag, holding_flag, grounded_flag],
            axis=-1,
        )  # (A, Fs)
        self_feat = self_feat * agent_active[:, None].astype(jnp.float32)

        return {
            "entities": local,                # (A, E, Fe)
            "entity_mask": entity_mask,       # (A, E) bool
            "self": self_feat,                # (A, Fs)
            "global_entities": global_entities,  # (E, Fg)
            "global_mask": global_mask,       # (E,) bool
            "agent_active": agent_active,     # (A,) bool
        }

    def _is_held_mask(self, game: GameState) -> jnp.ndarray:
        """Return ``(E,)`` bool: entities currently grabbed by some agent.

        Parameters
        ----------
        game:
            Current game state.

        Returns
        -------
        held:
            ``(E,)`` bool.
        """
        E = self.E
        holding = game.holding                              # (A,)
        valid = holding >= 0
        idx = jnp.clip(holding, 0, E - 1)
        held = jnp.zeros((E,), dtype=bool)
        held = held.at[idx].max(valid)
        return held
