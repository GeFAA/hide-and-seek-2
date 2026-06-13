"""
envs/state.py -- Core state pytrees for **Hide & Seek 2.0**.

These three ``flax.struct.dataclass`` definitions (registered pytrees, fully
``jax.jit`` / ``jax.vmap`` safe) are the single in-memory representation of an
environment. They implement CONTRACT.md §2 *exactly* -- field names, shapes and
dtypes. **All shapes here are per single environment**; the trainer adds a
leading ``num_envs`` axis by ``jax.vmap``-ing :class:`HideAndSeekEnv.reset` /
``step``.

Shape symbols (read from :mod:`config`, never hard-coded):

* ``A = cfg.env.max_agents``     -- padded number of agent slots.
* ``E = cfg.env.max_entities``   -- padded number of *all* entities (agents +
  props). Agents occupy entity rows ``[0:A]`` by convention (see
  :mod:`envs.procedural`); props follow.
* ``Pf = cfg.env.n_fog_patches`` -- number of fog patch centers.

Notes
-----
* Static obstacles (walls) use a large finite "inf-proxy" mass rather than
  ``jnp.inf`` so that ``F / mass`` never produces NaNs in the physics step.
* ``type_id`` indexes into :data:`config.ENTITY_TYPES`; the *order* of that tuple
  is part of the public contract.
"""
from __future__ import annotations

import flax.struct
import jax.numpy as jnp

__all__ = ["PhysicsState", "GameState", "State"]


@flax.struct.dataclass
class PhysicsState:
    """Rigid-body kinematic & collision state for every entity (per env).

    All arrays have a leading entity axis of size ``E = cfg.env.max_entities``
    unless noted. The planar dynamics live in ``pos[..., :2]`` / ``vel[..., :2]``;
    the scalar elevation ``pos[..., 2]`` (``z``) is the 2.5D channel used only to
    model climbing / box-surfing for the anti-exploit ground-contact gate.

    Attributes
    ----------
    pos:
        ``(E, 3)`` float32 -- ``(x, y, z)`` where ``z`` is height / elevation.
    vel:
        ``(E, 3)`` float32 -- linear velocity (``z`` component models vertical
        motion only schematically; gravity/jumping is not simulated).
    heading:
        ``(E,)`` float32 -- facing angle in radians (meaningful for agents).
    ang_vel:
        ``(E,)`` float32 -- angular velocity (radians / second).
    mass:
        ``(E,)`` float32 -- inertial mass. Heavy boxes are large; static walls
        carry a finite inf-proxy mass.
    size:
        ``(E,)`` float32 -- collision radius / half-extent.
    type_id:
        ``(E,)`` int32 -- index into :data:`config.ENTITY_TYPES`.
    grounded:
        ``(E,)`` bool -- agent is in contact with the ground (``z == 0``). This
        is the anti box-surfing gate; only meaningful for agent rows.
    active:
        ``(E,)`` bool -- entity exists this episode (padding / existence mask).
    """

    pos: jnp.ndarray        # (E, 3) float32  x, y, z(height/elevation)
    vel: jnp.ndarray        # (E, 3) float32
    heading: jnp.ndarray    # (E,)   float32  facing angle (radians)
    ang_vel: jnp.ndarray    # (E,)   float32
    mass: jnp.ndarray       # (E,)   float32
    size: jnp.ndarray       # (E,)   float32  radius / half-extent
    type_id: jnp.ndarray    # (E,)   int32    index into ENTITY_TYPES
    grounded: jnp.ndarray   # (E,)   bool     anti-surf ground-contact gate
    active: jnp.ndarray     # (E,)   bool     padding / existence mask


@flax.struct.dataclass
class GameState:
    """Game-rule, ownership and 2.0-mechanic state (per env).

    Agent-indexed arrays (leading axis ``A = cfg.env.max_agents``) describe the
    agents; entity-indexed arrays (leading axis ``E``) describe ownership /
    interactable state of *all* entities. Agents are the first ``A`` entity rows,
    so an agent's id is identical in both index spaces.

    Attributes
    ----------
    team:
        ``(A,)`` int32 -- ``0=hider``, ``1=seeker``, ``-1`` for pad slots.
    stamina:
        ``(A,)`` float32 -- remaining stamina in ``[0, stamina_max]``.
    holding:
        ``(A,)`` int32 -- entity id this agent currently grabs, ``-1`` if none.
    locked:
        ``(E,)`` bool -- object locked immovable by a team.
    locked_by:
        ``(E,)`` int32 -- team id that locked the object, ``-1`` if none.
    is_decoy:
        ``(E,)`` bool -- **true** decoy identity (PRIVILEGED; never leaks into
        the actor's local observation).
    decoy_timer:
        ``(E,)`` int32 -- control-steps remaining while a decoy actively emits.
    emitted_noise:
        ``(E,)`` float32 -- noise intensity each entity broadcasts (a decoy
        spoofs this in the local view).
    wall_hp:
        ``(E,)`` float32 -- destructible-wall health; ``<= 0`` => broken.
    door_progress:
        ``(E,)`` float32 -- ``0..1`` door-open progress.
    fog_pos:
        ``(Pf, 2)`` float32 -- fog patch centers (xy).
    phase:
        ``()`` int32 -- ``0=prep``, ``1=main``.
    step:
        ``()`` int32 -- control-step counter within the episode.
    """

    team: jnp.ndarray           # (A,)  int32   0=hider 1=seeker, -1 pad
    stamina: jnp.ndarray        # (A,)  float32
    holding: jnp.ndarray        # (A,)  int32   held entity id, -1 if none
    locked: jnp.ndarray         # (E,)  bool
    locked_by: jnp.ndarray      # (E,)  int32   locking team, -1 if none
    is_decoy: jnp.ndarray       # (E,)  bool    PRIVILEGED true identity
    decoy_timer: jnp.ndarray    # (E,)  int32
    emitted_noise: jnp.ndarray  # (E,)  float32
    wall_hp: jnp.ndarray        # (E,)  float32
    door_progress: jnp.ndarray  # (E,)  float32
    fog_pos: jnp.ndarray        # (Pf, 2) float32
    phase: jnp.ndarray          # ()    int32   0=prep, 1=main
    step: jnp.ndarray           # ()    int32


@flax.struct.dataclass
class State:
    """Full environment state returned by ``reset`` / ``step`` (per env).

    Auto-reset on ``done`` is handled by the **trainer** wrapper (so ``step``
    stays branch-free), per PureJaxRL convention.

    Attributes
    ----------
    physics:
        :class:`PhysicsState` rigid-body kinematics.
    game:
        :class:`GameState` rules / ownership / 2.0-mechanic state.
    obs:
        Observation dict per CONTRACT.md §3 (see :meth:`HideAndSeekEnv.observe`).
    reward:
        ``(A,)`` float32 -- per-agent reward (team reward broadcast to members).
    done:
        ``()`` bool -- episode terminated or truncated.
    info:
        Metrics dict (``hiders_reward``, ``seekers_reward``, ``seen_frac`` and
        the 2.0 metrics).
    key:
        ``jax.random.PRNGKey`` carried for stochastic dynamics.
    """

    physics: PhysicsState
    game: GameState
    obs: dict
    reward: jnp.ndarray  # (A,) float32
    done: jnp.ndarray    # () bool
    info: dict
    key: jnp.ndarray     # PRNGKey
