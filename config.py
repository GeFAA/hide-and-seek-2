"""
config.py -- Central, immutable configuration for **Hide & Seek 2.0**.

A single ``Config`` object (composed of ``EnvConfig`` / ``ModelConfig`` /
``TrainConfig``) is threaded through the ENTIRE stack: the JAX environment, the
Flax networks, and the MAPPO trainer.

Why one file?
-------------
Entity-centric MARL lives and dies by shape consistency. The environment emits a
padded list of *entity feature vectors*; the Transformer policy consumes them as
tokens. If the env says an entity vector is 20-d and the network expects 18-d,
you get either a crash or -- worse -- a silent broadcast bug that quietly destroys
training. To make that class of bug impossible, every shared dimension
(``entity_feat_dim``, ``self_feat_dim``, ``global_entity_feat_dim``,
``max_entities``, ``max_agents``, action dims) is **derived here once** and read
everywhere else. Never hard-code these downstream.

See ``docs/CONTRACT.md`` for the authoritative description of how these
dimensions map onto concrete feature layouts.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Tuple

# ---------------------------------------------------------------------------
# Canonical entity taxonomy.
# The *order* of this tuple defines the one-hot index of every entity type and
# is part of the public contract -- do not reorder without bumping a version.
# ---------------------------------------------------------------------------
ENTITY_TYPES: Tuple[str, ...] = (
    "hider",      # 0  agent, hider team
    "seeker",     # 1  agent, seeker team
    "box_light",  # 2  movable by a single agent
    "box_heavy",  # 3  needs >= coop_required_agents pushing      (2.0: cooperative physics)
    "ramp",       # 4  climbable; enables reaching elevated surfaces
    "decoy",      # 5  noise-maker / sensory decoy                (2.0: deception)
    "wall",       # 6  static; fragile variant is rammable        (2.0: destructible env)
    "door",       # 7  opens after sustained contact -> chokepoint (2.0: interactable env)
)
N_ENTITY_TYPES: int = len(ENTITY_TYPES)
TYPE_TO_ID = {name: i for i, name in enumerate(ENTITY_TYPES)}


# ===========================================================================
# Environment configuration
# ===========================================================================
@dataclass
class EnvConfig:
    """Physics, episode, sensing and (crucially) the 2.0 mechanics."""

    # --- population ---------------------------------------------------------
    n_hiders_max: int = 3
    n_seekers_max: int = 3
    min_team_size: int = 1          # procedural team-size range (inclusive)
    max_team_size: int = 3

    # --- props (max counts double as the padding sizes) ---------------------
    n_boxes_max: int = 4
    n_ramps_max: int = 2
    n_decoys_max: int = 2
    n_walls_max: int = 6
    n_doors_max: int = 2

    # --- arena & time -------------------------------------------------------
    arena_size: float = 12.0        # square arena half-extent is arena_size/2
    dt: float = 0.1                 # control timestep (seconds)
    physics_substeps: int = 4       # sub-steps per control step (stability)
    max_steps: int = 240            # episode horizon (control steps)
    prep_steps: int = 96            # hiders-only preparation phase length

    # --- locomotion ---------------------------------------------------------
    agent_radius: float = 0.4
    agent_max_force: float = 12.0
    agent_max_torque: float = 4.0
    linear_damping: float = 4.0
    angular_damping: float = 6.0

    # --- 2.0: variable mass & cooperative physics ---------------------------
    box_light_mass: float = 1.0
    box_heavy_mass: float = 6.0
    coop_required_agents: int = 2   # # of simultaneous pushers a heavy box needs
    coop_force_threshold: float = 10.0  # combined applied force to budge a heavy box

    # --- 2.0: stamina -------------------------------------------------------
    stamina_max: float = 100.0
    stamina_regen: float = 8.0      # /sec while not sprinting
    sprint_drain: float = 20.0      # /sec at full sprint
    heavy_push_drain: float = 15.0  # extra /sec while moving heavy objects
    sprint_force_mult: float = 1.8  # force multiplier when sprint is engaged
    sprint_cmd_threshold: float = 0.8  # |move| above this infers sprint intent

    # --- 2.0: sensing / fog of war / lighting -------------------------------
    lidar_n_rays: int = 30
    lidar_range: float = 8.0
    vision_range: float = 9.0
    vision_cone_deg: float = 135.0
    n_fog_patches: int = 3
    fog_radius: float = 2.5
    fog_vision_mult: float = 0.45   # range multiplier for rays passing through fog

    # --- 2.0: decoys --------------------------------------------------------
    decoy_noise_radius: float = 6.0
    decoy_active_steps: int = 40

    # --- 2.0: destructible walls & doors ------------------------------------
    wall_break_speed: float = 6.0   # ram speed above which a fragile wall breaks
    door_open_steps: int = 25       # contact-steps required to open a door

    # --- anti box-surfing exploit (strict Newtonian gating) -----------------
    ground_contact_required: bool = True  # locomotion force only applies if grounded

    # --- rewards ------------------------------------------------------------
    reward_scale: float = 1.0

    # --- derived (do not set; filled by __post_init__) ----------------------
    max_agents: int = field(init=False)
    max_entities: int = field(init=False)

    def __post_init__(self) -> None:
        self.max_agents = self.n_hiders_max + self.n_seekers_max
        self.max_entities = (
            self.max_agents
            + self.n_boxes_max
            + self.n_ramps_max
            + self.n_decoys_max
            + self.n_walls_max
            + self.n_doors_max
        )


# ===========================================================================
# Model configuration
# ===========================================================================
@dataclass
class ModelConfig:
    """Transformer encoder + GRU memory + hybrid (continuous/discrete) heads."""

    n_entity_types: int = N_ENTITY_TYPES

    # --- transformer encoder ------------------------------------------------
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    ff_dim: int = 256
    dropout: float = 0.0

    # --- recurrent memory (object permanence) -------------------------------
    gru_hidden: int = 128

    # --- action space -------------------------------------------------------
    # Continuous locomotion: (force_x, force_y, torque) squashed to [-1, 1].
    action_move_dim: int = 3
    # Discrete toggles, one categorical per entry:
    #   [0] grab/release nearest grabbable, [1] lock/unlock, [2] decoy on/off.
    action_discrete_nvec: Tuple[int, ...] = (2, 2, 2)
    log_std_init: float = -0.5

    # --- derived feature dims (do not set; see docs/CONTRACT.md) ------------
    entity_feat_dim: int = field(init=False)
    global_entity_feat_dim: int = field(init=False)
    self_feat_dim: int = field(init=False)

    def __post_init__(self) -> None:
        # Local (actor) entity vector layout:
        #   rel_pos(3) rel_vel(3) dist(1) mass(1) type_onehot(T)
        #   locked(1) emitted_noise(1) is_held(1) size(1)
        # => 12 + T
        self.entity_feat_dim = 12 + self.n_entity_types
        # Critic gets 2 privileged extras appended: true_is_decoy(1) grounded(1)
        self.global_entity_feat_dim = self.entity_feat_dim + 2
        # Self proprioception:
        #   pos(3) vel(3) facing_sincos(2) stamina(1) team_onehot(2)
        #   prep_flag(1) holding_flag(1) grounded(1)  => 14
        self.self_feat_dim = 14

    @property
    def n_discrete_actions(self) -> int:
        return len(self.action_discrete_nvec)


# ===========================================================================
# Training configuration (MAPPO, vectorized, end-to-end on device)
# ===========================================================================
@dataclass
class TrainConfig:
    # --- vectorization & horizon -------------------------------------------
    num_envs: int = 2048            # parallel environments (all on GPU)
    num_steps: int = 128            # rollout length per PPO update
    total_timesteps: int = 200_000_000

    # --- PPO optimization ---------------------------------------------------
    update_epochs: int = 4
    num_minibatches: int = 8
    lr: float = 3e-4
    anneal_lr: bool = True
    max_grad_norm: float = 0.5
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.01

    # --- CTDE / parameter sharing ------------------------------------------
    shared_policy_per_team: bool = True   # share actor params within a team

    # --- historical self-play via ELO --------------------------------------
    elo_k: float = 16.0
    elo_init: float = 1200.0
    opponent_pool_size: int = 20
    snapshot_interval_updates: int = 50
    past_opponent_prob: float = 0.5       # P(sample a frozen historical opponent)

    # --- misc ---------------------------------------------------------------
    seed: int = 0
    log_interval: int = 1
    use_wandb: bool = False

    # --- derived ------------------------------------------------------------
    batch_size: int = field(init=False)
    minibatch_size: int = field(init=False)
    num_updates: int = field(init=False)

    def __post_init__(self) -> None:
        self.batch_size = self.num_envs * self.num_steps
        self.minibatch_size = self.batch_size // self.num_minibatches
        self.num_updates = self.total_timesteps // (self.num_envs * self.num_steps)


# ===========================================================================
# Top-level config
# ===========================================================================
@dataclass
class Config:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def default_config() -> Config:
    """Return a fresh default configuration."""
    return Config()


def debug_config() -> Config:
    """A tiny config for CPU smoke-tests (fast to trace, cheap to run)."""
    cfg = default_config()
    cfg.train = replace(
        cfg.train,
        num_envs=8,
        num_steps=16,
        total_timesteps=8 * 16 * 4,
        num_minibatches=2,
    )
    cfg.train.__post_init__()
    return cfg


__all__ = [
    "ENTITY_TYPES",
    "N_ENTITY_TYPES",
    "TYPE_TO_ID",
    "EnvConfig",
    "ModelConfig",
    "TrainConfig",
    "Config",
    "default_config",
    "debug_config",
]
