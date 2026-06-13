"""
tests/test_config.py -- Derived-dimension & hyperparameter consistency checks.

This module needs **no JAX** (it imports only the pure-Python :mod:`config`),
so it always runs and is the fastest guard against the single most dangerous
class of bug in an entity-centric MARL stack: a shared dimension drifting out of
sync with the documented contract (``docs/CONTRACT.md``).

Every shared dimension is *derived once* in ``config.py``; these tests pin the
concrete values the rest of the project (and the contract) assume.
"""
from __future__ import annotations

from config import (
    ENTITY_TYPES,
    N_ENTITY_TYPES,
    Config,
    debug_config,
    default_config,
)


def test_entity_taxonomy_order() -> None:
    """The canonical entity order is part of the public one-hot contract."""
    assert ENTITY_TYPES == (
        "hider",
        "seeker",
        "box_light",
        "box_heavy",
        "ramp",
        "decoy",
        "wall",
        "door",
    )
    assert N_ENTITY_TYPES == 8
    # box_light is id 2, box_heavy is id 3 (see docs/FEATURES_2.0.md).
    assert ENTITY_TYPES.index("box_light") == 2
    assert ENTITY_TYPES.index("box_heavy") == 3


def test_derived_feature_dims() -> None:
    """Pin the derived feature/population dims (CONTRACT §3)."""
    cfg = default_config()
    model, env = cfg.model, cfg.env

    # Local entity vector: 12 + N_ENTITY_TYPES = 20.
    assert model.entity_feat_dim == 20
    assert model.entity_feat_dim == 12 + N_ENTITY_TYPES
    # Critic gets two privileged extras appended: 20 + 2 = 22.
    assert model.global_entity_feat_dim == 22
    assert model.global_entity_feat_dim == model.entity_feat_dim + 2
    # Self proprioception vector is fixed at 14.
    assert model.self_feat_dim == 14

    # Population / padding dims.
    assert env.max_agents == 6
    assert env.max_agents == env.n_hiders_max + env.n_seekers_max
    assert env.max_entities == 22
    assert env.max_entities == (
        env.max_agents
        + env.n_boxes_max
        + env.n_ramps_max
        + env.n_decoys_max
        + env.n_walls_max
        + env.n_doors_max
    )


def test_action_space_dims() -> None:
    """The hybrid action space matches the contract (move 3-d, 3 categoricals)."""
    model = default_config().model
    assert model.action_move_dim == 3
    assert tuple(model.action_discrete_nvec) == (2, 2, 2)
    assert model.n_discrete_actions == 3


def _assert_train_consistency(cfg: Config) -> None:
    """Shared assertions on a :class:`TrainConfig`'s derived batch sizes."""
    train = cfg.train
    # batch_size = num_envs * num_steps.
    assert train.batch_size == train.num_envs * train.num_steps
    # minibatch_size partitions the batch exactly (no remainder).
    assert train.minibatch_size == train.batch_size // train.num_minibatches
    assert train.minibatch_size * train.num_minibatches == train.batch_size
    # num_updates is total_timesteps over per-update experience.
    assert train.num_updates == train.total_timesteps // (
        train.num_envs * train.num_steps
    )
    assert train.batch_size > 0
    assert train.minibatch_size > 0


def test_train_default_consistency() -> None:
    """The default training config has internally consistent batch sizes."""
    _assert_train_consistency(default_config())


def test_train_debug_consistency() -> None:
    """``debug_config`` re-derives its sizes after shrinking num_envs/num_steps."""
    cfg = debug_config()
    _assert_train_consistency(cfg)
    # Sanity: the debug config really is tiny (fast CPU smoke tests).
    assert cfg.train.num_envs == 8
    assert cfg.train.num_steps == 16
    assert cfg.train.num_updates >= 1
