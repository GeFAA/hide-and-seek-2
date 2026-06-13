"""
trainers/ -- Vectorized MAPPO training stack for **Hide & Seek 2.0**.

End-to-end on-device (PureJaxRL / JaxMARL style) multi-agent PPO with CTDE
(decentralized actor, centralized privileged critic) and ELO-weighted historical
self-play. See ``docs/CONTRACT.md`` §8 for the authoritative API.

Public surface
--------------
* :func:`make_train` -- ``make_train(cfg) -> train(rng)``; the jittable MAPPO
  training function (rollout scan -> GAE -> PPO update -> self-play snapshot).
* :class:`EloManager` -- live ELO bookkeeping wrapping ``utils.elo``.
* :class:`OpponentPool` -- fixed-size frozen-snapshot ring buffer with ELO,
  driving the self-play autocurriculum.

Submodules (importable directly for the finer-grained helpers):
``trainers.rollout`` (Transition + vec env glue), ``trainers.selfplay``
(pool + ELO ops), ``trainers.mappo`` (the train fn).
"""
from __future__ import annotations

from trainers.mappo import make_train
from trainers.selfplay import (
    EloManager,
    OpponentPool,
    batched_elo_update,
    init_opponent_pool,
    pool_is_empty,
    push_snapshot,
    sample_opponent,
    set_rating,
)
from trainers.rollout import (
    Transition,
    make_vec_env_reset,
    make_vec_env_step,
    stack_actor_obs,
    stack_critic_obs,
)

__all__ = [
    # primary contract exports
    "make_train",
    "EloManager",
    "OpponentPool",
    # self-play helpers
    "init_opponent_pool",
    "push_snapshot",
    "sample_opponent",
    "pool_is_empty",
    "set_rating",
    "batched_elo_update",
    # rollout helpers
    "Transition",
    "make_vec_env_reset",
    "make_vec_env_step",
    "stack_actor_obs",
    "stack_critic_obs",
]
