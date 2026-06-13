"""
tests/test_env_smoke.py -- end-to-end environment smoke test (envs/hide_and_seek.py).

A tiny, single-environment sanity check of the Brax-style functional API
(CONTRACT §2): ``reset(key)`` produces the observation dict with the contract
keys and shapes, and one ``step`` returns a finite per-agent reward of shape
``(A,)``. This exercises the whole observe -> visibility -> reward pipeline for
one transition without any training.

Guarded with ``importorskip("jax")`` so collection is clean without JAX.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # noqa: F841  (skip whole module without JAX)
import jax.numpy as jnp  # noqa: E402

from config import default_config  # noqa: E402
from envs import HideAndSeekEnv  # noqa: E402


def _expected_obs_shapes(cfg) -> dict:
    """Return the contract obs key -> shape mapping for one env (CONTRACT §3)."""
    A = cfg.env.max_agents
    E = cfg.env.max_entities
    Fe = cfg.model.entity_feat_dim
    Fg = cfg.model.global_entity_feat_dim
    Fs = cfg.model.self_feat_dim
    return {
        "entities": (A, E, Fe),
        "entity_mask": (A, E),
        "self": (A, Fs),
        "global_entities": (E, Fg),
        "global_mask": (E,),
        "agent_active": (A,),
    }


def _zero_action(cfg) -> dict:
    """A do-nothing action dict per CONTRACT §4 (move 0s, interact 0s)."""
    A = cfg.env.max_agents
    return {
        "move": jnp.zeros((A, cfg.model.action_move_dim), dtype=jnp.float32),
        "interact": jnp.zeros((A, cfg.model.n_discrete_actions), dtype=jnp.int32),
    }


def test_reset_obs_keys_and_shapes() -> None:
    """reset returns exactly the contract obs keys with the right shapes."""
    cfg = default_config()
    env = HideAndSeekEnv(cfg)
    state = env.reset(jax.random.PRNGKey(0))

    expected = _expected_obs_shapes(cfg)
    assert set(state.obs.keys()) == set(expected.keys())
    for key, shape in expected.items():
        assert tuple(state.obs[key].shape) == shape, key

    # Masks are boolean; reward starts zeroed at shape (A,).
    assert state.obs["entity_mask"].dtype == bool
    assert state.obs["global_mask"].dtype == bool
    assert state.obs["agent_active"].dtype == bool
    assert state.reward.shape == (cfg.env.max_agents,)


def test_single_step_returns_finite_reward() -> None:
    """One step yields a finite per-agent reward of shape (A,)."""
    cfg = default_config()
    env = HideAndSeekEnv(cfg)
    state = env.reset(jax.random.PRNGKey(1))

    next_state = env.step(state, _zero_action(cfg))

    assert next_state.reward.shape == (cfg.env.max_agents,)
    assert bool(jnp.all(jnp.isfinite(next_state.reward)))
    # The obs contract is preserved across a step.
    expected = _expected_obs_shapes(cfg)
    for key, shape in expected.items():
        assert tuple(next_state.obs[key].shape) == shape, key
    # done is a scalar boolean (single env, no auto-reset in step).
    assert next_state.done.shape == ()
    assert next_state.done.dtype == bool


def test_step_is_jit_compatible() -> None:
    """reset/step inline into jax.jit (no host branching in the hot path)."""
    cfg = default_config()
    env = HideAndSeekEnv(cfg)

    reset_jit = jax.jit(env.reset)
    step_jit = jax.jit(env.step)

    state = reset_jit(jax.random.PRNGKey(2))
    next_state = step_jit(state, _zero_action(cfg))
    assert next_state.reward.shape == (cfg.env.max_agents,)
    assert bool(jnp.all(jnp.isfinite(next_state.reward)))
