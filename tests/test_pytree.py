"""
tests/test_pytree.py -- batchify / unbatchify multi-agent layout (utils/pytree.py).

Validates the P4 convention: ``batchify`` collapses a *fixed* number of LEADING
axes (default the ``(num_envs, A)`` pair), preserving ALL trailing axes -- so the
4-D ``entities`` leaf ``(num_envs, A, E, Fe)`` becomes ``(num_envs*A, E, Fe)``
(NOT ``(num_envs*A*E, Fe)``). ``unbatchify(x, n_agents)`` is the inverse, with the
**env-outer / agent-inner** row convention (agent varies fastest within an env),
matching the trainer's ``reshape((num_envs, A) + rest)``.

Guarded with ``importorskip("jax")`` so collection is clean without JAX.
"""
from __future__ import annotations

import pytest

jax = pytest.importorskip("jax")  # noqa: F841  (skip whole module without JAX)
import jax.numpy as jnp  # noqa: E402

from config import default_config  # noqa: E402
from utils.pytree import batchify, unbatchify  # noqa: E402


def _make_obs(num_envs: int, A: int, E: int, Fe: int, Fs: int) -> dict:
    """Build a small obs-like pytree with distinct, recoverable values."""
    # Fill each leaf with arange so we can check exact element identity after a
    # batchify -> unbatchify round-trip.
    entities = jnp.arange(num_envs * A * E * Fe, dtype=jnp.float32).reshape(
        (num_envs, A, E, Fe)
    )
    entity_mask = (
        jnp.arange(num_envs * A * E).reshape((num_envs, A, E)) % 2
    ).astype(bool)
    self_feat = jnp.arange(num_envs * A * Fs, dtype=jnp.float32).reshape(
        (num_envs, A, Fs)
    )
    agent_active = (
        jnp.arange(num_envs * A).reshape((num_envs, A)) % 3 != 0
    )
    return {
        "entities": entities,
        "entity_mask": entity_mask,
        "self": self_feat,
        "agent_active": agent_active,
    }


def test_batchify_collapses_two_leading_axes() -> None:
    """(num_envs, A, *rest) -> (num_envs*A, *rest) for every leaf rank."""
    cfg = default_config()
    num_envs, A = 4, cfg.env.max_agents
    E, Fe, Fs = cfg.env.max_entities, cfg.model.entity_feat_dim, cfg.model.self_feat_dim
    obs = _make_obs(num_envs, A, E, Fe, Fs)

    flat = batchify(obs)  # default n_lead=2 merges (num_envs, A)

    # 4-D entities -> (num_envs*A, E, Fe): trailing E and Fe axes are PRESERVED.
    assert flat["entities"].shape == (num_envs * A, E, Fe)
    # 3-D entity_mask -> (num_envs*A, E).
    assert flat["entity_mask"].shape == (num_envs * A, E)
    # 3-D self -> (num_envs*A, Fs).
    assert flat["self"].shape == (num_envs * A, Fs)
    # 2-D agent_active -> (num_envs*A,).
    assert flat["agent_active"].shape == (num_envs * A,)


def test_round_trip_restores_shapes_and_values() -> None:
    """unbatchify(batchify(x), A) restores (num_envs, A, *rest) exactly."""
    cfg = default_config()
    num_envs, A = 5, cfg.env.max_agents
    E, Fe, Fs = cfg.env.max_entities, cfg.model.entity_feat_dim, cfg.model.self_feat_dim
    obs = _make_obs(num_envs, A, E, Fe, Fs)

    flat = batchify(obs)
    restored = unbatchify(flat, n_agents=A)

    for key, original in obs.items():
        assert restored[key].shape == original.shape, key
        assert bool(jnp.array_equal(restored[key], original)), key


def test_agent_active_round_trip() -> None:
    """The 2-D (num_envs, A) -> (num_envs*A,) -> (num_envs, A) round-trip (P4)."""
    num_envs, A = 3, 6
    agent_active = (jnp.arange(num_envs * A).reshape((num_envs, A)) % 2).astype(bool)

    flat = batchify({"agent_active": agent_active})["agent_active"]
    assert flat.shape == (num_envs * A,)

    restored = unbatchify({"agent_active": flat}, n_agents=A)["agent_active"]
    assert restored.shape == (num_envs, A)
    assert bool(jnp.array_equal(restored, agent_active))


def test_env_outer_agent_inner_ordering() -> None:
    """Rows are ordered env-outer / agent-inner (agent varies fastest)."""
    num_envs, A = 2, 3
    # Encode (env, agent) as env*10 + agent so the ordering is human-readable.
    env_idx = jnp.arange(num_envs)[:, None]
    agent_idx = jnp.arange(A)[None, :]
    tagged = (env_idx * 10 + agent_idx).astype(jnp.float32)  # (num_envs, A)

    flat = batchify({"x": tagged[..., None]})["x"][:, 0]  # (num_envs*A,)
    # env-outer / agent-inner => [00,01,02, 10,11,12].
    expected = jnp.array([0.0, 1.0, 2.0, 10.0, 11.0, 12.0])
    assert bool(jnp.array_equal(flat, expected))
