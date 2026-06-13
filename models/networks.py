"""
models/networks.py -- Construction & initialization helpers for the policy stack.

This module is the *non-jitted setup* layer (CONTRACT §9): it builds dummy inputs
shaped purely from :class:`Config`, calls each module's ``.init`` to materialize
parameters, and provides convenience factories for the recurrent carries. The
trainer imports these to obtain initial ``params`` / ``carry`` pytrees, then runs
everything else inside ``jax.jit`` / ``jax.lax.scan``.

Nothing here hard-codes a dimension -- every shape is read from ``cfg`` (which in
turn derives ``entity_feat_dim``, ``global_entity_feat_dim``, ``self_feat_dim``,
``max_entities``, ``max_agents`` per ``config.py`` / CONTRACT §3).

The :class:`ActorCritic` dataclass bundles the two modules together with their
**separate** parameter pytrees (actor and critic do not share weights -- the
critic is privileged/centralized).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax.core import FrozenDict

from config import Config
from models.actor import ActorRNN
from models.critic import CriticRNN
from models.memory import ScannedGRU

# Type aliases for clarity.
Params = Any           # a Flax parameter pytree
Carry = jnp.ndarray    # a GRU hidden-state array


# ---------------------------------------------------------------------------
# Dummy-input builders. Shapes follow CONTRACT §3 with a leading TIME axis (the
# recurrent modules are scanned over time) and a single batch dim for init.
# We init with T=1, B=1 -- the parameters are shape-agnostic in the batch/time
# axes, so the smallest valid trace suffices.
# ---------------------------------------------------------------------------
def _dummy_actor_inputs(
    cfg: Config,
    time: int = 1,
    batch: int = 1,
) -> Tuple[Tuple[Carry, Tuple[Dict[str, jnp.ndarray], jnp.ndarray]]]:
    """Build dummy ``(carry, (obs, dones))`` inputs for :class:`ActorRNN.init`.

    Args:
        cfg: The full configuration (dims read from ``cfg.env`` / ``cfg.model``).
        time: Leading time-axis length for the dummy trace.
        batch: Batch-axis length for the dummy trace.

    Returns:
        A 1-tuple wrapping ``(carry, (obs, dones))`` ready to splat into
        ``ActorRNN().init(key, *args)``.
    """
    E = cfg.env.max_entities
    Fe = cfg.model.entity_feat_dim
    Fs = cfg.model.self_feat_dim

    obs = {
        # CONTRACT §3 actor keys (time-major, single batch).
        "entities": jnp.zeros((time, batch, E, Fe), dtype=jnp.float32),
        "entity_mask": jnp.ones((time, batch, E), dtype=bool),
        "self": jnp.zeros((time, batch, Fs), dtype=jnp.float32),
        "agent_active": jnp.ones((time, batch), dtype=bool),
    }
    dones = jnp.zeros((time, batch), dtype=bool)
    carry = ScannedGRU.initialize_carry((batch,), cfg.model.gru_hidden)
    return ((carry, (obs, dones)),)


def _dummy_critic_inputs(
    cfg: Config,
    time: int = 1,
    batch: int = 1,
) -> Tuple[Tuple[Carry, Tuple[Dict[str, jnp.ndarray], jnp.ndarray]]]:
    """Build dummy ``(carry, (global_obs, dones))`` inputs for the critic.

    Args:
        cfg: The full configuration.
        time: Leading time-axis length for the dummy trace.
        batch: Batch-axis length for the dummy trace.

    Returns:
        A 1-tuple wrapping ``(carry, (global_obs, dones))`` ready to splat into
        ``CriticRNN().init(key, *args)``.
    """
    E = cfg.env.max_entities
    Fg = cfg.model.global_entity_feat_dim

    global_obs = {
        # CONTRACT §3 critic keys (time-major, single batch).
        "global_entities": jnp.zeros((time, batch, E, Fg), dtype=jnp.float32),
        "global_mask": jnp.ones((time, batch, E), dtype=bool),
    }
    dones = jnp.zeros((time, batch), dtype=bool)
    carry = ScannedGRU.initialize_carry((batch,), cfg.model.gru_hidden)
    return ((carry, (global_obs, dones)),)


# ---------------------------------------------------------------------------
# Parameter initializers.
# ---------------------------------------------------------------------------
def init_actor(cfg: Config, key: jax.Array) -> Params:
    """Initialize :class:`ActorRNN` parameters from config-shaped dummy inputs.

    Args:
        cfg: The full configuration.
        key: A ``PRNGKey`` for parameter initialization.

    Returns:
        The actor parameter pytree (a ``FrozenDict``).
    """
    model = ActorRNN(cfg.model)
    (args,) = _dummy_actor_inputs(cfg)
    variables = model.init(key, *args)
    return variables["params"]


def init_critic(cfg: Config, key: jax.Array) -> Params:
    """Initialize :class:`CriticRNN` parameters from config-shaped dummy inputs.

    Args:
        cfg: The full configuration.
        key: A ``PRNGKey`` for parameter initialization.

    Returns:
        The critic parameter pytree (a ``FrozenDict``).
    """
    model = CriticRNN(cfg.model)
    (args,) = _dummy_critic_inputs(cfg)
    variables = model.init(key, *args)
    return variables["params"]


def initialize_carries(
    cfg: Config,
    batch_actor: int,
    batch_critic: int,
) -> Tuple[Carry, Carry]:
    """Create zero recurrent carries for the actor and critic.

    The actor and critic are scanned independently and may be flattened over
    different leading batch shapes (CONTRACT §5 / §8). In the current MAPPO
    trainer both are run per-env, per-agent, so both batches equal
    ``num_envs * max_agents`` -- but they are passed as **separate** arguments so
    a caller is free to give the critic a different (e.g. per-env-only) batch.

    Args:
        cfg: The full configuration (uses ``cfg.model.gru_hidden``).
        batch_actor: Leading batch dimension for the **actor** carry, e.g.
            ``num_envs * max_agents`` (one recurrent state per agent per env).
        batch_critic: Leading batch dimension for the **critic** carry, e.g.
            ``num_envs * max_agents`` (the centralized critic is run per-agent in
            the current trainer; a per-env critic would pass ``num_envs``).

    Returns:
        ``(actor_carry, critic_carry)`` -- two zero arrays of shape
        ``(batch_actor, gru_hidden)`` and ``(batch_critic, gru_hidden)``,
        each built via :meth:`ScannedGRU.initialize_carry`.
    """
    actor_carry = ScannedGRU.initialize_carry((batch_actor,), cfg.model.gru_hidden)
    critic_carry = ScannedGRU.initialize_carry((batch_critic,), cfg.model.gru_hidden)
    return actor_carry, critic_carry


# ---------------------------------------------------------------------------
# Param-count utility.
# ---------------------------------------------------------------------------
def count_params(params: Params) -> int:
    """Count the total number of scalar parameters in a pytree.

    Args:
        params: Any parameter pytree (e.g. an actor or critic ``params``).

    Returns:
        The total scalar parameter count.
    """
    leaves = jax.tree_util.tree_leaves(params)
    return int(sum(jnp.asarray(leaf).size for leaf in leaves))


# ---------------------------------------------------------------------------
# Bundled Actor + Critic (separate params, CTDE).
# ---------------------------------------------------------------------------
@dataclass
class ActorCritic:
    """A convenience bundle of the actor and critic modules + their params.

    The actor and critic keep **separate** parameter pytrees (no weight sharing):
    the actor is decentralized/local, the critic is centralized/privileged. This
    dataclass simply pairs the two ``nn.Module`` instances with their params so a
    trainer can carry one object around.

    Attributes:
        actor: The :class:`ActorRNN` module (apply-able with ``actor.apply``).
        critic: The :class:`CriticRNN` module.
        actor_params: The actor's parameter pytree.
        critic_params: The critic's parameter pytree.
    """

    actor: ActorRNN
    critic: CriticRNN
    actor_params: Params
    critic_params: Params

    @classmethod
    def create(cls, cfg: Config, key: jax.Array) -> "ActorCritic":
        """Build modules and initialize both parameter sets from one key.

        Args:
            cfg: The full configuration.
            key: A ``PRNGKey``; split internally into actor/critic init keys.

        Returns:
            A fully-initialized :class:`ActorCritic` bundle.
        """
        actor_key, critic_key = jax.random.split(key)
        actor = ActorRNN(cfg.model)
        critic = CriticRNN(cfg.model)
        actor_params = init_actor(cfg, actor_key)
        critic_params = init_critic(cfg, critic_key)
        return cls(
            actor=actor,
            critic=critic,
            actor_params=actor_params,
            critic_params=critic_params,
        )

    def param_counts(self) -> Dict[str, int]:
        """Return scalar parameter counts for actor, critic and the total.

        Returns:
            A dict ``{"actor": ..., "critic": ..., "total": ...}``.
        """
        a = count_params(self.actor_params)
        c = count_params(self.critic_params)
        return {"actor": a, "critic": c, "total": a + c}
