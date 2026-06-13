"""
trainers/rollout.py -- Rollout plumbing for **Hide & Seek 2.0** MAPPO.

This module defines the on-device data structures and the *vectorized* env
glue used by the MAPPO trainer (``trainers/mappo.py``). Everything here is
written to live inside ``jax.jit`` / ``jax.lax.scan`` with **zero host<->device
copies** -- the whole point of the PureJaxRL / JaxMARL style is that a rollout is
just another tensor op the XLA compiler fuses with the learning update.

Contents
--------
* :class:`Transition` -- the per-timestep pytree collected during a rollout,
  matching the field list mandated by ``docs/CONTRACT.md`` §8.
* :func:`make_vec_env_step` / :func:`make_vec_env_reset` -- ``jax.vmap`` wrappers
  around the single-env :class:`envs.HideAndSeekEnv` functional API, with the
  **auto-reset on done** behaviour (``where(done, reset_state, next_state)``)
  that keeps ``env.step`` itself branch-free (per PureJaxRL convention).
* :func:`stack_actor_obs` / :func:`stack_critic_obs` -- pull the contract
  observation keys out of ``State.obs`` and shape them for the networks.

Shapes follow the contract: a single env's observation arrays carry a leading
agent axis ``A = cfg.env.max_agents``; the trainer adds a leading ``num_envs``
axis via :func:`jax.vmap`. We never hard-code dims here -- they flow from
``config.py`` through the env and the obs dict.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import flax

# NOTE: imported lazily-by-name in type hints only; the concrete env is passed
# into the wrapper factories so this module never needs a hard import of the
# (parallel-built) envs package at import time. This keeps `rollout.py`
# importable on a machine without the full stack present.
try:  # pragma: no cover - convenience import, not required for the module to load
    from envs import HideAndSeekEnv, State  # noqa: F401
except Exception:  # pragma: no cover
    HideAndSeekEnv = Any  # type: ignore
    State = Any  # type: ignore


# ===========================================================================
# Transition pytree (CONTRACT §8)
# ===========================================================================
@flax.struct.dataclass
class Transition:
    """One timestep of experience, stacked over time by ``lax.scan``.

    Every field carries a leading ``(num_steps, num_envs, ...)`` axis once a
    rollout has been scanned. Per the contract, the agent axis ``A`` is the
    *innermost* logical batch dimension and is flattened with
    ``utils.pytree.batchify`` only when feeding the network.

    Attributes
    ----------
    done:
        ``(num_envs,)`` bool -- episode terminal/truncation flag for the env at
        the step the transition was *emitted* (used both for GAE bootstrapping
        and for resetting the recurrent carry, PureJaxRL style).
    action:
        Dict ``{"move": (num_envs, A, move_dim), "interact": (num_envs, A,
        n_discrete)}`` -- the sampled hybrid action.
    value:
        ``(num_envs, A)`` float32 -- critic value estimate (centralized).
    reward:
        ``(num_envs, A)`` float32 -- per-agent (team-broadcast) reward.
    log_prob:
        ``(num_envs, A)`` float32 -- summed log-prob of the sampled hybrid
        action under the behaviour policy.
    obs:
        Dict of the **local** actor observation arrays (keys ``entities``,
        ``entity_mask``, ``self``, ``agent_active``), each with a leading
        ``(num_envs, A, ...)`` shape.
    global_obs:
        Dict of the **privileged** critic observation arrays (keys
        ``global_entities``, ``global_mask``), each with a leading
        ``(num_envs, ...)`` shape (no per-agent axis on the global tokens; the
        critic broadcasts over agents internally / via CTDE).
    avail:
        ``(num_envs, A)`` bool -- which agent slots are real this episode
        (mirror of ``obs['agent_active']``); kept as a distinct field because
        the loss masks padded agents out of every term.
    info:
        Dict of scalar metrics emitted by the env this step (see CONTRACT §7);
        carried through so the logger can aggregate without a host transfer.
    """

    done: jnp.ndarray
    action: Dict[str, jnp.ndarray]
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: Dict[str, jnp.ndarray]
    global_obs: Dict[str, jnp.ndarray]
    avail: jnp.ndarray
    info: Dict[str, jnp.ndarray]


# ===========================================================================
# Observation extraction helpers
# ===========================================================================
# The contract pins these key names; centralizing them here means a rename in
# the env only has to be reflected in one place downstream.
ACTOR_OBS_KEYS: Tuple[str, ...] = ("entities", "entity_mask", "self", "agent_active")
CRITIC_OBS_KEYS: Tuple[str, ...] = ("global_entities", "global_mask")


def stack_actor_obs(obs: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Select the decentralized **actor** inputs from a ``State.obs`` dict.

    Parameters
    ----------
    obs:
        The full observation dict produced by the env (CONTRACT §3).

    Returns
    -------
    dict
        Sub-dict containing exactly ``ACTOR_OBS_KEYS``. The actor is
        decentralized: it never sees ``global_*`` keys.
    """
    return {k: obs[k] for k in ACTOR_OBS_KEYS}


def stack_critic_obs(obs: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Select the centralized **critic** (privileged) inputs from ``State.obs``.

    Parameters
    ----------
    obs:
        The full observation dict produced by the env (CONTRACT §3).

    Returns
    -------
    dict
        Sub-dict containing exactly ``CRITIC_OBS_KEYS`` -- absolute-coord
        global tokens plus an existence mask. This is the CTDE privileged
        information the actor is forbidden from using.
    """
    return {k: obs[k] for k in CRITIC_OBS_KEYS}


# ===========================================================================
# Vectorized env wrappers (vmap over num_envs) with AUTO-RESET on done
# ===========================================================================
def make_vec_env_reset(env: "HideAndSeekEnv") -> Callable[[jnp.ndarray], "State"]:
    """Build a ``jax.vmap``-ed reset over a batch of PRNG keys.

    Parameters
    ----------
    env:
        A single-environment :class:`envs.HideAndSeekEnv` instance exposing the
        Brax-style functional API ``reset(key) -> State``.

    Returns
    -------
    callable
        ``vec_reset(keys) -> State`` where ``keys`` has shape ``(num_envs, 2)``
        and the returned :class:`State` has a leading ``num_envs`` axis on every
        leaf. No Python loop over environments -- pure ``vmap``.
    """

    def vec_reset(keys: jnp.ndarray) -> "State":
        return jax.vmap(env.reset)(keys)

    return vec_reset


def make_vec_env_step(
    env: "HideAndSeekEnv",
) -> Callable[
    [jnp.ndarray, "State", Dict[str, jnp.ndarray]], Tuple["State", jnp.ndarray]
]:
    """Build a ``jax.vmap``-ed step with **auto-reset on done**.

    The single-env ``env.step`` is intentionally branch-free (CONTRACT §2): it
    never resets itself. We layer the reset here, PureJaxRL style, by computing
    *both* the natural next state and a fresh reset state and selecting per-leaf
    with ``jnp.where(done, reset_leaf, next_leaf)``. Because ``done`` is a traced
    boolean we must use ``where`` (never a Python ``if``) so the whole thing
    stays inside ``jit``/``scan``.

    PRE-RESET TERMINAL DONE
    -----------------------
    The auto-reset overwrites *every* leaf of the terminal state with the fresh
    episode's leaf -- including ``done`` itself, which becomes ``False`` again in
    the returned (already-reset) state. If the trainer recorded *that* flag it
    would never see a ``True`` done, so GAE could never cut the bootstrap at an
    episode boundary and advantages would leak across episodes. To prevent that
    we return the **pre-reset** ``done`` (the genuine terminal flag from
    ``env.step``) *alongside* the auto-reset state, so the caller can store the
    true terminal flag in the :class:`Transition` while still handing the fresh
    observation to the next step. This keeps everything branch-free (the select
    is a pure ``jnp.where``; the terminal done is just carried out unchanged).

    Parameters
    ----------
    env:
        Single-environment :class:`envs.HideAndSeekEnv` exposing
        ``reset(key) -> State`` and ``step(state, action) -> State``.

    Returns
    -------
    callable
        ``vec_step(reset_keys, state, action) -> (State, terminal_done)``
        operating on the leading ``num_envs`` axis. ``State`` is the auto-reset
        state (fresh obs on terminal envs); ``terminal_done`` is the ``(num_envs,)``
        bool **pre-reset** done from ``env.step``. ``reset_keys`` supplies fresh
        randomness for the envs that happen to terminate this step (unused
        entries are simply ignored by the ``where`` for non-done envs).
    """

    def _single_step(
        reset_key: jnp.ndarray, state: "State", action: Dict[str, jnp.ndarray]
    ) -> Tuple["State", jnp.ndarray]:
        """Step one env, then auto-reset that one env if it just terminated.

        Returns the auto-reset state and the *pre-reset* terminal done flag.
        """
        next_state = env.step(state, action)
        reset_state = env.reset(reset_key)

        done = next_state.done  # scalar bool for this single env (PRE-reset)

        def _select(next_leaf: jnp.ndarray, reset_leaf: jnp.ndarray) -> jnp.ndarray:
            # Broadcast the scalar `done` across the leaf's shape and pick.
            # where(done, reset_state, next_state): on terminal steps we hand the
            # *fresh* episode's observation to the trainer; the terminal reward /
            # done flag we want for learning are captured in the Transition,
            # which the caller builds from `next_state` (pre-reset) and the
            # returned `terminal_done` below.
            mask = jnp.broadcast_to(done, next_leaf.shape)
            return jnp.where(mask, reset_leaf, next_leaf)

        # Auto-reset only the dynamic, observation-bearing leaves. The carried
        # PRNGKey leaf is replaced wholesale by the reset state's key so the env
        # stays decorrelated after a respawn.
        autoreset_state = jax.tree_util.tree_map(_select, next_state, reset_state)
        # Carry the genuine terminal flag out separately (auto-reset clobbered it
        # to False inside `autoreset_state.done`).
        return autoreset_state, done

    def vec_step(
        reset_keys: jnp.ndarray, state: "State", action: Dict[str, jnp.ndarray]
    ) -> Tuple["State", jnp.ndarray]:
        # in_axes: a fresh reset key per env, the batched state, the batched
        # action dict. vmap maps every leaf along axis 0 (the num_envs axis).
        # Returns (autoreset_state, terminal_done) both with a leading num_envs
        # axis; terminal_done is (num_envs,) bool.
        return jax.vmap(_single_step, in_axes=(0, 0, 0))(reset_keys, state, action)

    return vec_step


# ===========================================================================
# Convenience: split a batched PRNGKey for per-env reset randomness
# ===========================================================================
@partial(jax.jit, static_argnums=(1,))
def split_env_keys(key: jnp.ndarray, num_envs: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Split ``key`` into a carry key and ``num_envs`` per-env keys.

    Parameters
    ----------
    key:
        A single ``jax.random.PRNGKey``.
    num_envs:
        Static number of parallel environments.

    Returns
    -------
    (carry_key, env_keys):
        ``carry_key`` to thread forward, ``env_keys`` of shape
        ``(num_envs, 2)`` to scatter across the vectorized env.
    """
    key, sub = jax.random.split(key)
    env_keys = jax.random.split(sub, num_envs)
    return key, env_keys


__all__ = [
    "Transition",
    "ACTOR_OBS_KEYS",
    "CRITIC_OBS_KEYS",
    "stack_actor_obs",
    "stack_critic_obs",
    "make_vec_env_reset",
    "make_vec_env_step",
    "split_env_keys",
]
