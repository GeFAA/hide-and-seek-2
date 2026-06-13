"""Quickstart demo for Hide & Seek 2.0.

A minimal, self-contained sanity check that ties the whole stack together:

1. build a tiny :func:`config.debug_config`;
2. construct the :class:`~envs.HideAndSeekEnv`;
3. ``jax.vmap`` ``reset`` over a handful of parallel environments;
4. initialise :class:`~models.ActorRNN` and :class:`~models.CriticRNN`;
5. run a short rollout — sampling actions from the (randomly initialised)
   policy — and print the shapes of the observation tensors at each step.

Run it from the repository root so that absolute imports resolve::

    PYTHONPATH=. python examples/quickstart.py

This file is written so that it **imports cleanly even when JAX (or the rest of
the project) is not installed**: the heavy lifting lives inside :func:`main`,
which is guarded so that a missing dependency prints an instructive message and
exits with a non-zero status rather than raising an opaque traceback. That makes
the script usable as a layout sanity check before an accelerator is set up.
"""

from __future__ import annotations

import sys
from typing import Any

# ---------------------------------------------------------------------------
# config.py has no third-party dependencies, so it is always importable. We use
# it both for the demo and to print useful contract dimensions even when JAX is
# absent.
# ---------------------------------------------------------------------------
from config import Config, debug_config


def _print_contract_summary(cfg: Config) -> None:
    """Print the key derived dimensions from the config (no JAX required).

    Args:
        cfg: the resolved :class:`config.Config` for this demo.
    """
    env, model, train = cfg.env, cfg.model, cfg.train
    print("Hide & Seek 2.0 — contract dimensions (from config.py)")
    print("-" * 60)
    print(f"  max_agents   A  = {env.max_agents}")
    print(f"  max_entities E  = {env.max_entities}")
    print(f"  entity_feat_dim        Fe = {model.entity_feat_dim}")
    print(f"  global_entity_feat_dim Fg = {model.global_entity_feat_dim}")
    print(f"  self_feat_dim          Fs = {model.self_feat_dim}")
    print(f"  action: move dim = {model.action_move_dim}, "
          f"discrete nvec = {model.action_discrete_nvec}")
    print(f"  demo rollout: num_envs = {train.num_envs}, "
          f"num_steps = {train.num_steps}")
    print("-" * 60)


def _import_jax_stack() -> dict[str, Any]:
    """Import every JAX-dependent symbol the demo needs.

    Isolated in a function so the module itself imports without JAX. Raises
    :class:`ImportError` (caught by :func:`main`) if anything is missing.

    Returns:
        A namespace dict of the imported callables/classes.
    """
    import jax  # noqa: F401  (re-exported below)
    import jax.numpy as jnp  # noqa: F401

    # Project modules — these themselves require JAX/Flax, so they live here.
    from envs import HideAndSeekEnv
    from models import ActorRNN, CriticRNN, ScannedGRU
    from models.actor import sample_and_logprob

    return {
        "jax": jax,
        "jnp": jnp,
        "HideAndSeekEnv": HideAndSeekEnv,
        "ActorRNN": ActorRNN,
        "CriticRNN": CriticRNN,
        "ScannedGRU": ScannedGRU,
        "sample_and_logprob": sample_and_logprob,
    }


def _run_demo(cfg: Config, ns: dict[str, Any], n_steps: int = 4) -> None:
    """Run the short vmapped rollout and print observation shapes.

    Args:
        cfg: the demo configuration.
        ns:  the imported JAX-stack namespace from :func:`_import_jax_stack`.
        n_steps: number of environment steps to roll out.
    """
    jax = ns["jax"]
    jnp = ns["jnp"]
    HideAndSeekEnv = ns["HideAndSeekEnv"]
    ActorRNN = ns["ActorRNN"]
    CriticRNN = ns["CriticRNN"]
    ScannedGRU = ns["ScannedGRU"]
    sample_and_logprob = ns["sample_and_logprob"]

    num_envs = cfg.train.num_envs
    A = cfg.env.max_agents

    # --- environment -------------------------------------------------------
    env = HideAndSeekEnv(cfg)
    key = jax.random.PRNGKey(cfg.train.seed)
    key, reset_key = jax.random.split(key)

    # One reset per environment; vmap turns the single-env pure function into a
    # batched one (the JaxMARL / PureJaxRL convention). No Python loop over envs.
    reset_keys = jax.random.split(reset_key, num_envs)
    state = jax.vmap(env.reset)(reset_keys)

    print("\nInitial observation shapes (leading axis = num_envs):")
    for obs_key, value in state.obs.items():
        print(f"  obs[{obs_key!r:18}] -> {tuple(value.shape)}  {value.dtype}")

    # --- networks ----------------------------------------------------------
    actor = ActorRNN(cfg.model)
    critic = CriticRNN(cfg.model)

    # GRU carries are (num_envs, A, gru_hidden): one hidden state per agent slot
    # per env. The agent axis A is folded into the batch by the trainer via
    # utils.pytree.batchify; here we keep it explicit for clarity.
    carry_shape = (num_envs, A)
    actor_carry = ScannedGRU.initialize_carry(carry_shape, cfg.model.gru_hidden)
    critic_carry = ScannedGRU.initialize_carry(carry_shape, cfg.model.gru_hidden)

    # The ScannedGRU expects a leading TIME axis on its inputs; for this demo we
    # step one timestep at a time, so we add a length-1 time axis and squeeze it.
    def add_time(x):
        return jax.tree_util.tree_map(lambda a: a[None, ...], x)

    def drop_time(x):
        return jax.tree_util.tree_map(lambda a: a[0], x)

    # Initialise parameters. Actors see only LOCAL masked obs; the critic sees
    # the PRIVILEGED global obs (CTDE — see docs/CONTRACT.md §5).
    key, ka, kc = jax.random.split(key, 3)
    dones0 = jnp.zeros((num_envs, A), dtype=bool)
    local_obs0 = {
        "entities": state.obs["entities"],
        "entity_mask": state.obs["entity_mask"],
        "self": state.obs["self"],
        "agent_active": state.obs["agent_active"],
    }
    global_obs0 = {
        "global_entities": state.obs["global_entities"],
        "global_mask": state.obs["global_mask"],
    }
    actor_params = actor.init(ka, actor_carry, add_time((local_obs0, dones0)))
    critic_params = critic.init(kc, critic_carry, add_time((global_obs0, dones0)))

    # --- rollout -----------------------------------------------------------
    print(f"\nRolling out {n_steps} steps over {num_envs} envs "
          f"(random-init policy)...")
    for t in range(n_steps):
        dones = jnp.zeros((num_envs, A), dtype=bool)
        local_obs = {
            "entities": state.obs["entities"],
            "entity_mask": state.obs["entity_mask"],
            "self": state.obs["self"],
            "agent_active": state.obs["agent_active"],
        }

        actor_carry, pi = actor.apply(
            actor_params, actor_carry, add_time((local_obs, dones))
        )
        # sample_and_logprob returns (action_dict, log_prob, entropy) per
        # CONTRACT §5 / PINNED P2; drop the time axis before sampling.
        key, sk = jax.random.split(key)
        action, _log_prob, _entropy = sample_and_logprob(drop_time(pi), sk)

        state = jax.vmap(env.step)(state, action)
        rew = state.reward
        print(f"  step {t}: reward shape {tuple(rew.shape)}, "
              f"mean reward {float(jnp.mean(rew)):+.3f}, "
              f"done(any) {bool(jnp.any(state.done))}")

    print("\nQuickstart complete. The stack is wired end-to-end on device.")


def main() -> int:
    """Entry point. Returns a process exit code (0 on success)."""
    cfg = debug_config()
    _print_contract_summary(cfg)

    try:
        ns = _import_jax_stack()
    except ImportError as exc:  # JAX / Flax / project modules not available.
        print()
        print("=" * 60)
        print("JAX (or another required dependency) is not importable, so the")
        print("live rollout was skipped. The contract dimensions above were")
        print("printed straight from config.py and require no dependencies.")
        print()
        print(f"  underlying import error: {exc}")
        print()
        print("To run the full demo, install the dependencies and re-run:")
        print("    pip install -r requirements.txt")
        print("    PYTHONPATH=. python examples/quickstart.py")
        print("(For GPU, install a CUDA jaxlib per the JAX install matrix.)")
        print("=" * 60)
        return 0  # Not a failure: importing cleanly without JAX is intended.

    _run_demo(cfg, ns)
    return 0


if __name__ == "__main__":
    sys.exit(main())
