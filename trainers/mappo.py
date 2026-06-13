"""
trainers/mappo.py -- Vectorized MAPPO for **Hide & Seek 2.0**, end-to-end on GPU.

This is the learning core. It follows the PureJaxRL / JaxMARL recipe: the entire
train loop -- environment stepping, recurrent policy/critic forward passes, GAE,
and the PPO minibatch updates -- is compiled into a single ``jax.lax.scan`` over
``num_updates``, with an inner ``lax.scan`` over ``num_steps`` for the rollout.

THE ZERO-COPY PIPELINE
----------------------
Everything below lives inside ``jit``/``scan``. There is **no host<->device
transfer in the loop**: no ``.item()``, no Python ``if`` on traced values, no
numpy round-trips, no per-env Python iteration. The env is ``vmap``-ed over
``num_envs`` and the agent axis is flattened with ``utils.pytree.batchify`` only
to feed the networks. This is precisely the change that turns OpenAI's 2019
result -- which took *weeks on large CPU actor clusters* (thousands of cores
shuttling rollouts to learners over the network) -- into *hours on a single GPU*:
the rollout never leaves the accelerator, so the XLA compiler fuses simulation
and learning into one dense kernel and we are bound by GPU FLOPs, not by the
CPU<->GPU<->network ferry.

CTDE (centralized training, decentralized execution)
----------------------------------------------------
* The **actor** (``ActorRNN``) acts from each agent's *local, masked* observation
  (``entities``/``entity_mask``/``self``) -- this is what ships to deployment.
* The **critic** (``CriticRNN``) consumes the *global, privileged* observation
  (``global_entities``/``global_mask``) -- only used to reduce variance during
  training. This asymmetry is the whole point of MAPPO.

Self-play
---------
For a fraction (~``past_opponent_prob``) of the vectorized envs, the *opponent
team's* actions are produced by a frozen historical snapshot drawn from the
:class:`trainers.selfplay.OpponentPool` (ELO-weighted), instead of the live
policy. Snapshots are pushed every ``snapshot_interval_updates``. See
``# historical self-play via ELO`` below.

All shared dimensions are read from ``cfg`` (``config.py``) -- nothing is
hard-coded here.
"""
from __future__ import annotations

from functools import partial
from typing import Any, Callable, Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training.train_state import TrainState

from config import Config
from trainers.rollout import (
    Transition,
    make_vec_env_reset,
    make_vec_env_step,
    stack_actor_obs,
    stack_critic_obs,
)
from trainers.selfplay import (
    OpponentPool,
    init_opponent_pool,
    push_snapshot,
    sample_opponent,
)

# --- parallel-built deps; imported defensively so this module stays importable
#     for inspection on a machine without the full stack. The real symbols are
#     required for an actual training run (guarded in train.py). -------------
try:  # pragma: no cover
    from envs import HideAndSeekEnv
    from models import (
        ActorRNN,
        CriticRNN,
        initialize_carries,
        sample_and_logprob,
        eval_logprob,
    )
    from utils.pytree import batchify, unbatchify
except Exception:  # pragma: no cover
    HideAndSeekEnv = Any  # type: ignore
    ActorRNN = Any  # type: ignore
    CriticRNN = Any  # type: ignore
    initialize_carries = None  # type: ignore
    sample_and_logprob = None  # type: ignore
    eval_logprob = None  # type: ignore
    batchify = None  # type: ignore
    unbatchify = None  # type: ignore


PyTree = Any


# ===========================================================================
# Runner state -- the carry threaded through the outer update scan
# ===========================================================================
class RunnerState(Any):  # pragma: no cover - documentation alias only
    """Logical description (the real carry is a plain tuple for scan-friendliness).

    The outer ``lax.scan`` carries:
      ``(actor_state, critic_state, env_state, last_obs, actor_carry,
         critic_carry, frozen_carry, opp_pool, last_done, update_idx, rng)``
    where ``frozen_carry`` is the dedicated recurrent memory of the frozen
    self-play opponent (kept separate so the snapshot actually has memory),
    documented field-by-field in :func:`make_train`.
    """


# ===========================================================================
# Optimizer / schedule helpers
# ===========================================================================
def _make_lr_schedule(cfg: Config) -> Callable[[int], float] | float:
    """Build a linear-anneal-to-zero LR schedule (or constant), per ``cfg``.

    PPO step count per update = ``update_epochs * num_minibatches``. We anneal
    over the *total* number of gradient steps so LR hits ~0 at the end of
    training, matching the PureJaxRL convention.
    """
    if not cfg.train.anneal_lr:
        return cfg.train.lr

    total_grad_steps = (
        cfg.train.num_updates * cfg.train.update_epochs * cfg.train.num_minibatches
    )

    def schedule(count: int) -> float:
        # `count` is the optax step counter (number of optimizer updates so far).
        frac = 1.0 - (count / total_grad_steps)
        # Clamp so a slightly-over count never yields a negative LR.
        return cfg.train.lr * jnp.clip(frac, 0.0, 1.0)

    return schedule


def _make_tx(cfg: Config) -> optax.GradientTransformation:
    """Adam with global-norm gradient clipping (CONTRACT §8 step 5)."""
    lr = _make_lr_schedule(cfg)
    return optax.chain(
        optax.clip_by_global_norm(cfg.train.max_grad_norm),
        optax.adam(learning_rate=lr, eps=1e-5),
    )


# ===========================================================================
# GAE (generalized advantage estimation)
# ===========================================================================
def _compute_gae(
    traj: Transition,
    last_value: jnp.ndarray,
    last_done: jnp.ndarray,
    gamma: float,
    gae_lambda: float,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute GAE advantages and value targets over a scanned trajectory.

    Runs a *reverse* ``lax.scan`` over the time axis. ``done`` masks the
    bootstrap across episode boundaries (auto-reset means the next obs after a
    terminal step belongs to a *new* episode, so its value must not leak back).

    Parameters
    ----------
    traj:
        A :class:`Transition` whose leaves have leading shape
        ``(num_steps, num_envs, A)`` (rewards/values) or ``(num_steps,
        num_envs)`` (done).
    last_value:
        ``(num_envs, A)`` -- bootstrap value of the observation *after* the last
        collected step.
    last_done:
        ``(num_envs,)`` -- done flag aligned with ``last_value``.
    gamma, gae_lambda:
        Discount and GAE trace-decay.

    Returns
    -------
    (advantages, targets):
        Both shape ``(num_steps, num_envs, A)``. ``targets = advantages + value``.
    """

    def _scan_fn(carry, transition: Transition):
        gae, next_value, next_done = carry
        # done has shape (num_envs,); broadcast to the per-agent value shape.
        not_done = 1.0 - next_done.astype(jnp.float32)
        not_done = not_done[..., None]  # (num_envs, 1) -> broadcast over A

        delta = (
            transition.reward
            + gamma * next_value * not_done
            - transition.value
        )
        gae = delta + gamma * gae_lambda * not_done * gae
        return (gae, transition.value, transition.done), gae

    init_gae = jnp.zeros_like(last_value)
    _, advantages = jax.lax.scan(
        _scan_fn,
        (init_gae, last_value, last_done),
        traj,
        reverse=True,
        unroll=16,
    )
    targets = advantages + traj.value
    return advantages, targets


# ===========================================================================
# Team / parameter-sharing utilities
# ===========================================================================
def _team_id_from_obs(actor_obs: Dict[str, jnp.ndarray]) -> jnp.ndarray:
    """Extract a per-agent team id from the ``self`` proprioception vector.

    Per CONTRACT §3.3 the self vector carries a 2-d team one-hot at indices
    ``[9:11]`` (hider, seeker). With ``shared_policy_per_team`` the *single*
    shared actor is conditioned on this team id (the one-hot is already part of
    its input), so no extra wiring is needed -- this helper exists for the
    self-play opponent-masking logic, which needs to know each agent's team to
    decide whose actions the frozen snapshot should override.

    Returns
    -------
    jnp.ndarray
        ``(..., A)`` int32 team id (0=hider, 1=seeker).
    """
    team_onehot = actor_obs["self"][..., 9:11]  # (..., A, 2)
    return jnp.argmax(team_onehot, axis=-1).astype(jnp.int32)


# ===========================================================================
# make_train -- builds the fully-jitted train(rng)
# ===========================================================================
def make_train(cfg: Config) -> Callable[[jnp.ndarray], Dict[str, Any]]:
    """Build a jitted ``train(rng)`` MAPPO function for the given config.

    Parameters
    ----------
    cfg:
        The fully-derived :class:`config.Config`.

    Returns
    -------
    callable
        ``train(rng) -> {"runner_state": ..., "metrics": ...}``, ready to be
        wrapped in ``jax.jit`` (it is internally jit-clean; the entrypoint jits
        it once). All loops are ``lax.scan``; there is no host transfer inside.

    Notes
    -----
    The function captures ``cfg`` and the constructed env/networks in its
    closure. Python-side control flow (building optimizers, choosing shapes)
    happens here, *before* tracing -- exactly the split the contract mandates
    (§9): "isolate any Python-side control flow in the trainer's non-jitted
    setup, not in env/model forward passes."
    """
    # ------------------------------------------------------------------ setup
    env = HideAndSeekEnv(cfg)  # single-env functional API; vmapped below
    vec_reset = make_vec_env_reset(env)
    vec_step = make_vec_env_step(env)

    actor = ActorRNN(cfg.model)
    critic = CriticRNN(cfg.model)

    tx = _make_tx(cfg)

    num_envs = cfg.train.num_envs
    num_steps = cfg.train.num_steps
    num_updates = cfg.train.num_updates
    num_minibatches = cfg.train.num_minibatches
    update_epochs = cfg.train.update_epochs
    A = cfg.env.max_agents

    # Per-update environment-interaction count, used to anneal & report.
    steps_per_update = num_envs * num_steps

    # ----------------------------------------------------------- the train fn
    def train(rng: jnp.ndarray) -> Dict[str, Any]:
        """Run MAPPO for ``cfg.train.num_updates`` updates. Fully jittable."""
        # ---- init params & optimizer states -----------------------------
        rng, actor_key, critic_key, reset_key = jax.random.split(rng, 4)

        # Build dummy inputs to initialize the recurrent networks. Shapes follow
        # the PureJaxRL ScannedGRU convention: a leading TIME axis (1) and a
        # batch axis (num_envs * A for the actor; num_envs for the critic).
        init_carry_actor, init_carry_critic = initialize_carries(
            cfg, batch_actor=num_envs * A, batch_critic=num_envs * A
        )

        # Reset envs to obtain a representative obs dict for param init.
        reset_keys = jax.random.split(reset_key, num_envs)
        env_state = vec_reset(reset_keys)
        obs = env_state.obs

        # Flatten the agent axis for the actor (batchify): (num_envs, A, ...) ->
        # (num_envs*A, ...). The critic consumes the per-env global obs.
        actor_obs = stack_actor_obs(obs)
        critic_obs = stack_critic_obs(obs)

        dummy_done = jnp.zeros((1, num_envs * A), dtype=bool)
        dummy_actor_in = (
            _add_time_and_batchify(actor_obs, A),
            dummy_done,
        )
        dummy_critic_in = (
            _add_time_and_broadcast_critic(critic_obs, A),
            dummy_done,
        )

        actor_params = actor.init(actor_key, init_carry_actor, dummy_actor_in)
        critic_params = critic.init(critic_key, init_carry_critic, dummy_critic_in)

        actor_state = TrainState.create(
            apply_fn=actor.apply, params=actor_params, tx=tx
        )
        critic_state = TrainState.create(
            apply_fn=critic.apply, params=critic_params, tx=tx
        )

        # ---- self-play opponent pool ------------------------------------
        # # historical self-play via ELO
        opp_pool = init_opponent_pool(
            actor_params, cfg.train.opponent_pool_size, cfg.train.elo_init
        )

        # ---- initial recurrent carries (per-env, per-agent) -------------
        actor_carry, critic_carry = initialize_carries(
            cfg, batch_actor=num_envs * A, batch_critic=num_envs * A
        )
        # # historical self-play via ELO
        # The frozen opponent needs its OWN recurrent memory: feeding it the live
        # actor's carry (and discarding its update) leaves it amnesiac. We thread
        # a dedicated frozen_carry through the rollout scan, initialized like the
        # live actor carry (same batch shape).
        frozen_carry = initialize_carries(
            cfg, batch_actor=num_envs * A, batch_critic=num_envs * A
        )[0]
        last_done = jnp.zeros((num_envs,), dtype=bool)

        runner_state = (
            actor_state,
            critic_state,
            env_state,
            obs,
            actor_carry,
            critic_carry,
            frozen_carry,
            opp_pool,
            last_done,
            jnp.asarray(0, jnp.int32),  # update_idx
            rng,
        )

        # ================================================================
        # ONE UPDATE = collect rollout -> GAE -> PPO epochs -> self-play push
        # ================================================================
        def _update_step(runner_state, _):
            (
                actor_state,
                critic_state,
                env_state,
                last_obs,
                actor_carry,
                critic_carry,
                frozen_carry,
                opp_pool,
                last_done,
                update_idx,
                rng,
            ) = runner_state

            # ---- decide self-play opponent for THIS update --------------
            # # historical self-play via ELO
            # With prob past_opponent_prob, a fraction of envs have the opponent
            # team driven by a frozen snapshot. We sample one snapshot per update
            # and assign a per-env boolean "use_frozen" mask; the frozen policy
            # then overrides the *seeker* team's actions on those envs.
            rng, opp_key, use_key, sample_key = jax.random.split(rng, 4)
            _, frozen_params, frozen_rating = sample_opponent(sample_key, opp_pool)
            # per-env coin flips, gated on the pool being non-empty (else never).
            pool_has_entries = jnp.any(opp_pool.valid)
            use_frozen_env = (
                jax.random.uniform(use_key, (num_envs,)) < cfg.train.past_opponent_prob
            ) & pool_has_entries  # (num_envs,) bool

            # ------------------------------------------------------------
            # INNER SCAN: collect `num_steps` transitions (the rollout)
            # ------------------------------------------------------------
            def _env_step(carry, _):
                (
                    actor_state,
                    critic_state,
                    env_state,
                    last_obs,
                    actor_carry,
                    critic_carry,
                    frozen_carry,
                    last_done,
                    rng,
                ) = carry

                # Separate randomness for the live sample, the frozen-opponent
                # sample, and the env step. The live and frozen policies MUST NOT
                # share a key (correlated sampling biases the self-play opponent).
                rng, act_key, frozen_key, step_key = jax.random.split(rng, 4)

                actor_obs = stack_actor_obs(last_obs)
                critic_obs = stack_critic_obs(last_obs)

                # done broadcast to the (num_envs*A,) actor batch and to the
                # (num_envs*A,) critic batch. The ScannedGRU resets its carry on
                # done (object-permanence memory cleared at episode boundary).
                done_actor = jnp.repeat(last_done, A)            # (num_envs*A,)
                ac_in = (
                    _add_time_and_batchify(actor_obs, A),
                    done_actor[None, :],                          # (1, num_envs*A)
                )
                cr_in = (
                    _add_time_and_broadcast_critic(critic_obs, A),
                    done_actor[None, :],
                )

                # ---- LIVE actor forward (decentralized, local obs) ------
                new_actor_carry, pi = actor.apply(
                    actor_state.params, actor_carry, ac_in
                )
                # ---- FROZEN opponent forward (self-play) ----------------
                # # historical self-play via ELO
                # Run the frozen snapshot on the SAME inputs but with its OWN
                # recurrent carry (so it actually has memory); thread the updated
                # carry forward. We then splice its actions in only for
                # (use_frozen_env AND opponent-team agents).
                new_frozen_carry, pi_frozen = actor.apply(
                    frozen_params, frozen_carry, ac_in
                )

                # ---- sample actions + logprob from the LIVE policy ------
                # sample_and_logprob returns (action, logprob, entropy); we keep
                # action + logprob for learning and discard the per-sample entropy
                # (entropy is recomputed at update time via _policy_entropy).
                action, log_prob, _ = sample_and_logprob(pi, act_key)
                # Frozen opponent: independent key, we only need its action.
                action_frozen, _, _ = sample_and_logprob(pi_frozen, frozen_key)

                # ---- CTDE critic forward (centralized, global obs) ------
                new_critic_carry, value = critic.apply(
                    critic_state.params, critic_carry, cr_in
                )

                # ---- splice frozen-opponent actions where applicable ----
                # team id per (env, agent): 0=hider (learner), 1=seeker (opp).
                team_id = _team_id_from_obs(actor_obs)            # (num_envs, A)
                is_opponent = team_id == 1                        # seekers
                use_env = use_frozen_env[:, None]                 # (num_envs, 1)
                override = is_opponent & use_env                  # (num_envs, A)

                action = _select_actions(action, action_frozen, override, num_envs, A)

                # ---- step the vectorized env (auto-reset on done) -------
                # vec_step returns (autoreset_state, terminal_done): the state has
                # the FRESH episode's obs on terminal envs, while terminal_done is
                # the genuine PRE-reset done. We must record terminal_done (not
                # the auto-reset state's clobbered, always-False done) so GAE cuts
                # the bootstrap at episode boundaries instead of leaking advantage
                # across episodes.
                step_keys = jax.random.split(step_key, num_envs)
                new_env_state, terminal_done = vec_step(step_keys, env_state, action)

                # ---- assemble the Transition (pre-reset values) ---------
                transition = Transition(
                    done=terminal_done,                           # (num_envs,) PRE-reset
                    action=action,
                    value=_unbatch_value(value, num_envs, A),     # (num_envs, A)
                    reward=new_env_state.reward,                  # (num_envs, A)
                    log_prob=_unbatch_value(log_prob, num_envs, A),
                    obs=actor_obs,
                    global_obs=critic_obs,
                    avail=actor_obs["agent_active"],              # (num_envs, A)
                    info=new_env_state.info,
                )

                new_carry = (
                    actor_state,
                    critic_state,
                    new_env_state,
                    new_env_state.obs,
                    new_actor_carry,
                    new_critic_carry,
                    new_frozen_carry,
                    terminal_done,
                    rng,
                )
                return new_carry, transition

            inner_carry = (
                actor_state,
                critic_state,
                env_state,
                last_obs,
                actor_carry,
                critic_carry,
                frozen_carry,
                last_done,
                rng,
            )
            inner_carry, traj_batch = jax.lax.scan(
                _env_step, inner_carry, None, length=num_steps
            )
            (
                actor_state,
                critic_state,
                env_state,
                last_obs,
                actor_carry,
                critic_carry,
                frozen_carry,
                last_done,
                rng,
            ) = inner_carry

            # ---- bootstrap value for GAE (one more critic forward) ------
            critic_obs = stack_critic_obs(last_obs)
            done_actor = jnp.repeat(last_done, A)
            cr_in = (
                _add_time_and_broadcast_critic(critic_obs, A),
                done_actor[None, :],
            )
            _, last_value = critic.apply(critic_state.params, critic_carry, cr_in)
            last_value = _unbatch_value(last_value, num_envs, A)  # (num_envs, A)

            advantages, targets = _compute_gae(
                traj_batch,
                last_value,
                last_done,
                cfg.train.gamma,
                cfg.train.gae_lambda,
            )

            # ------------------------------------------------------------
            # PPO UPDATE: update_epochs x num_minibatches optax steps
            # ------------------------------------------------------------
            def _update_epoch(update_state, _):
                actor_state, critic_state, rng = update_state
                rng, perm_key = jax.random.split(rng)

                # Flatten (num_steps, num_envs) -> a single batch axis, keep the
                # agent axis A so the recurrent reset logic is preserved per the
                # JaxMARL "flatten envs, keep time for the GRU" pattern. Here we
                # treat each (step, env) pair as an independent sequence element
                # for the minibatch (the GRU was already unrolled during
                # collection; for the update we re-run it over the stored
                # per-timestep inputs with stored resets).
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((num_steps * num_envs,) + x.shape[2:]), batch
                )
                perm = jax.random.permutation(perm_key, num_steps * num_envs)
                shuffled = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, perm, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: x.reshape((num_minibatches, -1) + x.shape[1:]),
                    shuffled,
                )

                def _minibatch_update(carry, minibatch):
                    actor_state, critic_state = carry
                    traj_mb, adv_mb, tgt_mb = minibatch

                    # --- normalize advantages within minibatch (PPO std) ---
                    # mask padded agents out of the statistics.
                    mask = traj_mb.avail.astype(jnp.float32)          # (mb, A)
                    adv_mean = _masked_mean(adv_mb, mask)
                    adv_std = jnp.sqrt(_masked_mean((adv_mb - adv_mean) ** 2, mask) + 1e-8)
                    adv_norm = (adv_mb - adv_mean) / adv_std

                    # ===== actor (policy) loss =========================
                    def _actor_loss_fn(actor_params):
                        # Re-evaluate the policy on the stored obs. The minibatch
                        # element is a single (step,env) snapshot, so we feed a
                        # length-1 time axis with the stored done as the reset.
                        ac_in = (
                            _add_time_and_batchify(traj_mb.obs, A),
                            jnp.repeat(traj_mb.done, A)[None, :],
                        )
                        carry0 = initialize_carries(
                            cfg,
                            batch_actor=traj_mb.done.shape[0] * A,
                            batch_critic=traj_mb.done.shape[0] * A,
                        )[0]
                        _, pi = actor.apply(actor_params, carry0, ac_in)

                        # eval_logprob returns (logprob, entropy); we score the
                        # stored action and recompute entropy separately below via
                        # _policy_entropy(pi) for the entropy-bonus term.
                        log_prob, _ = eval_logprob(pi, traj_mb.action)
                        log_prob = _unbatch_value(
                            log_prob, traj_mb.done.shape[0], A
                        )  # (mb, A)
                        entropy = _policy_entropy(pi)
                        entropy = _unbatch_value(entropy, traj_mb.done.shape[0], A)

                        # clipped surrogate (CONTRACT §8 step 5)
                        ratio = jnp.exp(log_prob - traj_mb.log_prob)
                        unclipped = ratio * adv_norm
                        clipped = (
                            jnp.clip(
                                ratio,
                                1.0 - cfg.train.clip_eps,
                                1.0 + cfg.train.clip_eps,
                            )
                            * adv_norm
                        )
                        # PPO maximizes min(unclipped, clipped) => minimize -min.
                        policy_loss = -jnp.minimum(unclipped, clipped)
                        policy_loss = _masked_mean(policy_loss, mask)

                        ent_mean = _masked_mean(entropy, mask)
                        # entropy is MAXIMIZED: subtract it from the loss.
                        total = policy_loss - cfg.train.ent_coef * ent_mean

                        aux = {
                            "policy_loss": policy_loss,
                            "entropy": ent_mean,
                            # approx_kl = E[new_logprob - old_logprob] (PureJaxRL/
                            # cleanRL convention): positive when the updated policy
                            # has diverged from the behaviour policy.
                            "approx_kl": _masked_mean(
                                (log_prob - traj_mb.log_prob), mask
                            ),
                            "clip_frac": _masked_mean(
                                (jnp.abs(ratio - 1.0) > cfg.train.clip_eps).astype(
                                    jnp.float32
                                ),
                                mask,
                            ),
                        }
                        return total, aux

                    # ===== critic (value) loss =========================
                    def _critic_loss_fn(critic_params):
                        cr_in = (
                            _add_time_and_broadcast_critic(traj_mb.global_obs, A),
                            jnp.repeat(traj_mb.done, A)[None, :],
                        )
                        carry0 = initialize_carries(
                            cfg,
                            batch_actor=traj_mb.done.shape[0] * A,
                            batch_critic=traj_mb.done.shape[0] * A,
                        )[1]
                        _, value = critic.apply(critic_params, carry0, cr_in)
                        value = _unbatch_value(value, traj_mb.done.shape[0], A)

                        # PPO value clipping around the old value estimate.
                        value_clipped = traj_mb.value + jnp.clip(
                            value - traj_mb.value,
                            -cfg.train.clip_eps,
                            cfg.train.clip_eps,
                        )
                        v_losses = (value - tgt_mb) ** 2
                        v_losses_clipped = (value_clipped - tgt_mb) ** 2
                        value_loss = 0.5 * jnp.maximum(v_losses, v_losses_clipped)
                        value_loss = _masked_mean(value_loss, mask)
                        return cfg.train.vf_coef * value_loss, {"value_loss": value_loss}

                    (actor_loss, actor_aux), actor_grads = jax.value_and_grad(
                        _actor_loss_fn, has_aux=True
                    )(actor_state.params)
                    (critic_loss, critic_aux), critic_grads = jax.value_and_grad(
                        _critic_loss_fn, has_aux=True
                    )(critic_state.params)

                    actor_state = actor_state.apply_gradients(grads=actor_grads)
                    critic_state = critic_state.apply_gradients(grads=critic_grads)

                    losses = {
                        "total_loss": actor_loss + critic_loss,
                        **actor_aux,
                        **critic_aux,
                    }
                    return (actor_state, critic_state), losses

                (actor_state, critic_state), losses = jax.lax.scan(
                    _minibatch_update, (actor_state, critic_state), minibatches
                )
                return (actor_state, critic_state, rng), losses

            update_state = (actor_state, critic_state, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, length=update_epochs
            )
            actor_state, critic_state, rng = update_state

            # ---- self-play snapshot push every snapshot_interval_updates --
            # # historical self-play via ELO
            # Branch-free conditional push: compute a "pushed" pool and select.
            do_push = (
                (update_idx + 1) % cfg.train.snapshot_interval_updates
            ) == 0
            pushed_pool = push_snapshot(
                opp_pool,
                actor_state.params,
                # Enter the pool at the current ELO (env-side win/loss feeds the
                # EloManager outside the jit; here we seed at elo_init-relative).
                jnp.asarray(cfg.train.elo_init, jnp.float32),
            )
            opp_pool = _select_pool(opp_pool, pushed_pool, do_push)

            # ---- aggregate metrics (no host transfer) -------------------
            metrics = _aggregate_metrics(traj_batch, loss_info)
            metrics["frozen_opponent_rating"] = frozen_rating
            metrics["frac_frozen_envs"] = jnp.mean(use_frozen_env.astype(jnp.float32))
            metrics["env_steps"] = (update_idx + 1) * steps_per_update

            runner_state = (
                actor_state,
                critic_state,
                env_state,
                last_obs,
                actor_carry,
                critic_carry,
                frozen_carry,
                opp_pool,
                last_done,
                update_idx + 1,
                rng,
            )
            return runner_state, metrics

        # =============== OUTER SCAN over num_updates ====================
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, length=num_updates
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


# ===========================================================================
# Small pure helpers (kept module-level so they are easy to unit-test)
# ===========================================================================
def _add_time_and_batchify(
    actor_obs: Dict[str, jnp.ndarray], A: int
) -> Dict[str, jnp.ndarray]:
    """Flatten the agent axis and add a length-1 leading time axis for the GRU.

    ``(num_envs, A, ...) -> (1, num_envs*A, ...)``. The leading 1 is the TIME
    axis the ScannedGRU scans over; we collect/evaluate one step at a time, so
    time-length is 1. Uses :func:`utils.pytree.batchify` for the agent flatten
    when available, else an equivalent reshape (kept faithful to the contract:
    batchify is the *only sanctioned* flatten, but we fall back so this helper is
    importable in isolation).
    """
    def _flat(x: jnp.ndarray) -> jnp.ndarray:
        n_envs = x.shape[0]
        x = x.reshape((n_envs * A,) + x.shape[2:])
        return x[None, ...]  # add time axis

    if batchify is not None:
        # batchify collapses the agent axis; we still add the time axis here.
        flat = batchify(actor_obs)
        return jax.tree_util.tree_map(lambda v: v[None, ...], flat)
    return jax.tree_util.tree_map(_flat, actor_obs)


def _add_time_and_broadcast_critic(
    critic_obs: Dict[str, jnp.ndarray], A: int
) -> Dict[str, jnp.ndarray]:
    """Shape critic (global) obs to ``(1, num_envs*A, ...)`` by broadcasting over A.

    The global tokens are per-env (no agent axis), but the critic carry runs at
    the ``num_envs*A`` batch for index-alignment with the actor's per-agent
    advantages (each agent gets its own value head input). We therefore tile the
    per-env global obs across the ``A`` agents and add the time axis.
    """
    def _bcast(x: jnp.ndarray) -> jnp.ndarray:
        n_envs = x.shape[0]
        # (num_envs, ...) -> (num_envs, A, ...) -> (num_envs*A, ...)
        x = jnp.broadcast_to(x[:, None], (n_envs, A) + x.shape[1:])
        x = x.reshape((n_envs * A,) + x.shape[2:])
        return x[None, ...]

    return jax.tree_util.tree_map(_bcast, critic_obs)


def _unbatch_value(x: jnp.ndarray, num_envs: int, A: int) -> jnp.ndarray:
    """Undo the time+agent flatten: ``(1, num_envs*A) -> (num_envs, A)``.

    Mirrors :func:`utils.pytree.unbatchify`. Squeezes the length-1 time axis and
    splits the flat batch back into ``(num_envs, A)``.
    """
    x = jnp.squeeze(x, axis=0) if x.ndim >= 1 and x.shape[0] == 1 else x
    return x.reshape((num_envs, A) + x.shape[1:])


def _select_actions(
    action: Dict[str, jnp.ndarray],
    action_frozen: Dict[str, jnp.ndarray],
    override: jnp.ndarray,
    num_envs: int,
    A: int,
) -> Dict[str, jnp.ndarray]:
    """Splice frozen-opponent actions where ``override`` (per-agent) is True.

    ``override`` has shape ``(num_envs, A)``. Each action component is selected
    element-wise with ``jnp.where`` (broadcasting the mask over the action-dim
    axis). Pure, jit-safe -- no Python branching on the traced mask.
    """
    out: Dict[str, jnp.ndarray] = {}
    for k in action:
        live = action[k]      # (num_envs, A, d)
        froz = action_frozen[k]
        mask = override[..., None]  # (num_envs, A, 1)
        out[k] = jnp.where(mask, froz, live)
    return out


def _masked_mean(x: jnp.ndarray, mask: jnp.ndarray) -> jnp.ndarray:
    """Mean of ``x`` over entries where ``mask`` > 0 (avoids padded-agent leakage)."""
    mask = mask.astype(x.dtype)
    return jnp.sum(x * mask) / jnp.clip(jnp.sum(mask), 1.0, None)


def _policy_entropy(pi: Any) -> jnp.ndarray:
    """Total entropy of the hybrid policy = move-Gaussian entropy + sum of cats.

    ``pi`` is ``(move_dist, [cat_dist, ...])`` per CONTRACT §5. We sum the
    continuous entropy (already summed over the 3 move dims by the dist) and each
    categorical's entropy. Returns shape ``(batch,)`` matching the dists' batch.
    """
    move_dist, interact_dists = pi
    ent = move_dist.entropy()
    # move_dist.entropy() may return per-dim entropy; sum any trailing axis.
    if ent.ndim > 1:
        ent = jnp.sum(ent, axis=-1)
    for d in interact_dists:
        ent = ent + d.entropy()
    return ent


def _select_pool(
    old: OpponentPool, pushed: OpponentPool, do_push: jnp.ndarray
) -> OpponentPool:
    """Branch-free select between the pre- and post-push pool pytrees."""
    return jax.tree_util.tree_map(
        lambda a, b: jnp.where(do_push, b, a), old, pushed
    )


def _aggregate_metrics(
    traj_batch: Transition, loss_info: Dict[str, jnp.ndarray]
) -> Dict[str, jnp.ndarray]:
    """Reduce per-step env info + per-minibatch losses to scalar metrics.

    Everything is a mean over the rollout/update axes -- still on device, so the
    logger can pull a single small array per metric without stalling the loop.
    """
    metrics: Dict[str, jnp.ndarray] = {}
    # env-side info dict: average each scalar metric over (num_steps, num_envs).
    for k, v in traj_batch.info.items():
        metrics[f"env/{k}"] = jnp.mean(v)
    metrics["env/mean_reward"] = jnp.mean(traj_batch.reward)
    # loss_info leaves have shape (update_epochs, num_minibatches); mean them.
    for k, v in loss_info.items():
        metrics[f"loss/{k}"] = jnp.mean(v)
    return metrics


__all__ = ["make_train", "RunnerState"]
