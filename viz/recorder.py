"""
viz/recorder.py -- export REAL Hide & Seek 2.0 rollouts to the viewer format.

This is the bridge from a live JAX episode (``envs.HideAndSeekEnv`` /
:class:`envs.state.State`) to the on-disk ``hns2-traj`` schema consumed by the
3D viewer. Unlike :mod:`viz.make_demo_trajectory` (pure stdlib), this module MAY
use numpy/jax -- but the imports are kept **lazy** so that importing
``viz.recorder`` never fails on a machine without jax installed. Only the methods
that actually pull device arrays touch numpy/jax.

Two entry points
----------------
:class:`TrajectoryRecorder`
    Streaming recorder. Construct it with the project ``Config``, call
    :meth:`record_step` once per environment step with the post-step
    :class:`State` (it host-pulls just the fields the viewer needs), then
    :meth:`save` to validate + write a trajectory file. Use this when you already
    have a rollout loop and just want to tap it.

:func:`rollout_to_trajectory`
    Convenience helper that runs a short rollout for you (single, un-vmapped
    environment) using a provided ``actor`` callable and returns the finished
    trajectory document. Use this for a one-liner "give me a viewable episode".

Field mapping (env CONTRACT -> schema FRAME_ENT_KEYS)
-----------------------------------------------------
Per-entity, id-aligned (agents occupy entity rows ``[0:A]``):

================  =========================================================
schema key        env source
================  =========================================================
``x, y, z``       ``physics.pos[:, 0/1/2]``  (z = elevation)
``h``             ``physics.heading``        (agents only; 0 for props)
``a``             ``physics.active`` AND (for walls) ``wall_hp > 0`` AND
                  (for doors) ``door_progress < 1``  -- "exists/closed"
``lk``            ``game.locked``
``hd`` / ``hb``   derived from ``game.holding`` (agent -> held entity id):
                  a prop is held iff some agent's ``holding`` equals its id;
                  ``hb`` is that agent's id (else -1)
``no``            ``game.emitted_noise``
``dc``            ``game.decoy_timer > 0``   (actively emitting)
``gr``            ``physics.grounded``
``st``            ``game.stamina / stamina_max``  (agents; -1 for props)
``sn``            seen-by-opposing-team this frame (from the env visibility
                  mask if exposed in ``info``/state, else a fallback heuristic)
================  =========================================================

Per-frame scalars:

================  =========================================================
schema key        env source
================  =========================================================
``phase``         ``game.phase``  (0 -> "prep", 1 -> "main")
``sh`` / ``ss``   cumulative sum of ``info["hiders_reward"]`` /
                  ``info["seekers_reward"]`` (falls back to ``state.reward``
                  split by team if those keys are absent)
``seen_any``      ``info["seen_frac"] > 0``  (any seeker sees any hider)
``fog``           ``game.fog_pos`` (xy) paired with ``cfg.env.fog_radius``
================  =========================================================

The static ``entities`` table is read once (from the first recorded state):
``type`` from ``physics.type_id`` -> :data:`config.ENTITY_TYPES`, ``team`` from
``game.team`` (props -> -1), ``size``/``mass`` from ``physics``, ``is_decoy`` from
the PRIVILEGED ``game.is_decoy`` (god-view truth).

Usage
-----
::

    from viz.recorder import TrajectoryRecorder
    rec = TrajectoryRecorder(cfg)
    state = env.reset(key)
    rec.record_step(state)                 # record the initial frame
    for _ in range(cfg.env.max_steps):
        action = actor(state)              # your policy
        state = env.step(state, action)
        rec.record_step(state)
    rec.save("viz/web/trajectories/run.json", title="trained agents", seed=0)
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

from viz import schema


# --------------------------------------------------------------------------- #
# Lazy numpy import. We never import numpy at module load -- only when a host
# pull actually happens -- so importing viz.recorder is safe without jax/numpy.
# --------------------------------------------------------------------------- #
def _np():
    """Return the numpy module, importing it lazily.

    Raises a clear :class:`RuntimeError` (rather than a bare ``ImportError`` from
    deep inside) if numpy is unavailable, since the recorder is fundamentally a
    device->host bridge and cannot function without it.
    """
    try:
        import numpy as np  # noqa: WPS433 (intentional lazy import)
        return np
    except ImportError as exc:  # pragma: no cover - exercised only without numpy
        raise RuntimeError(
            "viz.recorder needs numpy to pull device arrays to host. "
            "Install it (`pip install numpy`) or use viz.make_demo_trajectory "
            "for a pure-stdlib synthetic episode."
        ) from exc


def _to_host(x: Any):
    """Convert a (possibly jax) array to a host numpy array.

    Works for jax DeviceArrays, numpy arrays, and Python scalars/lists alike --
    ``numpy.asarray`` copies device arrays to host transparently. Kept tiny so the
    per-step host-pull cost is just the unavoidable copy.
    """
    np = _np()
    return np.asarray(x)


def _scalar(x: Any) -> float:
    """Pull a 0-d / size-1 array (or scalar) to a Python float."""
    return float(_to_host(x).reshape(-1)[0])


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #
class TrajectoryRecorder:
    """Collect per-step host data from a single environment and emit a trajectory.

    The recorder operates on a SINGLE (un-vmapped) environment's :class:`State`.
    If you are running a vmapped batch, index out one environment before passing
    its state in (e.g. ``jax.tree_util.tree_map(lambda a: a[env_idx], state)``).

    Parameters
    ----------
    cfg:
        The project :class:`config.Config`. We read ``cfg.env`` for arena size,
        ``dt``, ``max_steps``, ``prep_steps``, ``max_agents``, ``max_entities``,
        ``fog_radius`` and ``stamina_max``.

    Notes
    -----
    The static ``entities`` table is captured on the FIRST :meth:`record_step`.
    Every subsequent frame is validated to be id-aligned by
    :func:`viz.schema.validate_trajectory` at save time.
    """

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.env_cfg = cfg.env
        self.A = int(self.env_cfg.max_agents)
        self.E = int(self.env_cfg.max_entities)
        self.fog_radius = float(getattr(self.env_cfg, "fog_radius", 2.5))
        self.stamina_max = float(getattr(self.env_cfg, "stamina_max", 100.0))

        self._entities: Optional[List[Dict[str, Any]]] = None
        self._frames: List[Dict[str, Any]] = []
        # Running cumulative scores (the env reports per-step team rewards; the
        # viewer wants the running total).
        self._sh = 0.0
        self._ss = 0.0

    # --------------------------------------------------------------------- #
    # Static entity table (captured once).
    # --------------------------------------------------------------------- #
    def _capture_entities(self, state: Any) -> List[Dict[str, Any]]:
        """Build the static ``entities`` table from the first state seen.

        ``type`` comes from ``physics.type_id`` (index into ENTITY_TYPES), ``team``
        from ``game.team`` for agent rows (props -> -1), ``size``/``mass`` from
        ``physics``, and ``is_decoy`` from the PRIVILEGED ``game.is_decoy``.
        """
        physics, game = state.physics, state.game
        type_id = _to_host(physics.type_id).astype(int)
        size = _to_host(physics.size).astype(float)
        mass = _to_host(physics.mass).astype(float)
        is_decoy = _to_host(game.is_decoy).astype(bool)
        team = _to_host(game.team).astype(int)  # (A,)

        entities: List[Dict[str, Any]] = []
        for i in range(self.E):
            tname = schema.ENTITY_TYPES[int(type_id[i])]
            # Agent rows (first A) carry a real team; props are team -1.
            t = int(team[i]) if i < self.A else schema.TEAM_NONE
            entities.append(schema.make_entity_meta(
                id=i, type=tname, team=t,
                size=float(size[i]), mass=float(mass[i]),
                is_decoy=bool(is_decoy[i]),
            ))
        return entities

    # --------------------------------------------------------------------- #
    # Per-step recording.
    # --------------------------------------------------------------------- #
    def record_step(self, state: Any) -> None:
        """Record one frame from a post-step (or post-reset) :class:`State`.

        Host-pulls only the fields the viewer needs and appends a schema-shaped
        frame. The static entity table is captured on the first call.
        """
        if self._entities is None:
            self._entities = self._capture_entities(state)

        np = _np()
        physics, game = state.physics, state.game

        pos = _to_host(physics.pos).astype(float)            # (E, 3)
        heading = _to_host(physics.heading).astype(float)    # (E,)
        active = _to_host(physics.active).astype(bool)       # (E,)
        grounded = _to_host(physics.grounded).astype(bool)   # (E,)
        type_id = _to_host(physics.type_id).astype(int)      # (E,)

        team = _to_host(game.team).astype(int)               # (A,)
        stamina = _to_host(game.stamina).astype(float)       # (A,)
        holding = _to_host(game.holding).astype(int)         # (A,)
        locked = _to_host(game.locked).astype(bool)          # (E,)
        emitted_noise = _to_host(game.emitted_noise).astype(float)  # (E,)
        decoy_timer = _to_host(game.decoy_timer).astype(int)        # (E,)
        wall_hp = _to_host(game.wall_hp).astype(float)              # (E,)
        door_progress = _to_host(game.door_progress).astype(float)  # (E,)
        fog_pos = _to_host(game.fog_pos).astype(float)              # (Pf, 2)
        phase_i = int(_scalar(game.phase))
        step_i = int(_scalar(game.step))
        phase = "main" if phase_i == 1 else "prep"

        id_hider = schema.ENTITY_TYPES.index("hider")
        id_seeker = schema.ENTITY_TYPES.index("seeker")
        id_wall = schema.ENTITY_TYPES.index("wall")
        id_door = schema.ENTITY_TYPES.index("door")

        # --- held-by: invert game.holding (agent -> held entity) to a per-prop
        # holder map. holding[a] == eid means agent a holds entity eid.
        held_by = {}
        for a in range(self.A):
            tgt = int(holding[a])
            if tgt >= 0 and bool(active[a]):
                held_by[tgt] = a

        # --- "seen" mask: prefer an env-provided per-entity visibility flag if
        # the info/state exposes one; otherwise fall back to seen_frac applied to
        # the hiders as a whole (so the viewer still highlights *something* when a
        # hider is seen). This keeps the recorder robust to env-version drift.
        sn_mask = self._seen_mask(state, type_id, active, id_hider, id_seeker)

        # --- frame scalars: scores + seen_any.
        seen_any, hr, sr = self._frame_scalars(state)
        self._sh += hr
        self._ss += sr

        ent: List[Dict[str, Any]] = []
        for i in range(self.E):
            tname = schema.ENTITY_TYPES[int(type_id[i])]
            is_agent = i < self.A and tname in ("hider", "seeker")

            # active / existence: base mask, refined for destructibles & doors.
            a_flag = 1 if bool(active[i]) else 0
            if a_flag and int(type_id[i]) == id_wall and i < len(wall_hp):
                if float(wall_hp[i]) <= 0.0:
                    a_flag = 0   # broken wall -> "gone"
            if a_flag and int(type_id[i]) == id_door and i < len(door_progress):
                if float(door_progress[i]) >= 1.0:
                    a_flag = 0   # fully-open door -> path is clear

            # held flags.
            holder = held_by.get(i, -1)
            hd = 1 if holder >= 0 else 0

            # stamina (agents only, normalized 0..1; props -> -1).
            if is_agent:
                st = float(stamina[i]) / self.stamina_max if i < len(stamina) else -1.0
                st = min(1.0, max(0.0, st))
                h = float(heading[i])
            else:
                st = -1.0
                h = 0.0

            ent.append(schema.make_frame_ent(
                id=i,
                x=float(pos[i, 0]), y=float(pos[i, 1]), z=float(pos[i, 2]),
                h=h,
                a=a_flag,
                lk=1 if bool(locked[i]) else 0,
                hd=hd,
                hb=int(holder),
                no=float(emitted_noise[i]) if i < len(emitted_noise) else 0.0,
                dc=1 if (i < len(decoy_timer) and int(decoy_timer[i]) > 0) else 0,
                gr=1 if bool(grounded[i]) else 0,
                st=st,
                sn=1 if sn_mask[i] else 0,
            ))

        fog = [
            [float(fog_pos[k, 0]), float(fog_pos[k, 1]), self.fog_radius]
            for k in range(fog_pos.shape[0])
        ]

        self._frames.append(schema.make_frame(
            t=step_i, phase=phase, sh=self._sh, ss=self._ss,
            seen_any=bool(seen_any), fog=fog, ent=ent,
        ))

    # --------------------------------------------------------------------- #
    # Frame-scalar + visibility derivation (robust to env-version drift).
    # --------------------------------------------------------------------- #
    def _frame_scalars(self, state: Any):
        """Return ``(seen_any, hider_reward, seeker_reward)`` for this frame.

        Prefers the env ``info`` dict keys (``seen_frac``, ``hiders_reward``,
        ``seekers_reward``); if those are absent, falls back to splitting
        ``state.reward`` by team. ``seen_any`` is ``seen_frac > 0`` (or, in the
        fallback, "any hider currently has negative reward", since the env pays a
        hider -1 exactly when it is seen).
        """
        info = getattr(state, "info", None) or {}
        np = _np()

        # seen_frac / seen_any.
        if "seen_frac" in info:
            seen_any = _scalar(info["seen_frac"]) > 0.0
        else:
            # Fallback: derive from the team rewards below (handled after).
            seen_any = None

        # Per-step team rewards.
        if "hiders_reward" in info and "seekers_reward" in info:
            hr = _scalar(info["hiders_reward"])
            sr = _scalar(info["seekers_reward"])
        else:
            # Split state.reward by team: hiders are team 0, seekers team 1.
            reward = _to_host(getattr(state, "reward")).astype(float)  # (A,)
            team = _to_host(state.game.team).astype(int)               # (A,)
            active = _to_host(state.physics.active[: self.A]).astype(bool)
            hmask = (team == 0) & active
            smask = (team == 1) & active
            hr = float(reward[hmask].sum()) if hmask.any() else 0.0
            sr = float(reward[smask].sum()) if smask.any() else 0.0

        if seen_any is None:
            # The env pays a seen hider -1 in the main phase; treat any negative
            # hider reward as "a hider is seen".
            seen_any = hr < 0.0

        return bool(seen_any), float(hr), float(sr)

    def _seen_mask(self, state: Any, type_id, active, id_hider, id_seeker) -> List[bool]:
        """Return a per-entity "seen by the opposing team this frame" mask.

        We look for an explicit per-entity visibility signal in ``state.info``
        under a few likely keys (``seen_mask`` / ``vis_mask`` / ``seen_by_opp``);
        if found and shaped per-entity, it is used directly. Otherwise we fall
        back to: if any seeker sees any hider this frame (``seen_frac > 0`` or the
        reward-derived flag), mark all active hiders as seen. This guarantees the
        viewer highlights the exposed team even on env versions that don't export
        a per-entity mask.
        """
        np = _np()
        info = getattr(state, "info", None) or {}
        mask = [False] * self.E

        for key in ("seen_mask", "vis_mask", "seen_by_opp", "seen"):
            if key in info:
                try:
                    arr = _to_host(info[key])
                except Exception:  # pragma: no cover - defensive
                    continue
                flat = arr.reshape(-1)
                if flat.shape[0] >= self.E:
                    return [bool(flat[i]) for i in range(self.E)]

        # Fallback: any-hider-seen -> highlight all active hiders.
        seen_any, _hr, _sr = self._frame_scalars(state)
        if seen_any:
            for i in range(self.E):
                if int(type_id[i]) == id_hider and bool(active[i]):
                    mask[i] = True
        return mask

    # --------------------------------------------------------------------- #
    # Build / save.
    # --------------------------------------------------------------------- #
    def build(self, title: str = "Hide & Seek 2.0 rollout", seed: int = 0) -> Dict[str, Any]:
        """Assemble (but do not write) the trajectory document.

        Raises ``RuntimeError`` if no frames were recorded.
        """
        if self._entities is None or not self._frames:
            raise RuntimeError("nothing recorded: call record_step() before build()/save()")
        meta = {
            "title": str(title),
            "seed": int(seed),
            "arena_size": float(self.env_cfg.arena_size),
            "dt": float(self.env_cfg.dt),
            "max_steps": int(self.env_cfg.max_steps),
            "prep_steps": int(self.env_cfg.prep_steps),
            "entity_types": list(schema.ENTITY_TYPES),
            "max_agents": int(self.A),
            "max_entities": int(self.E),
        }
        return schema.make_trajectory(meta, self._entities, self._frames)

    def save(self, path: str, title: str = "Hide & Seek 2.0 rollout", seed: int = 0) -> Dict[str, Any]:
        """Validate and write the recorded trajectory to ``path``.

        Returns the document (handy for inspection/testing). Raises ``ValueError``
        via :func:`viz.schema.save_trajectory` if validation fails.
        """
        import os

        doc = self.build(title=title, seed=seed)
        os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
        schema.save_trajectory(doc, path)
        return doc


# --------------------------------------------------------------------------- #
# One-shot rollout helper.
# --------------------------------------------------------------------------- #
def rollout_to_trajectory(
    env: Any,
    params: Any,
    actor: Callable[..., Dict[str, Any]],
    cfg: Any,
    key: Any,
    n_steps: Optional[int] = None,
    title: str = "Hide & Seek 2.0 rollout",
    seed: int = 0,
) -> Dict[str, Any]:
    """Run a short SINGLE-environment rollout and return a trajectory document.

    This is the batteries-included path: it resets ``env`` once, steps it
    ``n_steps`` times driving actions through ``actor``, recording every frame,
    and returns the finished (validated-shaped) document. jax is imported lazily
    here so importing this module stays dependency-free.

    Parameters
    ----------
    env:
        A constructed :class:`envs.HideAndSeekEnv` (UN-vmapped; this helper drives
        one environment).
    params:
        Policy parameters passed through to ``actor`` (opaque to the recorder).
    actor:
        Callable producing an action dict from ``(params, state, key)`` -- it must
        return the env action dict ``{"move": (A,3), "interact": (A,3)}``. Its exact
        signature is flexible: we try ``actor(params, state, subkey)`` and fall back
        to ``actor(state)`` so a simple closure works too.
    cfg:
        The project :class:`config.Config` (drives the recorder + horizon default).
    key:
        A ``jax.random.PRNGKey``.
    n_steps:
        Number of steps to roll out; defaults to ``cfg.env.max_steps``.
    title, seed:
        Stored in ``meta`` for the viewer.

    Returns
    -------
    The trajectory document (``hns2-traj`` v1). Write it with
    :func:`viz.schema.save_trajectory` or
    :func:`viz.schema.save_trajectory`-equivalent.

    Notes
    -----
    For a vmapped/batched rollout, prefer running your own loop and feeding a
    single indexed env's state into :class:`TrajectoryRecorder`. This helper
    deliberately keeps to one env so the mapping stays unambiguous.
    """
    import jax  # lazy: only needed when actually rolling out

    if n_steps is None:
        n_steps = int(cfg.env.max_steps)

    recorder = TrajectoryRecorder(cfg)
    key, reset_key = jax.random.split(key)
    state = env.reset(reset_key)
    recorder.record_step(state)  # initial frame

    for _ in range(n_steps):
        key, sub = jax.random.split(key)
        try:
            action = actor(params, state, sub)
        except TypeError:
            # Allow a simpler closure form: actor(state).
            action = actor(state)
        state = env.step(state, action)
        recorder.record_step(state)
        # Stop early if the episode terminated.
        try:
            if bool(_to_host(state.done)):
                break
        except Exception:  # pragma: no cover - done may be absent in odd envs
            pass

    return recorder.build(title=title, seed=seed)


__all__ = ["TrajectoryRecorder", "rollout_to_trajectory"]
