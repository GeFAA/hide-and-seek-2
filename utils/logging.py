"""
utils/logging.py -- Scalar metric aggregation & optional experiment-tracker sinks.

``MetricLogger`` accumulates scalar metric dicts across training updates, pretty-
prints them to the console, and -- if available -- mirrors them to Weights & Biases
and/or TensorBoard. The heavy third-party imports are guarded behind ``try/except``
so the repo imports cleanly with none of them installed.

Hard rule (CONTRACT §0): **no host calls inside jit.** Logging is a host-side
concern. Call ``MetricLogger.log`` only *after* a jitted train step has returned
and its metrics have been pulled to the host (e.g. via ``jax.device_get``). Inside
the jitted hot loop, metrics are accumulated as device arrays in the ``info`` pytree
and handed to the logger only at the trainer boundary.
"""
from __future__ import annotations

import math
import time
from collections import defaultdict
from typing import Any, Dict, List, Mapping, Optional

__all__ = ["MetricLogger"]


# --- Optional dependencies, guarded -----------------------------------------
try:  # pragma: no cover - availability depends on environment
    import wandb as _wandb  # type: ignore
    _HAS_WANDB = True
except Exception:  # noqa: BLE001 - any import failure => feature off
    _wandb = None
    _HAS_WANDB = False

try:  # pragma: no cover
    from torch.utils.tensorboard import SummaryWriter as _SummaryWriter  # type: ignore
    _HAS_TB = True
except Exception:  # noqa: BLE001
    try:  # tensorboardX is a common fallback
        from tensorboardX import SummaryWriter as _SummaryWriter  # type: ignore
        _HAS_TB = True
    except Exception:  # noqa: BLE001
        _SummaryWriter = None  # type: ignore
        _HAS_TB = False


def _to_float(value: Any) -> float:
    """Coerce a scalar-like (Python number, 0-d array, 1-elem array) to ``float``.

    Host-side only. Uses ``float(...)`` which triggers a device->host copy for JAX
    arrays — acceptable here because logging happens outside the jitted loop.
    """
    try:
        # 0-d / 1-elem numpy or jax arrays support float() directly.
        return float(value)
    except (TypeError, ValueError):
        # Fall back to mean for small multi-element arrays (e.g. per-agent vec).
        import numpy as _np  # local import keeps numpy optional at module load
        return float(_np.asarray(value).mean())


class MetricLogger:
    """Aggregate scalar metrics and emit them to console / wandb / tensorboard.

    Typical use::

        logger = MetricLogger(use_wandb=cfg.train.use_wandb, log_dir="runs/hs2")
        ...
        metrics = jax.device_get(metrics_pytree)   # device -> host once
        logger.log(metrics, step=update_idx)
        logger.close()

    Args:
        use_wandb: Enable the wandb sink (no-op if wandb isn't importable).
        use_tensorboard: Enable the TensorBoard sink (no-op if unavailable).
        log_dir: Directory for the TensorBoard ``SummaryWriter``.
        wandb_project: Project name passed to ``wandb.init`` when starting a run.
        wandb_run: An already-initialized wandb run to reuse (skips ``init``).
        prefix: Optional string prefixed to every metric key (e.g. ``"train/"``).
        console: Whether to pretty-print to stdout on each ``log`` call.
    """

    def __init__(
        self,
        use_wandb: bool = False,
        use_tensorboard: bool = False,
        log_dir: str = "runs",
        wandb_project: str = "hide-and-seek-2.0",
        wandb_run: Optional[Any] = None,
        prefix: str = "",
        console: bool = True,
    ) -> None:
        self.prefix = prefix
        self.console = console
        self._start_time = time.time()
        # Running history of every logged scalar (host-side), for summaries/plots.
        self._history: Dict[str, List[float]] = defaultdict(list)

        # --- wandb sink ---
        self.use_wandb = bool(use_wandb) and _HAS_WANDB
        self._wandb_run = None
        if use_wandb and not _HAS_WANDB:
            print("[MetricLogger] wandb requested but not installed; skipping.")
        if self.use_wandb:
            if wandb_run is not None:
                self._wandb_run = wandb_run
            else:  # pragma: no cover - network/side-effect, exercised in real runs
                self._wandb_run = _wandb.init(project=wandb_project)

        # --- tensorboard sink ---
        self.use_tensorboard = bool(use_tensorboard) and _HAS_TB
        self._tb_writer = None
        if use_tensorboard and not _HAS_TB:
            print("[MetricLogger] tensorboard requested but not installed; skipping.")
        if self.use_tensorboard:  # pragma: no cover - filesystem side-effect
            self._tb_writer = _SummaryWriter(log_dir=log_dir)

    # ------------------------------------------------------------------ #
    def _key(self, name: str) -> str:
        return f"{self.prefix}{name}"

    def log(self, metrics: Mapping[str, Any], step: int) -> Dict[str, float]:
        """Record a dict of scalar metrics at ``step`` and emit to all sinks.

        Args:
            metrics: Mapping ``name -> scalar-like`` (Python number or 0-d/1-elem
                array). Already on the host — do **not** pass traced/jit values.
            step: The global step / update index for the x-axis.

        Returns:
            The cleaned ``{name: float}`` dict actually logged (NaN/inf dropped).
        """
        clean: Dict[str, float] = {}
        for name, value in metrics.items():
            v = _to_float(value)
            if math.isnan(v) or math.isinf(v):
                # Skip non-finite metrics rather than poisoning wandb/TB charts.
                continue
            clean[name] = v
            self._history[name].append(v)

        if self.console:
            self._print(clean, step)

        if self.use_wandb and self._wandb_run is not None:  # pragma: no cover
            self._wandb_run.log(
                {self._key(k): v for k, v in clean.items()}, step=step
            )

        if self.use_tensorboard and self._tb_writer is not None:  # pragma: no cover
            for k, v in clean.items():
                self._tb_writer.add_scalar(self._key(k), v, step)

        return clean

    def _print(self, metrics: Mapping[str, float], step: int) -> None:
        """Pretty-print a single update's metrics as an aligned table to stdout."""
        elapsed = time.time() - self._start_time
        header = f"--- step {step}  (elapsed {elapsed:7.1f}s) ---"
        if not metrics:
            print(header + " [no finite metrics]")
            return
        width = max(len(k) for k in metrics)
        lines = [header]
        for k in sorted(metrics):
            lines.append(f"  {k:<{width}} : {metrics[k]:+.4f}")
        print("\n".join(lines))

    def summary(self) -> Dict[str, float]:
        """Return the latest value of every metric seen so far (host-side)."""
        return {k: v[-1] for k, v in self._history.items() if v}

    def mean(self, name: str) -> float:
        """Mean over all recorded values of ``name`` (``nan`` if never logged)."""
        vals = self._history.get(name, [])
        return float(sum(vals) / len(vals)) if vals else float("nan")

    def close(self) -> None:
        """Flush & close any open sinks. Safe to call multiple times."""
        if self.use_tensorboard and self._tb_writer is not None:  # pragma: no cover
            self._tb_writer.flush()
            self._tb_writer.close()
            self._tb_writer = None
        if self.use_wandb and self._wandb_run is not None:  # pragma: no cover
            self._wandb_run.finish()
            self._wandb_run = None

    def __enter__(self) -> "MetricLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
