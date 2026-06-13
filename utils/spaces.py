"""
utils/spaces.py -- Lightweight, gym-free space descriptors for Hide & Seek 2.0.

These are *pure* shape/dtype descriptors with a JAX-native ``sample(key)`` method.
They exist so the env/model can advertise their observation & action structure
without taking a hard dependency on ``gym``/``gymnasium`` (which pull in NumPy-only
RNG and host-side machinery that does not belong on-device).

Design goals
------------
* **Pure & jit-friendly.** ``sample`` takes an explicit ``jax.random.PRNGKey`` and
  returns a device array; no global RNG, no host transfers.
* **Composable.** ``Dict`` nests arbitrary spaces; ``contains`` / ``sample`` recurse.
* **Cheap.** Spaces are frozen dataclasses; shapes are Python tuples.

Conventions
-----------
* ``shape`` is the per-sample shape (no batch axis). The caller ``vmap``s for batches.
* ``dtype`` follows the contract: continuous => ``float32``, discrete ids => ``int32``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict as PyDict, Mapping, Tuple

import jax
import jax.numpy as jnp

__all__ = ["Space", "Box", "Discrete", "MultiDiscrete", "Dict"]


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------
class Space:
    """Abstract base for all space descriptors.

    Subclasses must define ``shape``/``dtype`` attributes and implement
    :meth:`sample` and :meth:`contains`.
    """

    shape: Tuple[int, ...]
    dtype: Any

    def sample(self, key: jax.Array) -> Any:
        """Draw a single random element of this space.

        Args:
            key: A ``jax.random.PRNGKey``.

        Returns:
            A device array (or nested structure for :class:`Dict`).
        """
        raise NotImplementedError

    def contains(self, x: Any) -> bool:
        """Cheap *structural* membership test (shape/dtype), host-side.

        This is intended for tests/assertions, **not** the hot path. It returns a
        plain Python ``bool`` and therefore must not be called on traced values.
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Box: bounded continuous (or general) array space
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Box(Space):
    """A (possibly bounded) continuous box ``[low, high]^shape``.

    Args:
        low: Lower bound. Scalar (broadcast over ``shape``) or array of ``shape``.
        high: Upper bound. Scalar or array of ``shape``.
        shape: Per-sample shape.
        dtype: Sample dtype (default ``float32``).

    Notes:
        ``low``/``high`` are stored as ``jnp`` arrays broadcast to ``shape`` so that
        :meth:`sample` is a single ``jax.random.uniform`` call with no Python logic.
    """

    low: Any
    high: Any
    shape: Tuple[int, ...]
    dtype: Any = jnp.float32

    # broadcast bounds cached as arrays (not part of equality / repr noise)
    _low: jax.Array = field(init=False, repr=False, compare=False)
    _high: jax.Array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        low = jnp.broadcast_to(jnp.asarray(self.low, self.dtype), self.shape)
        high = jnp.broadcast_to(jnp.asarray(self.high, self.dtype), self.shape)
        # frozen dataclass: bypass the immutability guard to cache derived arrays.
        object.__setattr__(self, "_low", low)
        object.__setattr__(self, "_high", high)

    def sample(self, key: jax.Array) -> jax.Array:
        """Uniformly sample from ``[low, high]`` (element-wise)."""
        u = jax.random.uniform(key, self.shape, dtype=jnp.float32)
        out = self._low + u * (self._high - self._low)
        return out.astype(self.dtype)

    def contains(self, x: Any) -> bool:
        x = jnp.asarray(x)
        if tuple(x.shape) != tuple(self.shape):
            return False
        return bool(jnp.all((x >= self._low) & (x <= self._high)))


# ---------------------------------------------------------------------------
# Discrete: single categorical in {0, ..., n-1}
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Discrete(Space):
    """A single categorical variable taking integer values in ``[0, n)``.

    Args:
        n: Number of categories (must be >= 1).
        dtype: Integer dtype of samples (default ``int32`` per contract).
    """

    n: int
    dtype: Any = jnp.int32
    shape: Tuple[int, ...] = field(default=(), init=False)

    def __post_init__(self) -> None:
        if self.n < 1:
            raise ValueError(f"Discrete requires n >= 1, got {self.n}.")

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample a uniform category id in ``[0, n)`` as a scalar array."""
        return jax.random.randint(key, (), 0, self.n, dtype=self.dtype)

    def contains(self, x: Any) -> bool:
        x = jnp.asarray(x)
        if x.shape != ():
            return False
        return bool((x >= 0) & (x < self.n))


# ---------------------------------------------------------------------------
# MultiDiscrete: vector of independent categoricals
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class MultiDiscrete(Space):
    """A vector of independent categoricals; entry ``i`` lies in ``[0, nvec[i])``.

    Mirrors the discrete-action contract: ``action["interact"]`` is a
    ``MultiDiscrete(action_discrete_nvec)`` per agent.

    Args:
        nvec: Per-component category counts.
        dtype: Integer dtype (default ``int32``).
    """

    nvec: Tuple[int, ...]
    dtype: Any = jnp.int32
    shape: Tuple[int, ...] = field(init=False)
    _nvec: jax.Array = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        nvec = tuple(int(v) for v in self.nvec)
        if any(v < 1 for v in nvec):
            raise ValueError(f"MultiDiscrete requires all nvec >= 1, got {nvec}.")
        object.__setattr__(self, "nvec", nvec)
        object.__setattr__(self, "shape", (len(nvec),))
        object.__setattr__(self, "_nvec", jnp.asarray(nvec, dtype=self.dtype))

    def sample(self, key: jax.Array) -> jax.Array:
        """Sample one category per component (shape ``(len(nvec),)``)."""
        # randint's high bound is per-element; a single vectorized call suffices.
        return jax.random.randint(
            key, self.shape, jnp.zeros_like(self._nvec), self._nvec, dtype=self.dtype
        )

    def contains(self, x: Any) -> bool:
        x = jnp.asarray(x)
        if tuple(x.shape) != tuple(self.shape):
            return False
        return bool(jnp.all((x >= 0) & (x < self._nvec)))


# ---------------------------------------------------------------------------
# Dict: ordered mapping of named subspaces
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dict(Space):
    """A namespace of named subspaces (the observation/action container).

    Args:
        spaces: Mapping ``name -> Space``. Insertion order is preserved and is the
            canonical key order for sampling/iteration.

    Notes:
        ``shape``/``dtype`` are intentionally ``None`` for a Dict; introspect the
        children instead.
    """

    spaces: Mapping[str, Space]
    shape: Tuple[int, ...] = field(default=None, init=False)  # type: ignore[assignment]
    dtype: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        # Normalize to a plain dict (preserves insertion order in py>=3.7).
        object.__setattr__(self, "spaces", dict(self.spaces))

    def sample(self, key: jax.Array) -> PyDict[str, Any]:
        """Recursively sample every subspace with independent split keys."""
        keys = jax.random.split(key, len(self.spaces))
        return {
            name: space.sample(k)
            for (name, space), k in zip(self.spaces.items(), keys)
        }

    def contains(self, x: Any) -> bool:
        if not isinstance(x, Mapping):
            return False
        if set(x.keys()) != set(self.spaces.keys()):
            return False
        return all(space.contains(x[name]) for name, space in self.spaces.items())

    def __getitem__(self, name: str) -> Space:
        return self.spaces[name]

    def keys(self):  # noqa: D401 - thin pass-through
        """Return the subspace names."""
        return self.spaces.keys()
