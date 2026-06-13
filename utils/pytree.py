"""
utils/pytree.py -- Multi-agent pytree helpers for Hide & Seek 2.0.

The network (actor/critic) is written for a *flat* batch of agents: it does not
care which environment or team a row came from. The trainer, by contrast, keeps
arrays with an explicit **leading agent axis** ``A`` (see CONTRACT §3, where every
``obs`` array is ``(A, ...)`` per env, and the trainer adds an outer ``num_envs``
axis via ``vmap``). ``batchify`` / ``unbatchify`` are the *only* sanctioned bridge
between these two layouts (CONTRACT §8).

Batchify convention (env-outer / agent-inner; collapse 2 leading axes)
---------------------------------------------------------------------
The trainer keeps every ``obs`` leaf with the leading pair ``(num_envs, A)``
(``num_envs`` added by the env ``vmap``, ``A`` the per-env agent axis of
CONTRACT §3). ``batchify`` merges *exactly that fixed number of leading axes*
(``n_lead=2`` by default) into a single network batch axis, **preserving ALL
trailing axes**; ``unbatchify`` restores them.

    batchify(x, n_lead=2) : leaf ``(d0, d1, *rest)`` -> ``(d0*d1, *rest)``.
        The default merges the ``(num_envs, A)`` pair while keeping every trailing
        axis intact. This is *required* for the multi-axis obs leaves:
          * 4-D ``entities``     ``(num_envs, A, E, Fe)`` -> ``(num_envs*A, E, Fe)``
                                 (NOT ``(num_envs*A*E, Fe)`` — E must survive!),
          * 3-D ``entity_mask``  ``(num_envs, A, E)``     -> ``(num_envs*A, E)``,
          * 2-D ``agent_active`` ``(num_envs, A)``        -> ``(num_envs*A,)``,
          * 2-D ``self``         ``(num_envs, A, Fs)``    -> ``(num_envs*A, Fs)``.

    unbatchify(x, n_agents) : leaf ``(B, *rest)`` -> ``(B//n_agents, n_agents, *rest)``
        i.e. splits the flat batch back into ``(num_envs, A, *rest)``.

Row-index convention: **env outer, agent inner** — the agent index varies
**fastest** within an env, so ``row = env * A + agent``. This matches the
trainer's ``_unbatch_value`` reshape ``(num_envs, A) + rest`` and the fallback
flatten ``reshape((num_envs*A,) + rest)``. A plain row-major
``reshape((-1, *rest))`` (batchify) and ``reshape((-1, n_agents, *rest))``
(unbatchify) realize this ordering exactly, with no axis transpose.

All functions are pure and operate via ``jax.tree_util`` so they transparently
handle dicts of arrays (the observation contract) and ``flax.struct`` pytrees.
"""
from __future__ import annotations

from typing import Any, Callable

import jax
import jax.numpy as jnp

__all__ = [
    "batchify",
    "unbatchify",
    "tree_stack",
    "tree_select",
    "tree_zeros_like",
]

PyTree = Any


def batchify(x: PyTree, n_lead: int = 2) -> PyTree:
    """Collapse a FIXED number of LEADING axes into one batch axis.

    Each leaf of shape ``(d0, d1, ..., d_{n_lead-1}, *rest)`` is reshaped to
    ``(d0 * d1 * ... * d_{n_lead-1}, *rest)`` — i.e. the first ``n_lead`` axes are
    merged and **every trailing axis is preserved unchanged**. With the default
    ``n_lead=2`` this merges the ``(num_envs, A)`` pair (CONTRACT §3 obs layout)
    while keeping ``E`` / ``Fe`` / ``Fs`` intact:

        * ``entities``     ``(num_envs, A, E, Fe)`` -> ``(num_envs*A, E, Fe)``
        * ``entity_mask``  ``(num_envs, A, E)``     -> ``(num_envs*A, E)``
        * ``self``         ``(num_envs, A, Fs)``    -> ``(num_envs*A, Fs)``
        * ``agent_active`` ``(num_envs, A)``        -> ``(num_envs*A,)``

    A plain row-major ``reshape`` realizes the **env-outer / agent-inner** ordering
    (agent varies fastest: ``row = env * A + agent``), matching the trainer.

    Args:
        x: A pytree of arrays whose leaves share the same ``n_lead`` leading axes.
            Leaves with fewer than ``n_lead`` dims are returned unchanged.
        n_lead: Number of leading axes to collapse into the batch axis (default 2,
            i.e. ``(num_envs, A)``).

    Returns:
        A pytree with each leaf's first ``n_lead`` axes merged into one.
    """

    def _flatten(leaf: jnp.ndarray) -> jnp.ndarray:
        leaf = jnp.asarray(leaf)
        if leaf.ndim < n_lead:
            # Not enough leading axes to merge — leave untouched.
            return leaf
        lead = 1
        for s in leaf.shape[:n_lead]:
            lead *= s
        return leaf.reshape((lead,) + leaf.shape[n_lead:])

    return jax.tree_util.tree_map(_flatten, x)


def unbatchify(x: PyTree, n_agents: int) -> PyTree:
    """Inverse of :func:`batchify` (default ``n_lead=2``): split the batch axis.

    Each leaf of shape ``(B, *rest)`` becomes ``(B // n_agents, n_agents, *rest)``,
    restoring the ``(num_envs, A, *rest)`` layout. The split is **env-outer /
    agent-inner** (agent varies fastest), matching the trainer's
    ``_unbatch_value`` reshape ``(num_envs, A) + rest``.

    Args:
        x: A pytree produced by :func:`batchify` (leaves are ``(num_envs*A, *rest)``).
        n_agents: The agent-axis size ``A`` to split back out (the inner, faster
            axis). Leaves with no leading axis (rank 0) are returned unchanged.

    Returns:
        A pytree with each leaf reshaped to ``(B // n_agents, n_agents, *rest)``.
    """

    def _unflatten(leaf: jnp.ndarray) -> jnp.ndarray:
        leaf = jnp.asarray(leaf)
        if leaf.ndim < 1:
            return leaf
        return leaf.reshape((-1, n_agents) + leaf.shape[1:])

    return jax.tree_util.tree_map(_unflatten, x)


def tree_stack(trees: list[PyTree], axis: int = 0) -> PyTree:
    """Stack a list of identically-structured pytrees along a new axis.

    Args:
        trees: Non-empty list of pytrees with identical structure & leaf shapes.
        axis: New axis position to stack along (default 0, i.e. a leading axis).

    Returns:
        A single pytree whose leaves are ``jnp.stack`` of the corresponding leaves.

    Raises:
        ValueError: If ``trees`` is empty.
    """
    if not trees:
        raise ValueError("tree_stack requires a non-empty list of pytrees.")
    return jax.tree_util.tree_map(
        lambda *leaves: jnp.stack(leaves, axis=axis), *trees
    )


def tree_select(mask: jnp.ndarray, a: PyTree, b: PyTree) -> PyTree:
    """Element/broadcast-wise ``where(mask, a, b)`` over two matching pytrees.

    This is the branch-free, jit-safe way to pick between two states (e.g.
    auto-reset: ``tree_select(done, reset_state, stepped_state)``).

    Args:
        mask: Boolean array. It is broadcast against each leaf from the **left**
            (leading-axis broadcasting), so a per-env/per-agent ``mask`` of shape
            ``(...,)`` correctly selects whole feature vectors of shape
            ``(..., F)``.
        a: Pytree chosen where ``mask`` is True.
        b: Pytree chosen where ``mask`` is False. Same structure as ``a``.

    Returns:
        A pytree with leaves ``jnp.where(mask_b, a_leaf, b_leaf)``.
    """
    mask = jnp.asarray(mask)

    def _sel(a_leaf: jnp.ndarray, b_leaf: jnp.ndarray) -> jnp.ndarray:
        a_leaf = jnp.asarray(a_leaf)
        # Right-pad the mask with singleton axes so it broadcasts over the
        # trailing feature dims of the leaf (leading-axis alignment).
        m = mask.reshape(mask.shape + (1,) * (a_leaf.ndim - mask.ndim)) \
            if a_leaf.ndim >= mask.ndim else mask
        return jnp.where(m, a_leaf, b_leaf)

    return jax.tree_util.tree_map(_sel, a, b)


def tree_zeros_like(tree: PyTree) -> PyTree:
    """Return a pytree of zeros matching every leaf's shape & dtype.

    Args:
        tree: Any pytree of arrays.

    Returns:
        A structurally-identical pytree of zero-filled leaves.
    """
    return jax.tree_util.tree_map(lambda x: jnp.zeros_like(jnp.asarray(x)), tree)


# Exposed for callers that want to map an arbitrary fn over leaves without
# importing jax.tree_util directly (keeps utils as the single dependency surface).
tree_map: Callable[..., PyTree] = jax.tree_util.tree_map
