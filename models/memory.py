"""
models/memory.py -- PureJaxRL-style recurrent memory for Hide & Seek 2.0.

``ScannedGRU`` wraps a single :class:`flax.linen.GRUCell` and unrolls it over a
leading **TIME** axis with :func:`flax.linen.scan`. The defining feature -- and
the reason recurrence matters in this game -- is the **reset/done handling**:
before applying the cell at each timestep we *zero the carry* wherever a
``reset`` flag is set. This:

* respects **episode boundaries** in a batched rollout (a ``done`` at time ``t``
  must not let memory bleed into the next episode), and
* gives the policy **object permanence**: between resets the GRU integrates
  observations over time, so an agent can remember an object (e.g. a box, or a
  hider it briefly saw) that is currently occluded by fog / out of its vision
  cone -- exactly the partial-observability challenge Hide & Seek poses.

Convention (matches PureJaxRL / JaxMARL ``ScannedRNN``): ``__call__(carry, x)``
where ``x = (inputs, resets)`` are both stacked along the leading time axis.
"""
from __future__ import annotations

from typing import Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp


class ScannedGRU(nn.Module):
    """A GRU unrolled over time with PureJaxRL-style episode-boundary resets.

    The module is *scan-transformed*: ``nn.scan`` lifts a single
    :class:`flax.linen.GRUCell` application across the leading time axis,
    broadcasting the (shared) GRU parameters and threading the recurrent carry.

    Attributes:
        hidden_size: Dimension of the GRU hidden state / carry. Set from
            ``cfg.gru_hidden`` by the caller -- never hard-coded.
    """

    hidden_size: int

    @nn.compact
    def __call__(
        self,
        carry: jnp.ndarray,
        x: Tuple[jnp.ndarray, jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Scan a GRU over time, resetting the carry at episode boundaries.

        Args:
            carry: Initial hidden state, shape ``(*batch_dims, hidden_size)``.
            x: Tuple ``(inputs, resets)`` stacked along a leading time axis ``T``:

                * ``inputs``: ``(T, *batch_dims, in_features)`` observations/
                  features for each timestep.
                * ``resets``: ``(T, *batch_dims)`` boolean ``done`` flags. Where
                  ``True``, the carry is zeroed *before* the cell is applied so
                  no memory crosses the episode boundary.

        Returns:
            ``(new_carry, outputs)`` where ``new_carry`` is the final hidden
            state ``(*batch_dims, hidden_size)`` and ``outputs`` is the stacked
            per-timestep hidden states ``(T, *batch_dims, hidden_size)``.
        """

        # The per-timestep body, written as a *method-style* function
        # ``fn(module, carry, step_input)``. ``nn.scan`` (configured below) lifts
        # it across the leading time axis; ``module`` is the scanned ``self`` whose
        # scope owns the (broadcast / time-shared) GRU parameters, so the GRUCell
        # submodule is created correctly inside the scan's tracing context.
        def _body(
            module: "ScannedGRU",
            cell_carry: jnp.ndarray,
            step_input: Tuple[jnp.ndarray, jnp.ndarray],
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            ins, reset = step_input
            # Zero the carry wherever the episode just reset. reset is
            # (*batch_dims,) -> broadcast against (*batch_dims, hidden_size).
            # Branch-free (jnp.where), as required in the hot path.
            cell_carry = jnp.where(
                reset[..., None],
                jnp.zeros_like(cell_carry),
                cell_carry,
            )
            new_carry, out = nn.GRUCell(
                features=module.hidden_size, name="gru_cell"
            )(cell_carry, ins)
            return new_carry, out

        # Lift the body across the leading TIME axis. ``nn.scan`` is Flax's
        # lifted wrapper over ``jax.lax.scan`` (CONTRACT §5 "lax.scan"): it
        # threads the carry while tracing the body ONCE, so there is no Python
        # loop over time. The GRU params are ``variable_broadcast="params"`` --
        # the SAME recurrent cell parameters are shared/reused across every
        # timestep (one cell, not T copies) -- and ``split_rngs={"params": False}``
        # keeps a single, non-per-step RNG so init draws those shared params once.
        # We scan over axis 0 (time) for both inputs and outputs.
        scan = nn.scan(
            _body,
            variable_broadcast="params",
            split_rngs={"params": False},
            in_axes=0,
            out_axes=0,
        )
        new_carry, outputs = scan(self, carry, x)
        return new_carry, outputs

    @staticmethod
    def initialize_carry(
        batch_dims: Tuple[int, ...],
        hidden_size: int,
    ) -> jnp.ndarray:
        """Create a zero recurrent carry.

        Args:
            batch_dims: Leading batch dimensions, e.g. ``(num_envs * num_agents,)``
                or ``(num_envs, num_agents)``. May be empty for an unbatched carry.
            hidden_size: GRU hidden width (``cfg.gru_hidden``).

        Returns:
            A zero array of shape ``(*batch_dims, hidden_size)``. Uses the same
            zeros init as :class:`flax.linen.GRUCell` so a freshly-initialized
            carry matches what the cell would produce after a reset.
        """
        return jnp.zeros((*batch_dims, hidden_size), dtype=jnp.float32)
