"""
models/critic.py -- Centralized (CTDE) recurrent critic for Hide & Seek 2.0.

In Centralized-Training / Decentralized-Execution (CTDE / MAPPO), the value
function is trained with **privileged global state** that the actors never see.
Concretely the critic consumes ``global_entities`` in *absolute* coordinates,
unmasked by visibility, and carrying the two privileged extras appended in the
global layout (CONTRACT §3.2): ``true_is_decoy`` and ``grounded``. This lets the
value baseline be accurate even when individual agents are fooled by a decoy or
have objects occluded by fog -- which stabilizes advantage estimates and hence
policy learning, while the actor remains strictly decentralized.

Forward pipeline::

    global obs (global_entities, global_mask)
        -> EntityTransformer  (its OWN encoder; critic does not share actor params)
        -> ScannedGRU         (memory over time, reset at episode boundaries)
        -> Dense(1)           -> scalar state value V(s)
"""
from __future__ import annotations

from typing import Dict, Tuple

import flax.linen as nn
import jax.numpy as jnp

from config import ModelConfig
from models.memory import ScannedGRU
from models.transformer import EntityTransformer


class CriticRNN(nn.Module):
    """Recurrent centralized critic: privileged entity-Transformer -> GRU -> value.

    Attributes:
        cfg: Shared :class:`ModelConfig`. Provides ``d_model``, ``gru_hidden``,
            and the privileged feature width ``global_entity_feat_dim`` (all read
            from config, never hard-coded).
    """

    cfg: ModelConfig

    @nn.compact
    def __call__(
        self,
        carry: jnp.ndarray,
        x: Tuple[Dict[str, jnp.ndarray], jnp.ndarray],
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Estimate the state value from a TIME-major batch of global obs.

        Args:
            carry: GRU hidden state ``(*batch_dims, gru_hidden)``.
            x: Tuple ``(global_obs, dones)`` where:

                * ``global_obs`` is the **privileged** observation dict
                  (time-major), using contract keys (CONTRACT §3): ``"global_entities"``
                  ``(T, *B, E, Fg)`` and ``"global_mask"`` ``(T, *B, E)`` bool
                  (existence mask only -- NOT visibility).
                * ``dones`` is ``(T, *B)`` bool episode-boundary flags for the GRU.

        Returns:
            ``(new_carry, value)`` where ``value`` is ``(T, *B)`` -- the scalar
            state-value estimate with its trailing singleton dimension squeezed.
        """
        global_obs, dones = x

        # CTDE: critic sees privileged global state incl. true decoy identity.
        # No `self_feat` is supplied: the critic encodes the *whole* scene from a
        # god's-eye view, so the EntityTransformer falls back to its learned
        # constant query token rather than an observer-relative one.
        embed = EntityTransformer(self.cfg, name="encoder")(
            global_obs["global_entities"],
            global_obs["global_mask"],
            None,
        )  # (T, *B, 2 * d_model)

        # Recurrent memory over time, reset at episode boundaries.
        carry, gru_out = ScannedGRU(self.cfg.gru_hidden, name="memory")(
            carry, (embed, dones)
        )  # (T, *B, gru_hidden)

        trunk = nn.relu(nn.Dense(self.cfg.gru_hidden, name="trunk")(gru_out))
        value = nn.Dense(1, name="value_head")(trunk)  # (T, *B, 1)
        value = jnp.squeeze(value, axis=-1)            # (T, *B)
        return carry, value
