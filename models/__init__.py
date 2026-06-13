"""
models/ -- Entity-Transformer policy stack for Hide & Seek 2.0.

This package implements the CTDE (Centralized-Training / Decentralized-Execution)
neural architecture described in ``docs/CONTRACT.md`` §5:

* :class:`EntityTransformer` -- a permutation-invariant, masked self-attention
  encoder over a padded *set* of entity tokens. The mask drops invisible/padded
  entities; the lack of positional encoding makes it order-invariant. This is the
  shared perceptual primitive used by both actor and critic.
* :class:`ScannedGRU` -- a PureJaxRL-style GRU unrolled over a leading time axis
  with episode-boundary resets, giving the policy memory / object permanence.
* :class:`ActorRNN` -- the **decentralized** actor: local masked obs ->
  EntityTransformer -> ScannedGRU -> hybrid (Gaussian + Categorical) action heads.
* :class:`CriticRNN` -- the **centralized** critic: privileged global obs ->
  its own EntityTransformer -> ScannedGRU -> scalar value.
* :class:`ActorCritic` -- a convenience bundle pairing both modules with their
  (separate) parameter pytrees.

Plus the module-level helpers :func:`sample_and_logprob` / :func:`eval_logprob`
for hybrid-action (log-)probabilities, and the construction utilities in
``models.networks``.

All dimensions are read from ``config.py`` (CONTRACT §0) -- nothing is hard-coded.
"""
from __future__ import annotations

from models.actor import ActorRNN, eval_logprob, sample_and_logprob
from models.critic import CriticRNN
from models.memory import ScannedGRU
from models.networks import (
    ActorCritic,
    count_params,
    init_actor,
    init_critic,
    initialize_carries,
)
from models.transformer import EntityTransformer

__all__ = [
    # Core modules.
    "EntityTransformer",
    "ScannedGRU",
    "ActorRNN",
    "CriticRNN",
    "ActorCritic",
    # Action (log-)prob helpers.
    "sample_and_logprob",
    "eval_logprob",
    # Construction / init utilities.
    "init_actor",
    "init_critic",
    "initialize_carries",
    "count_params",
]
