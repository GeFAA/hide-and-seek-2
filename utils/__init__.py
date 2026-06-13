"""
utils/ -- Foundational, dependency-light helpers for Hide & Seek 2.0.

This package is built FIRST: ``envs/``, ``models/`` and ``trainers/`` all import
these names. Everything here is pure JAX (jit/vmap friendly) except
``logging.MetricLogger``, which is an explicitly host-side concern.

Public surface
--------------
spaces      : ``Box``, ``Discrete``, ``MultiDiscrete``, ``Dict`` -- gym-free space
              descriptors with a JAX-native ``sample(key)``.
pytree      : ``batchify`` / ``unbatchify`` (multi-agent axis flatten/restore) plus
              ``tree_stack`` / ``tree_select`` / ``tree_zeros_like``.
elo         : ``expected_score`` / ``update_elo`` for historical self-play ranking.
visibility  : GPU ray-casting -- ``cast_rays``, ``lidar_scan``, ``in_vision_cone``,
              ``compute_visibility`` (LOS + vision cone + fog-of-war attenuation).
logging     : ``MetricLogger`` -- scalar aggregation + optional wandb/tensorboard.
"""
from __future__ import annotations

from utils.spaces import Box, Discrete, Dict, MultiDiscrete, Space
from utils.pytree import (
    batchify,
    unbatchify,
    tree_stack,
    tree_select,
    tree_zeros_like,
)
from utils.elo import expected_score, update_elo, ELO_SCALE
from utils.visibility import (
    cast_rays,
    lidar_scan,
    in_vision_cone,
    compute_visibility,
    walls_to_segments,
    compute_visibility_batch,
)
from utils.logging import MetricLogger

__all__ = [
    # spaces
    "Space",
    "Box",
    "Discrete",
    "MultiDiscrete",
    "Dict",
    # pytree
    "batchify",
    "unbatchify",
    "tree_stack",
    "tree_select",
    "tree_zeros_like",
    # elo
    "expected_score",
    "update_elo",
    "ELO_SCALE",
    # visibility
    "cast_rays",
    "lidar_scan",
    "in_vision_cone",
    "compute_visibility",
    "walls_to_segments",
    "compute_visibility_batch",
    # logging
    "MetricLogger",
]
