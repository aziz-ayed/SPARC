"""Reproducible RNG seeding for Python, NumPy, PyTorch (CPU + CUDA), DDP-aware."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_seed(seed: int = 1337, rank: int = 0) -> None:
    """Seed Python, NumPy, and PyTorch RNGs deterministically.

    The effective seed is ``seed + rank``, so every DDP worker draws from a
    different stream while remaining fully deterministic.

    Args:
        seed: Base seed, identical across ranks. ``1337`` is the value used
              for every paper experiment.
        rank: DDP rank in ``[0, world_size)``. ``0`` for single-GPU.
    """
    full_seed = seed + rank
    random.seed(full_seed)
    np.random.seed(full_seed)
    torch.manual_seed(full_seed)
    torch.cuda.manual_seed_all(full_seed)
    os.environ["PYTHONHASHSEED"] = str(full_seed)
