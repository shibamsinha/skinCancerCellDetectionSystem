"""Reproducible random seeding."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def seed_everything(seed: int) -> torch.Generator:
    """Seed Python, NumPy and PyTorch RNGs; return a seeded ``torch.Generator``.

    Note: Apple MPS kernels are not guaranteed bit-for-bit deterministic, so
    repeated GPU runs can differ slightly even with identical seeds. Data
    splits, sampling order and augmentation draws are fully deterministic.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    return gen


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker initialiser: derive per-worker NumPy/Python seeds."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
