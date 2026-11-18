"""Checkpoint persistence.

Checkpoints hold only tensors and plain Python types (dict, list, str,
int, float, bool), so they load with ``torch.load(weights_only=True)``, which
does not execute arbitrary pickled code.

* ``best.pt``: model weights + model/preprocessing config + metadata (used by inference).
* ``last.pt``: everything needed to resume (optimiser, scheduler, phase/epoch, history, RNG).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

CHECKPOINT_FORMAT_VERSION = 1


def save_checkpoint(path: str | Path, payload: dict[str, Any]) -> None:
    """Atomically write a checkpoint (write to a temp file, then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save({"format_version": CHECKPOINT_FORMAT_VERSION, **payload}, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(path, map_location=map_location, weights_only=True)


def capture_rng_state() -> dict[str, Any]:
    np_state = np.random.get_state()
    return {
        "python": _to_plain(random.getstate()),
        "numpy": [np_state[0], np_state[1].tolist(), int(np_state[2]), int(np_state[3]), float(np_state[4])],
        "torch": torch.get_rng_state(),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    py = state["python"]
    random.setstate((py[0], tuple(py[1]), py[2]))
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
    torch.set_rng_state(state["torch"])


def _to_plain(obj: Any) -> Any:
    if isinstance(obj, tuple):
        return [_to_plain(o) for o in obj]
    return obj
