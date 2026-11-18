"""Compute device selection."""

from __future__ import annotations

import torch


def resolve_device(preference: str = "auto") -> torch.device:
    """Return the requested device, or the best available one for ``"auto"``.

    Order for ``auto``: Apple MPS -> CUDA -> CPU.
    """
    pref = preference.lower()
    if pref == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if pref == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS requested but not available on this machine")
    if pref == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available on this machine")
    return torch.device(pref)
