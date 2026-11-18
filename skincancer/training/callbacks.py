"""Training callbacks: early stopping and best-model tracking (Keras-style).

Both support ``mode="min"`` (e.g. ``val_loss``) and ``mode="max"`` (e.g. ``val_roc_auc``).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass


def _better(value: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    if mode == "min":
        return value < best - min_delta
    if mode == "max":
        return value > best + min_delta
    raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")


def _initial(mode: str) -> float:
    return math.inf if mode == "min" else -math.inf


@dataclass
class EarlyStopping:
    """Stop when the monitored value stops improving for ``patience`` epochs."""

    patience: int
    mode: str = "min"
    min_delta: float = 1e-4
    best: float | None = None
    bad_epochs: int = 0

    def __post_init__(self) -> None:
        if self.best is None:
            self.best = _initial(self.mode)

    def step(self, value: float) -> bool:
        """Record ``value``; return True if training should stop."""
        if _better(value, self.best, self.mode, self.min_delta):
            self.best = value
            self.bad_epochs = 0
        else:
            self.bad_epochs += 1
        return self.bad_epochs >= self.patience

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, state: dict) -> None:
        self.best = state["best"]
        self.bad_epochs = state["bad_epochs"]


@dataclass
class BestTracker:
    """Track the best monitored value across all phases (Keras ``ModelCheckpoint(save_best_only=True)``)."""

    mode: str = "min"
    best: float | None = None
    epoch: int = -1
    phase: str = ""

    def __post_init__(self) -> None:
        if self.best is None:
            self.best = _initial(self.mode)

    def is_improvement(self, value: float) -> bool:
        return _better(value, self.best, self.mode)

    def update(self, value: float, epoch: int, phase: str) -> None:
        self.best, self.epoch, self.phase = value, epoch, phase

    def state_dict(self) -> dict:
        return asdict(self)

    def load_state_dict(self, state: dict) -> None:
        self.best, self.epoch, self.phase = state["best"], state["epoch"], state["phase"]
