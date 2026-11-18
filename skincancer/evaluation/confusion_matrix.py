"""Confusion matrix computation and static rendering."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from skincancer.config import CLASS_NAMES  # noqa: E402
from skincancer.evaluation.style import SEQUENTIAL_BLUE, TEXT_PRIMARY, TEXT_SECONDARY  # noqa: E402


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """``cm[i, j]`` = number of samples with true class i predicted as class j."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, title: str, path: str | Path) -> None:
    """Counts with row-normalised percentages, on a single-hue sequential ramp."""
    cm = np.asarray(cm)
    row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    cmap = LinearSegmentedColormap.from_list("seq_blue", SEQUENTIAL_BLUE)
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=150)
    ax.imshow(row_pct, cmap=cmap, vmin=0, vmax=1)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ink = "white" if row_pct[i, j] > 0.55 else TEXT_PRIMARY
            ax.text(j, i - 0.08, f"{cm[i, j]:,}", ha="center", va="center", fontsize=13, fontweight="bold", color=ink)
            ax.text(j, i + 0.18, f"{100 * row_pct[i, j]:.1f}% of row", ha="center", va="center", fontsize=8, color=ink)
    ax.set_xticks(range(len(CLASS_NAMES)), [f"Pred. {c}" for c in CLASS_NAMES], color=TEXT_SECONDARY, fontsize=9)
    ax.set_yticks(range(len(CLASS_NAMES)), [f"True {c}" for c in CLASS_NAMES], color=TEXT_SECONDARY, fontsize=9)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, fontsize=10, color=TEXT_PRIMARY, loc="left")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
