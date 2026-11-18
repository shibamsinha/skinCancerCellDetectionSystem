"""Static (PNG) charts: training curves and ROC curves.

The Streamlit app draws interactive versions of the same data with Plotly;
these PNGs are for the README, reports and offline inspection.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from skincancer.evaluation.style import (  # noqa: E402
    GRID,
    REFERENCE_LINE,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TRAIN_COLOR,
    VAL_COLOR,
    model_color,
)


def _style_axes(ax: plt.Axes, title: str, ylabel: str) -> None:
    ax.set_title(title, fontsize=10, color=TEXT_PRIMARY, loc="left")
    ax.set_xlabel("Epoch", fontsize=9, color=TEXT_SECONDARY)
    ax.set_ylabel(ylabel, fontsize=9, color=TEXT_SECONDARY)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=8, length=0)


def plot_training_curves(history: list[dict[str, Any]], title: str, path: str | Path) -> None:
    """Two separate panels (loss, accuracy); never a dual axis."""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.6), dpi=150)
    for ax, key, label in ((axes[0], "loss", "Cross-entropy loss"), (axes[1], "accuracy", "Accuracy")):
        tr = [h[f"train_{key}"] for h in history]
        va = [h[f"val_{key}"] for h in history]
        ax.plot(epochs, tr, color=TRAIN_COLOR, linewidth=2, marker="o", markersize=3.5, label="Training")
        ax.plot(epochs, va, color=VAL_COLOR, linewidth=2, marker="o", markersize=3.5, label="Validation")
        ax.annotate(f"{va[-1]:.3f}", (epochs[-1], va[-1]), textcoords="offset points", xytext=(6, 0),
                    fontsize=8, color=TEXT_PRIMARY, va="center")
        _mark_phases(ax, history)
        _style_axes(ax, label, label)
        ax.legend(frameon=False, fontsize=8, labelcolor=TEXT_SECONDARY)
    fig.suptitle(title, fontsize=11, color=TEXT_PRIMARY, x=0.01, ha="left")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def _mark_phases(ax: plt.Axes, history: list[dict[str, Any]]) -> None:
    """Hairline at each phase change, labelled with the new phase name."""
    for prev, cur in zip(history, history[1:]):
        if cur["phase"] != prev["phase"]:
            x = cur["epoch"] - 0.5
            ax.axvline(x, color=REFERENCE_LINE, linewidth=0.8)
            ax.text(x, 1.0, f" {cur['phase'].replace('_', ' ')}", transform=ax.get_xaxis_transform(),
                    fontsize=7, color=TEXT_SECONDARY, va="top")


def plot_roc_curves(curves: dict[str, dict[str, Any]], path: str | Path, title: str = "ROC curves (test set)") -> None:
    """``curves``: model_name -> {"fpr": [...], "tpr": [...], "auc": float, "label": str}."""
    fig, ax = plt.subplots(figsize=(5.2, 4.6), dpi=150)
    ax.plot([0, 1], [0, 1], color=REFERENCE_LINE, linewidth=1, label="Chance")
    for name, c in curves.items():
        ax.plot(c["fpr"], c["tpr"], color=model_color(name), linewidth=2, label=f"{c['label']} (AUC {c['auc']:.3f})")
    _style_axes(ax, title, "Sensitivity (true-positive rate)")
    ax.set_xlabel("1 - Specificity (false-positive rate)", fontsize=9, color=TEXT_SECONDARY)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.01)
    ax.legend(frameon=False, fontsize=7.5, labelcolor=TEXT_SECONDARY, loc="lower right")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor="white")
    plt.close(fig)
