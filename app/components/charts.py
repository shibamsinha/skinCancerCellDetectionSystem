"""Interactive Plotly charts for the app.

Conventions (validated palette, see skincancer/evaluation/style.py):
one hue per series, colours follow the entity, hairline grids, a single y-axis
per chart, legends for multi-series charts, and selective direct labels.
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

from skincancer.evaluation.style import (
    GRID,
    REFERENCE_LINE,
    SEQUENTIAL_BLUE,
    SERIES,
    TEXT_PRIMARY,
    TEXT_SECONDARY,
    TRAIN_COLOR,
    VAL_COLOR,
    model_color,
)

MUTED_BAR = "#c9d1db"
_FONT = dict(family="Inter, -apple-system, Segoe UI, Roboto, sans-serif", color=TEXT_PRIMARY, size=13)
CONFIG = {"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"], "responsive": True}


def _layout(fig: go.Figure, title: str = "", height: int = 320, **kw: Any) -> go.Figure:
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", font=dict(size=14, color=TEXT_PRIMARY)))
    fig.update_layout(
        height=height,
        margin=dict(l=8, r=16, t=44 if title else 12, b=8),
        paper_bgcolor="white",
        plot_bgcolor="white",
        font=_FONT,
        hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(color=TEXT_PRIMARY, size=12)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(color=TEXT_SECONDARY, size=12)),
        **kw,
    )
    fig.update_xaxes(showgrid=False, linecolor=GRID, tickfont=dict(color=TEXT_SECONDARY), zeroline=False)
    fig.update_yaxes(gridcolor=GRID, gridwidth=1, linecolor=GRID, tickfont=dict(color=TEXT_SECONDARY), zeroline=False)
    return fig


def probability_bars(probabilities: dict[str, float], predicted: str) -> go.Figure:
    """Horizontal bars; the predicted class is emphasised, the other muted."""
    labels = list(probabilities)
    values = [probabilities[k] for k in labels]
    colors = [SERIES[0] if k == predicted else MUTED_BAR for k in labels]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker=dict(color=colors, cornerradius=4),
        text=[f"{100 * v:.1f}%" for v in values], textposition="outside", cliponaxis=False,
        textfont=dict(color=TEXT_PRIMARY, size=13),
        hovertemplate="%{y}: %{x:.2%}<extra></extra>", width=0.5,
    ))
    fig.add_vline(x=0.5, line=dict(color=REFERENCE_LINE, width=1))
    fig.add_annotation(x=0.5, y=1.08, yref="paper", text="decision threshold 0.5", showarrow=False,
                       font=dict(size=11, color=TEXT_SECONDARY))
    _layout(fig, height=190)
    fig.update_xaxes(range=[0, 1.12], tickformat=".0%", showgrid=True, gridcolor=GRID, dtick=0.25)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_layout(showlegend=False, margin=dict(l=8, r=24, t=28, b=8))
    return fig


def training_curves(history: list[dict[str, Any]], metric: str) -> go.Figure:
    """Training vs validation for ``metric`` in {'loss', 'accuracy'}; phase changes marked."""
    epochs = [h["epoch"] for h in history]
    title = "Cross-entropy loss" if metric == "loss" else "Accuracy"
    fig = go.Figure()
    for key, name, color in ((f"train_{metric}", "Training", TRAIN_COLOR), (f"val_{metric}", "Validation", VAL_COLOR)):
        ys = [h[key] for h in history]
        fig.add_trace(go.Scatter(
            x=epochs, y=ys, name=name, mode="lines+markers", line=dict(color=color, width=2),
            marker=dict(size=7, color=color, line=dict(color="white", width=1.5)),
            customdata=[h["phase"].replace("_", " ") for h in history],
            hovertemplate=f"{name}<br>epoch %{{x}} (%{{customdata}})<br>{title.lower()}: %{{y:.4f}}<extra></extra>",
        ))
    for prev, cur in zip(history, history[1:]):
        if cur["phase"] != prev["phase"]:
            x = cur["epoch"] - 0.5
            fig.add_vline(x=x, line=dict(color=REFERENCE_LINE, width=1))
            fig.add_annotation(x=x, y=1, yref="paper", text=cur["phase"].replace("_", " "), showarrow=False,
                               xanchor="left", yanchor="top", font=dict(size=11, color=TEXT_SECONDARY))
    _layout(fig, title=title, height=300, hovermode="x unified")
    fig.update_xaxes(title=dict(text="Epoch", font=dict(color=TEXT_SECONDARY, size=12)), dtick=2)
    if metric == "accuracy":
        fig.update_yaxes(tickformat=".0%")
    return fig


def confusion_heatmap(matrix: list[list[int]], labels: list[str]) -> go.Figure:
    """Counts with row percentages on a single-hue sequential ramp."""
    totals = [max(sum(r), 1) for r in matrix]
    pct = [[v / totals[i] for v in r] for i, r in enumerate(matrix)]
    text = [[f"<b>{v:,}</b><br>{100 * pct[i][j]:.1f}% of true {labels[i].lower()}" for j, v in enumerate(r)]
            for i, r in enumerate(matrix)]
    n = len(SEQUENTIAL_BLUE) - 1
    scale = [[i / n, c] for i, c in enumerate(SEQUENTIAL_BLUE)]
    fig = go.Figure(go.Heatmap(
        z=pct, x=[f"Predicted {lab}" for lab in labels], y=[f"True {lab}" for lab in labels], text=text,
        texttemplate="%{text}", textfont=dict(size=13), colorscale=scale, zmin=0, zmax=1, showscale=False,
        hovertemplate="%{y} / %{x}<br>%{text}<extra></extra>", xgap=3, ygap=3,
    ))
    _layout(fig, height=300)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    return fig


def roc_comparison(curves: dict[str, dict[str, Any]]) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color=REFERENCE_LINE, width=1),
                             hoverinfo="skip"))
    for name, c in curves.items():
        fig.add_trace(go.Scatter(
            x=c["fpr"], y=c["tpr"], mode="lines", name=f"{c['label']} (AUC {c['auc']:.3f})",
            line=dict(color=model_color(name), width=2),
            hovertemplate=f"{c['label']}<br>FPR %{{x:.3f}}<br>TPR %{{y:.3f}}<extra></extra>",
        ))
    _layout(fig, title="ROC curves on the held-out test set", height=420)
    fig.update_xaxes(title=dict(text="1 − specificity (false-positive rate)", font=dict(color=TEXT_SECONDARY, size=12)),
                     range=[0, 1], showgrid=True, gridcolor=GRID)
    fig.update_yaxes(title=dict(text="Sensitivity (true-positive rate)", font=dict(color=TEXT_SECONDARY, size=12)),
                     range=[0, 1.01])
    fig.update_layout(legend=dict(orientation="v", yanchor="bottom", y=0.02, xanchor="right", x=0.99,
                                  bgcolor="rgba(255,255,255,0.85)"))
    return fig


def metric_bars(names: list[str], labels: list[str], values: list[float], metric_label: str,
                baseline: float | None = None, baseline_label: str = "") -> go.Figure:
    """One metric across models: one series, one colour; values labelled at bar ends."""
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker=dict(color=SERIES[0], cornerradius=4), width=0.55,
        text=[f"{100 * v:.1f}%" for v in values], textposition="outside", cliponaxis=False,
        textfont=dict(color=TEXT_PRIMARY), hovertemplate="%{y}: %{x:.2%}<extra></extra>",
    ))
    if baseline is not None:
        fig.add_vline(x=baseline, line=dict(color=REFERENCE_LINE, width=1))
        fig.add_annotation(x=baseline, y=1.0, yref="paper", yanchor="bottom", text=baseline_label, showarrow=False,
                           font=dict(size=11, color=TEXT_SECONDARY))
    _layout(fig, title=metric_label, height=90 + 44 * len(labels))
    fig.update_xaxes(range=[0, 1.1], tickformat=".0%", showgrid=True, gridcolor=GRID)
    fig.update_yaxes(showgrid=False, autorange="reversed")
    fig.update_layout(showlegend=False)
    return fig
