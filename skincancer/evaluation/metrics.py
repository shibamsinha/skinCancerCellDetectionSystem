"""Binary classification metrics (positive class = Malignant)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    return {
        "tp": int(((y_true == 1) & (y_pred == 1)).sum()),
        "tn": int(((y_true == 0) & (y_pred == 0)).sum()),
        "fp": int(((y_true == 0) & (y_pred == 1)).sum()),
        "fn": int(((y_true == 1) & (y_pred == 0)).sum()),
    }


def binary_metrics(y_true: np.ndarray, prob_malignant: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """Compute the full metric set used across the project.

    ``prob_malignant`` is the softmax probability of class 1. With two classes,
    ``threshold = 0.5`` is identical to taking the softmax argmax.
    """
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob_malignant, dtype=np.float64)
    if y_true.shape != prob.shape:
        raise ValueError("y_true and prob_malignant must have the same shape")
    y_pred = (prob >= threshold).astype(int)
    c = confusion_counts(y_true, y_pred)
    tp, tn, fp, fn = c["tp"], c["tn"], c["fp"], c["fn"]
    n = tp + tn + fp + fn
    sensitivity = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    precision = _safe_div(tp, tp + fp)
    f1 = _safe_div(2 * precision * sensitivity, precision + sensitivity) if tp else 0.0
    both_classes = len(np.unique(y_true)) == 2
    return {
        "n": n,
        "n_malignant": int(y_true.sum()),
        "n_benign": int(n - y_true.sum()),
        "threshold": float(threshold),
        "accuracy": _safe_div(tp + tn, n),
        "precision": precision,
        "recall": sensitivity,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "balanced_accuracy": (sensitivity + specificity) / 2,
        "npv": _safe_div(tn, tn + fn),
        "roc_auc": float(roc_auc_score(y_true, prob)) if both_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, prob)) if both_classes else float("nan"),
        "majority_class_accuracy": _safe_div(max(int(y_true.sum()), n - int(y_true.sum())), n),
        **c,
    }


class _Resampler:
    """Bootstrap index generator: image-level, or cluster (lesion-level) when ``groups`` is given.

    HAM10000 has several images of the same lesion. Images of one lesion are correlated, so
    resampling images individually understates the uncertainty; resampling whole lesions does not.
    """

    def __init__(self, n: int, groups: np.ndarray | None, seed: int) -> None:
        self.n = n
        self.rng = np.random.default_rng(seed)
        self.clustered = groups is not None
        if self.clustered:
            _, inverse = np.unique(np.asarray(groups), return_inverse=True)
            self.order = np.argsort(inverse, kind="stable")
            self.counts = np.bincount(inverse)
            self.starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]])

    def __call__(self) -> np.ndarray:
        if not self.clustered:
            return self.rng.integers(0, self.n, self.n)
        pick = self.rng.integers(0, len(self.counts), len(self.counts))
        lengths = self.counts[pick]
        offsets = np.arange(lengths.sum()) - np.repeat(np.cumsum(lengths) - lengths, lengths)
        return self.order[np.repeat(self.starts[pick], lengths) + offsets]


def bootstrap_ci(
    y_true: np.ndarray,
    prob_malignant: np.ndarray,
    metrics: tuple[str, ...] = ("accuracy", "sensitivity", "specificity", "f1", "roc_auc"),
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
    groups: np.ndarray | None = None,
) -> dict[str, list[float]]:
    """Percentile bootstrap confidence intervals (lesion-level resampling when ``groups`` is given)."""
    y_true = np.asarray(y_true).astype(int)
    prob = np.asarray(prob_malignant, dtype=np.float64)
    resample = _Resampler(len(y_true), groups, seed)
    draws: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = resample()
        if len(np.unique(y_true[idx])) < 2:
            continue
        res = binary_metrics(y_true[idx], prob[idx])
        for m in metrics:
            draws[m].append(res[m])
    return {
        m: [float(np.nanpercentile(v, 100 * alpha / 2)), float(np.nanpercentile(v, 100 * (1 - alpha / 2)))]
        for m, v in draws.items()
        if v
    }


def _core_metrics(y_true: np.ndarray, prob: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    """The subset of ``binary_metrics`` used by the paired bootstrap (cheaper per round)."""
    y_pred = prob >= threshold
    pos = y_true == 1
    sens = _safe_div(int(np.sum(y_pred & pos)), int(pos.sum()))
    spec = _safe_div(int(np.sum(~y_pred & ~pos)), int((~pos).sum()))
    return {
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "sensitivity": sens,
        "specificity": spec,
        "balanced_accuracy": (sens + spec) / 2,
        "accuracy": float(np.mean(y_pred == pos)),
    }


def paired_bootstrap_difference(
    y_true: np.ndarray,
    prob_a: np.ndarray,
    prob_b: np.ndarray,
    groups: np.ndarray | None = None,
    metrics: tuple[str, ...] = ("roc_auc", "balanced_accuracy", "sensitivity", "specificity"),
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    """Difference ``metric(A) - metric(B)`` of two models scored on the *same* images.

    Both models are evaluated on the same resampled lesions each round, so the
    shared difficulty of the test set cancels out. Returns, per metric, the observed
    difference, its percentile 95% CI, and a two-sided bootstrap p-value (twice the
    smaller tail mass beyond zero). A CI that excludes 0 means the difference is
    unlikely to be resampling noise. No multiple-comparison correction is applied.
    """
    y_true = np.asarray(y_true).astype(int)
    a = np.asarray(prob_a, dtype=np.float64)
    b = np.asarray(prob_b, dtype=np.float64)
    if not (len(y_true) == len(a) == len(b)):
        raise ValueError("y_true, prob_a and prob_b must be aligned on the same images")
    observed_a, observed_b = _core_metrics(y_true, a), _core_metrics(y_true, b)
    resample = _Resampler(len(y_true), groups, seed)
    deltas: dict[str, list[float]] = {m: [] for m in metrics}
    for _ in range(n_boot):
        idx = resample()
        if len(np.unique(y_true[idx])) < 2:
            continue
        ra, rb = _core_metrics(y_true[idx], a[idx]), _core_metrics(y_true[idx], b[idx])
        for m in metrics:
            deltas[m].append(ra[m] - rb[m])
    out: dict[str, dict[str, float]] = {}
    for m, d in deltas.items():
        arr = np.asarray(d)
        p = 2 * min(float(np.mean(arr <= 0)), float(np.mean(arr >= 0)))
        out[m] = {
            "difference": float(observed_a[m] - observed_b[m]),
            "ci_low": float(np.percentile(arr, 100 * alpha / 2)),
            "ci_high": float(np.percentile(arr, 100 * (1 - alpha / 2))),
            "p_value": min(1.0, p),
            "n_boot": int(arr.size),
        }
    return out


def per_diagnosis_rates(dx: np.ndarray, y_pred: np.ndarray) -> dict[str, dict[str, float]]:
    """Fraction predicted Malignant within each original HAM10000 diagnosis.

    For malignant diagnoses this is that diagnosis' sensitivity; for benign
    diagnoses it is the false-positive rate.
    """
    dx = np.asarray(dx)
    y_pred = np.asarray(y_pred).astype(int)
    out = {}
    for d in sorted(np.unique(dx)):
        mask = dx == d
        out[str(d)] = {"n": int(mask.sum()), "predicted_malignant_rate": float(y_pred[mask].mean())}
    return out


def roc_points(y_true: np.ndarray, prob_malignant: np.ndarray, max_points: int = 200) -> dict[str, list[float]]:
    """Down-sampled ROC curve coordinates for plotting."""
    fpr, tpr, _ = roc_curve(np.asarray(y_true).astype(int), np.asarray(prob_malignant))
    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return {"fpr": fpr.round(5).tolist(), "tpr": tpr.round(5).tolist()}
