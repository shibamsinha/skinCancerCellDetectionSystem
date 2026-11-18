"""Evaluate a trained checkpoint on the held-out test split and save artefacts.

Outputs in ``results/<model>/``:
    metrics.json         test metrics, 95% bootstrap CIs, confusion matrix, ROC points,
                         per-diagnosis rates, checkpoint provenance
    predictions.csv      per-image probabilities and predictions
    confusion_matrix.png, roc_curve.png, training_curves.png
and an aggregated ``results/summary.json`` across all evaluated models.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from skincancer.config import ProjectConfig, model_config_from_dict, preprocessing_from_dict
from skincancer.data.dataset import LesionDataset
from skincancer.data.pipeline import load_prepared_splits
from skincancer.evaluation.confusion_matrix import confusion_matrix, plot_confusion_matrix
from skincancer.evaluation.metrics import (
    binary_metrics,
    bootstrap_ci,
    paired_bootstrap_difference,
    per_diagnosis_rates,
    roc_points,
)
from skincancer.evaluation.style import MODEL_ORDER
from skincancer.evaluation.visualization import plot_roc_curves, plot_training_curves
from skincancer.models import build_model
from skincancer.preprocessing.transforms import ImageTransform
from skincancer.training.checkpointing import load_checkpoint
from skincancer.training.trainer import predict_loader
from skincancer.utils.device import resolve_device
from skincancer.utils.logging import get_logger
from skincancer.utils.resources import ThermalGovernor

log = get_logger(__name__)

CI_METHOD = "95% percentile bootstrap, 1,000 lesion-level resamples (images of one lesion stay together)"
PAIRED_METHOD = ("paired lesion-level bootstrap, 2,000 resamples; difference = primary model minus comparison "
                 "model on the same test images; two-sided bootstrap p-value, no multiple-comparison correction")


def load_trained_model(checkpoint_path: str | Path, device: torch.device):
    """Rebuild a model from ``best.pt`` (no ImageNet download; weights come from the checkpoint)."""
    ckpt = load_checkpoint(checkpoint_path)
    model_cfg = model_config_from_dict(ckpt["model_config"])
    model = build_model(model_cfg, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.to(device).eval()
    contrast = preprocessing_from_dict(ckpt["contrast_stretch"])
    transform = ImageTransform(model.input_size, model.normalization, contrast, None)
    return model, transform, ckpt


def evaluate_model(cfg: ProjectConfig, model_name: str, split: str = "test", num_workers: int | None = None) -> dict[str, Any]:
    device = resolve_device(cfg.training.device)
    ckpt_path = cfg.checkpoint_dir(model_name) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No trained checkpoint for '{model_name}' at {ckpt_path}. Train it first.")
    model, transform, ckpt = load_trained_model(ckpt_path, device)

    splits = load_prepared_splits(cfg)
    part = splits[splits["split"] == split].reset_index(drop=True)
    if part.empty:
        raise ValueError(f"Split '{split}' is empty")
    loader = DataLoader(
        LesionDataset(part, cfg.resolve(cfg.data.images_dir), transform),
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=cfg.training.num_workers if num_workers is None else num_workers,
    )
    log.info("Evaluating %s on %s split (%d images) using %s", model_name, split, len(part), device)
    res = cfg.resources
    governor = ThermalGovernor(cooldown_seconds=res.cooldown_seconds_per_batch,
                               fair_multiplier=res.fair_cooldown_multiplier,
                               max_pause_seconds=res.max_thermal_pause_seconds)
    y_true, prob, loss = predict_loader(model, loader, device, governor)
    y_pred = (prob >= 0.5).astype(int)

    metrics = binary_metrics(y_true, prob)
    metrics["loss"] = loss
    cm = confusion_matrix(y_true, y_pred)
    out_dir = cfg.results_dir(model_name)
    out_dir.mkdir(parents=True, exist_ok=True)
    display = cfg.models[model_name].display_name if model_name in cfg.models else model_name

    result = {
        "model_name": model_name,
        "display_name": display,
        "split": split,
        "metrics": metrics,
        "ci95": bootstrap_ci(y_true, prob, seed=cfg.seed, groups=part["lesion_id"].to_numpy()),
        "ci_method": CI_METHOD,
        "confusion_matrix": {"labels": ["Benign", "Malignant"], "matrix": cm.tolist()},
        "per_diagnosis": per_diagnosis_rates(part["dx"].to_numpy(), y_pred),
        "roc": roc_points(y_true, prob),
        "checkpoint": {
            "path": _display_path(ckpt_path, cfg),
            "epoch": ckpt["epoch"],
            "phase": ckpt["phase"],
            "created_at": ckpt["created_at"],
            "val_metrics": ckpt.get("val_metrics", {}),
            "gradcam_target": ckpt.get("gradcam_target"),
            "input_size": ckpt["input_size"],
        },
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    pd.DataFrame({
        "image_id": part["image_id"], "lesion_id": part["lesion_id"], "dx": part["dx"],
        "label": y_true, "prob_malignant": np.round(prob, 6), "prediction": y_pred,
    }).to_csv(out_dir / "predictions.csv", index=False)

    plot_confusion_matrix(cm, f"{display}: test confusion matrix", out_dir / "confusion_matrix.png")
    plot_roc_curves({model_name: {**result["roc"], "auc": metrics["roc_auc"], "label": display}},
                    out_dir / "roc_curve.png", title=f"{display}: ROC (test)")
    hist_path = out_dir / "history.json"
    if hist_path.exists():
        plot_training_curves(json.loads(hist_path.read_text()), f"{display}: training history", out_dir / "training_curves.png")

    log.info("%s | acc=%.4f bal_acc=%.4f sens=%.4f spec=%.4f prec=%.4f f1=%.4f auc=%.4f",
             model_name, metrics["accuracy"], metrics["balanced_accuracy"], metrics["sensitivity"],
             metrics["specificity"], metrics["precision"], metrics["f1"], metrics["roc_auc"])
    update_summary(cfg)
    return result


def _display_path(path: Path, cfg: ProjectConfig) -> str:
    """Project-relative path when possible (keeps absolute user paths out of reports)."""
    try:
        return str(path.resolve().relative_to(cfg.resolve(".").resolve()))
    except ValueError:
        return path.name


def update_summary(cfg: ProjectConfig) -> dict[str, Any]:
    """Aggregate every ``results/<model>/metrics.json`` into ``results/summary.json``."""
    base = cfg.results_dir()
    rows, curves = {}, {}
    names = [n for n in MODEL_ORDER if n in cfg.models] + [n for n in cfg.models if n not in MODEL_ORDER]
    for name in names:
        path = base / name / "metrics.json"
        if not path.exists():
            continue
        res = json.loads(path.read_text())
        rows[name] = {"display_name": res["display_name"], "metrics": res["metrics"], "ci95": res["ci95"],
                      "ci_method": res.get("ci_method", "image-level bootstrap"),
                      "evaluated_at": res["evaluated_at"], "checkpoint": res["checkpoint"]}
        curves[name] = {**res["roc"], "auc": res["metrics"]["roc_auc"], "label": res["display_name"]}
    summary = {"generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"), "models": rows,
               "paired_vs_primary": paired_comparisons(cfg, list(rows))}
    (base / "summary.json").write_text(json.dumps(summary, indent=2))
    if curves:
        plot_roc_curves(curves, base / "roc_comparison.png")
    return summary


def paired_comparisons(cfg: ProjectConfig, names: list[str]) -> dict[str, Any] | None:
    """Paired bootstrap of the primary model against every other evaluated model.

    Uses the saved ``predictions.csv`` files (same test images for every model), so
    no inference is repeated.
    """
    base, primary = cfg.results_dir(), cfg.primary_model
    ref_path = base / primary / "predictions.csv"
    if not ref_path.exists():
        return None
    ref = pd.read_csv(ref_path)
    comparisons: dict[str, Any] = {}
    for name in names:
        path = base / name / "predictions.csv"
        if name == primary or not path.exists():
            continue
        other = pd.read_csv(path)[["image_id", "prob_malignant"]]
        merged = ref.merge(other, on="image_id", suffixes=("_primary", "_other"), validate="one_to_one")
        if len(merged) != len(ref) or len(merged) != len(other):
            log.warning("Skipping paired comparison with %s: predictions cover different images", name)
            continue
        comparisons[name] = paired_bootstrap_difference(
            merged["label"].to_numpy(), merged["prob_malignant_primary"].to_numpy(),
            merged["prob_malignant_other"].to_numpy(), groups=merged["lesion_id"].to_numpy(), seed=cfg.seed)
    return {"reference": primary, "method": PAIRED_METHOD, "comparisons": comparisons}
