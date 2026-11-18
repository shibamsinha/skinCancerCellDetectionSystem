"""Reproducible, leakage-free train/validation/test splits.

HAM10000 contains 10,015 images of only 7,470 lesions; many lesions were
photographed several times. Splitting by image would put near-identical views
of one lesion in both train and test, inflating every metric. Splits here are
therefore **grouped by ``lesion_id``** and **stratified by diagnosis** (which
also stratifies the binary label), using scikit-learn's ``StratifiedGroupKFold``.

Ratios follow the original notebook (inferred from its batch counts): 20% test,
then 20% of the remainder for validation, i.e. 64/16/20 overall.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from skincancer.utils.logging import get_logger

log = get_logger(__name__)

SPLITS = ("train", "val", "test")
SPLIT_COLUMNS = ["image_id", "lesion_id", "dx", "label", "split"]


class LeakageError(AssertionError):
    """Raised if a lesion appears in more than one split."""


def _holdout(df: pd.DataFrame, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Return (keep_idx, holdout_idx) positions with ~``fraction`` held out."""
    n_splits = max(2, int(round(1.0 / fraction)))
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keep_idx, hold_idx = next(sgkf.split(df, y=df["dx"], groups=df["lesion_id"]))
    return keep_idx, hold_idx


def make_lesion_splits(df: pd.DataFrame, test_fraction: float, val_fraction: float, seed: int) -> pd.DataFrame:
    """Assign each image to train/val/test without splitting any lesion."""
    per_lesion = df.groupby("lesion_id")["dx"].nunique()
    if (per_lesion > 1).any():
        bad = per_lesion[per_lesion > 1].index.tolist()[:5]
        raise ValueError(f"Lesions with conflicting diagnoses (cannot stratify safely): {bad}")

    df = df.reset_index(drop=True)
    rest_idx, test_idx = _holdout(df, test_fraction, seed)
    rest = df.iloc[rest_idx].reset_index(drop=True)
    train_idx, val_idx = _holdout(rest, val_fraction, seed + 1)

    out = df[["image_id", "lesion_id", "dx", "label"]].copy()
    out["split"] = "train"
    out.loc[test_idx, "split"] = "test"
    val_ids = set(rest.iloc[val_idx]["image_id"])
    out.loc[out["image_id"].isin(val_ids), "split"] = "val"
    verify_no_leakage(out)
    return out


def verify_no_leakage(splits: pd.DataFrame) -> None:
    """Raise ``LeakageError`` if any lesion_id occurs in more than one split."""
    n_splits_per_lesion = splits.groupby("lesion_id")["split"].nunique()
    leaked = n_splits_per_lesion[n_splits_per_lesion > 1]
    if len(leaked):
        raise LeakageError(f"{len(leaked)} lesions appear in multiple splits, e.g. {leaked.index[:5].tolist()}")


def summarize_splits(splits: pd.DataFrame) -> dict:
    """Counts and malignant fraction per split."""
    summary = {}
    for name in SPLITS:
        part = splits[splits["split"] == name]
        summary[name] = {
            "images": int(len(part)),
            "lesions": int(part["lesion_id"].nunique()),
            "benign": int((part["label"] == 0).sum()),
            "malignant": int((part["label"] == 1).sum()),
            "malignant_fraction": round(float(part["label"].mean()), 4) if len(part) else 0.0,
            "dx": {k: int(v) for k, v in part["dx"].value_counts().items()},
        }
    return summary


def load_or_create_splits(
    df: pd.DataFrame, path: str | Path, test_fraction: float, val_fraction: float, seed: int
) -> pd.DataFrame:
    """Load a saved split if it matches ``df`` exactly, else create and save one."""
    path = Path(path)
    if path.exists():
        saved = pd.read_csv(path)
        if set(saved["image_id"]) == set(df["image_id"]) and set(saved["split"]) <= set(SPLITS):
            verify_no_leakage(saved)
            log.info("Loaded existing split from %s", path)
            return saved
        log.warning("Saved split %s does not match the current image set; regenerating", path)
    splits = make_lesion_splits(df, test_fraction, val_fraction, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    splits[SPLIT_COLUMNS].to_csv(path, index=False)
    log.info("Saved new lesion-grouped split to %s", path)
    return splits
