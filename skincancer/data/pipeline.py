"""End-to-end dataset preparation: metadata -> validation -> lesion-grouped split."""

from __future__ import annotations

import json

import pandas as pd

from skincancer.config import ProjectConfig
from skincancer.data.metadata import class_distribution, load_metadata
from skincancer.data.splits import load_or_create_splits, summarize_splits, verify_no_leakage
from skincancer.data.validation import validate_images
from skincancer.utils.logging import get_logger

log = get_logger(__name__)


def prepare_dataset(cfg: ProjectConfig, force_resplit: bool = False) -> pd.DataFrame:
    """Validate all images, create (or load) the split, and write a data report.

    Returns the split table: one row per valid image with ``split`` in
    {train, val, test}.
    """
    meta = load_metadata(cfg.data, cfg.resolve(cfg.data.metadata_csv))
    valid, report = validate_images(meta, cfg.resolve(cfg.data.images_dir))
    splits_path = cfg.resolve(cfg.data.splits_csv)
    if force_resplit and splits_path.exists():
        splits_path.unlink()
    splits = load_or_create_splits(valid, splits_path, cfg.data.test_fraction, cfg.data.val_fraction, cfg.seed)

    report_path = cfg.resolve(cfg.data.validation_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "validation": report.to_dict(),
        "class_distribution": class_distribution(valid),
        "splits": summarize_splits(splits),
        "binary_mapping": {"malignant": cfg.data.malignant_dx, "benign": cfg.data.benign_dx},
        "seed": cfg.seed,
    }
    report_path.write_text(json.dumps(payload, indent=2))
    log.info("Data report written to %s", report_path)
    return splits


def load_prepared_splits(cfg: ProjectConfig) -> pd.DataFrame:
    """Load the saved split; prepare the dataset first if it does not exist."""
    path = cfg.resolve(cfg.data.splits_csv)
    if not path.exists():
        log.info("No saved split found; preparing dataset")
        return prepare_dataset(cfg)
    splits = pd.read_csv(path)
    verify_no_leakage(splits)
    return splits
