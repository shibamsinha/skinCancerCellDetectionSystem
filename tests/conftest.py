"""Shared fixtures: a tiny synthetic HAM10000-like dataset and matching config."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skincancer.config import load_config  # noqa: E402

DX_CYCLE = ["nv", "nv", "nv", "bkl", "mel", "bcc", "akiec", "nv", "df", "vasc", "mel", "nv"]


@pytest.fixture(scope="session")
def project_cfg():
    return load_config()


def _make_dataset(root: Path, n_lesions: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic images: malignant lesions are dark blobs, benign ones light blobs."""
    rng = np.random.default_rng(seed)
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    img_no = 0
    for lesion in range(n_lesions):
        dx = DX_CYCLE[lesion % len(DX_CYCLE)]
        malignant = dx in {"mel", "bcc", "akiec"}
        for _ in range(1 + lesion % 3):  # 1-3 images per lesion, like HAM10000
            yy, xx = np.mgrid[0:96, 0:128]
            cy, cx = rng.integers(30, 66), rng.integers(40, 88)
            mask = ((yy - cy) ** 2 + (xx - cx) ** 2) < rng.integers(200, 500)
            img = np.full((96, 128, 3), [205, 160, 150], dtype=np.float32)
            img[mask] = [60, 35, 30] if malignant else [170, 110, 90]
            img += rng.normal(0, 8, img.shape)
            image_id = f"ISIC_{img_no:07d}"
            Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(images_dir / f"{image_id}.jpg", quality=92)
            rows.append({"lesion_id": f"HAM_{lesion:07d}", "image_id": image_id, "dx": dx, "dx_type": "histo",
                         "age": 50.0, "sex": "male", "localization": "back", "dataset": "synthetic"})
            img_no += 1
    df = pd.DataFrame(rows)
    df.to_csv(root / "metadata.csv", index=False)
    return df


@pytest.fixture()
def synthetic_data(tmp_path):
    df = _make_dataset(tmp_path)
    return tmp_path, df


@pytest.fixture()
def synthetic_cfg(synthetic_data):
    """Config pointing every path at the synthetic dataset; CPU, no workers, short phases."""
    root, _ = synthetic_data
    overrides = {
        "data": {
            "images_dir": str(root / "images"),
            "metadata_csv": str(root / "metadata.csv"),
            "splits_csv": str(root / "splits.csv"),
            "validation_report": str(root / "results" / "data_validation.json"),
        },
        "training": {"device": "cpu", "num_workers": 0, "batch_size": 8, "log_every_n_batches": 0},
        "paths": {"checkpoints_dir": str(root / "checkpoints"), "results_dir": str(root / "results")},
        "resources": {"nice": 0, "torch_threads": 0, "cooldown_seconds_per_batch": 0.0, "thermal_guard": False},
    }
    cfg = load_config(overrides=overrides)
    return cfg
