"""HAM10000 metadata loading and the 7-class -> binary label mapping."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from skincancer.config import CLASS_NAMES, DataConfig

REQUIRED_COLUMNS = ("lesion_id", "image_id", "dx", "dx_type", "age", "sex", "localization")

DX_FULL_NAMES: dict[str, str] = {
    "nv": "Melanocytic nevi",
    "mel": "Melanoma",
    "bkl": "Benign keratosis-like lesions",
    "bcc": "Basal cell carcinoma",
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "vasc": "Vascular lesions",
    "df": "Dermatofibroma",
}


class MetadataError(ValueError):
    """Raised when the metadata file is missing or malformed."""


def load_metadata(cfg: DataConfig, csv_path: str | Path) -> pd.DataFrame:
    """Load HAM10000 metadata and attach binary labels.

    Adds columns:
        ``label``       0 = Benign, 1 = Malignant
        ``label_name``  "Benign" / "Malignant"
        ``dx_name``     full diagnosis name

    The mapping (malignant = mel, bcc, akiec) was verified against the
    original report's Figs. 2 and 3 (see RECONSTRUCTION_SPECIFICATION.md §2.3).
    """
    path = Path(csv_path)
    if not path.exists():
        raise MetadataError(
            f"Metadata file not found: {path}. Run `python scripts/download_ham10000.py` first."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise MetadataError(f"Metadata is missing required columns: {missing}")

    df = df.copy()
    df["dx"] = df["dx"].astype(str).str.strip().str.lower()
    known = set(cfg.malignant_dx) | set(cfg.benign_dx)
    unknown = sorted(set(df["dx"]) - known)
    if unknown:
        raise MetadataError(f"Diagnoses not covered by the binary mapping: {unknown}")

    dup = df["image_id"].duplicated()
    if dup.any():
        raise MetadataError(f"Duplicate image_id values in metadata: {df.loc[dup, 'image_id'].tolist()[:10]}")

    df["label"] = df["dx"].isin(cfg.malignant_dx).astype(int)
    df["label_name"] = df["label"].map(dict(enumerate(CLASS_NAMES)))
    df["dx_name"] = df["dx"].map(DX_FULL_NAMES).fillna(df["dx"])
    return df


def class_distribution(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Counts per binary label and per diagnosis (for reports and the app)."""
    return {
        "binary": {str(k): int(v) for k, v in df["label_name"].value_counts().items()},
        "dx": {str(k): int(v) for k, v in df["dx"].value_counts().items()},
    }
