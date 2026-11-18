"""Validate that every metadata row points to a readable, decodable image."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd
from PIL import Image

from skincancer.utils.logging import get_logger

log = get_logger(__name__)


@dataclass
class ValidationReport:
    total_rows: int
    valid: int
    missing: list[str] = field(default_factory=list)
    corrupted: list[dict[str, str]] = field(default_factory=list)
    unexpected_size: list[dict[str, object]] = field(default_factory=list)
    orphan_files: int = 0  # images on disk without a metadata row

    @property
    def ok(self) -> bool:
        return not self.missing and not self.corrupted

    def to_dict(self) -> dict:
        d = asdict(self)
        d["ok"] = self.ok
        return d


def _check(path: Path) -> tuple[str, str | None, tuple[int, int] | None]:
    if not path.exists():
        return "missing", None, None
    try:
        with Image.open(path) as img:
            img.load()  # full decode, catches truncated files
            return "ok", None, img.size
    except Exception as exc:  # noqa: BLE001 - any decode failure marks the file corrupt
        return "corrupted", f"{type(exc).__name__}: {exc}", None


def validate_images(
    df: pd.DataFrame,
    images_dir: str | Path,
    expected_size: tuple[int, int] | None = (600, 450),
    workers: int = 8,
) -> tuple[pd.DataFrame, ValidationReport]:
    """Check every image referenced by ``df``; return valid rows and a report.

    Missing and corrupted images are excluded from the returned frame and
    listed in the report. Non-standard dimensions are only reported, since
    the pipeline resizes all images.
    """
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {images_dir}. Run `python scripts/download_ham10000.py` first."
        )
    paths = [images_dir / f"{iid}.jpg" for iid in df["image_id"]]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(_check, paths))

    report = ValidationReport(total_rows=len(df), valid=0)
    keep = []
    for iid, (status, err, size) in zip(df["image_id"], results):
        if status == "missing":
            report.missing.append(iid)
        elif status == "corrupted":
            report.corrupted.append({"image_id": iid, "error": err or ""})
        else:
            keep.append(True)
            if expected_size and size != expected_size:
                report.unexpected_size.append({"image_id": iid, "size": list(size)})
            continue
        keep.append(False)
    valid_df = df.loc[keep].reset_index(drop=True)
    report.valid = len(valid_df)
    on_disk = {p.stem for p in images_dir.glob("*.jpg")}
    report.orphan_files = len(on_disk - set(df["image_id"]))
    log.info(
        "Image validation: %d/%d valid, %d missing, %d corrupted, %d non-standard size, %d orphan files",
        report.valid, report.total_rows, len(report.missing), len(report.corrupted),
        len(report.unexpected_size), report.orphan_files,
    )
    return valid_df, report
