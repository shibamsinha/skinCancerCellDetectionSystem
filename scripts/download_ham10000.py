#!/usr/bin/env python3
"""Download and extract the HAM10000 dataset from Harvard Dataverse.

Source: Tschandl, P., Rosendahl, C. & Kittler, H. "The HAM10000 dataset, a large
collection of multi-source dermatoscopic images of common pigmented skin lesions."
Sci. Data 5, 180161 (2018). Dataverse DOI: 10.7910/DVN/DBW86T.
Licence: Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).

The script:
  1. queries the Dataverse API for the file list (and MD5 checksums),
  2. downloads the two image archives and the metadata table (resumable),
  3. verifies MD5 checksums,
  4. extracts all JPEGs into ``<output>/images/`` and writes
     ``<output>/HAM10000_metadata.csv``,
  5. deletes the archives (unless ``--keep-zips``).

Usage:
    python scripts/download_ham10000.py --output data/HAM10000
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

DATAVERSE = "https://dataverse.harvard.edu"
DOI = "doi:10.7910/DVN/DBW86T"
WANTED = {
    "HAM10000_metadata.tab": "metadata",
    "HAM10000_images_part_1.zip": "zip",
    "HAM10000_images_part_2.zip": "zip",
}
CHUNK = 1 << 20  # 1 MiB
MAX_RETRIES = 8
# Dataverse rejects the default python-requests User-Agent with HTTP 403.
HEADERS = {"User-Agent": "skin-cancer-reconstruction/1.0 (dataset download script)"}

log = logging.getLogger("download_ham10000")


def list_files() -> list[dict]:
    """Return Dataverse file records for the HAM10000 dataset."""
    url = f"{DATAVERSE}/api/datasets/:persistentId/"
    resp = requests.get(url, params={"persistentId": DOI}, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    return resp.json()["data"]["latestVersion"]["files"]


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def download_small(file_id: int, dest: Path, original: bool = False) -> None:
    """Download a small Dataverse file in one request."""
    url = f"{DATAVERSE}/api/access/datafile/{file_id}"
    params = {"format": "original"} if original else {}
    resp = requests.get(url, params=params, headers=HEADERS, timeout=120)
    resp.raise_for_status()
    dest.write_bytes(resp.content)


def _download_segment(url: str, part: Path, start: int, end: int, progress: dict, lock: threading.Lock) -> None:
    """Download bytes [start, end] (inclusive) into ``part``, resuming and retrying."""
    length = end - start + 1
    for attempt in range(1, MAX_RETRIES + 1):
        have = part.stat().st_size if part.exists() else 0
        if have >= length:
            return
        headers = dict(HEADERS, Range=f"bytes={start + have}-{end}")
        try:
            with requests.get(url, headers=headers, stream=True, timeout=120) as resp:
                if resp.status_code != 206:
                    raise IOError(f"expected HTTP 206 for ranged request, got {resp.status_code}")
                with part.open("ab") as fh:
                    for chunk in resp.iter_content(CHUNK):
                        fh.write(chunk)
                        with lock:
                            progress["done"] += len(chunk)
            return
        except (requests.RequestException, IOError) as exc:
            log.warning("%s attempt %d/%d failed: %s", part.name, attempt, MAX_RETRIES, exc)
            time.sleep(min(30, 2 ** attempt))
    raise RuntimeError(f"segment {part.name} failed after {MAX_RETRIES} attempts")


def download_parallel(file_id: int, dest: Path, size: int, workers: int) -> None:
    """Download a large file with ``workers`` concurrent HTTP range requests.

    Each segment is written to ``<dest>.partK`` so an interrupted download can be
    resumed by re-running the script. Segments are concatenated at the end.
    """
    if dest.exists() and dest.stat().st_size == size:
        log.info("%s already complete", dest.name)
        return
    url = f"{DATAVERSE}/api/access/datafile/{file_id}"
    bounds = [(i * size // workers, (i + 1) * size // workers - 1) for i in range(workers)]
    parts = [dest.with_name(f"{dest.name}.part{i}") for i in range(workers)]
    progress = {"done": sum(p.stat().st_size for p in parts if p.exists())}
    lock = threading.Lock()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_download_segment, url, p, s, e, progress, lock) for p, (s, e) in zip(parts, bounds)]
        while not all(f.done() for f in futures):
            time.sleep(10)
            with lock:
                done = progress["done"]
            log.info("%s: %5.1f%% (%.0f / %.0f MB)", dest.name, 100 * done / size, done / 1e6, size / 1e6)
        for f in futures:
            f.result()  # re-raise any segment failure
    with dest.open("wb") as out:
        for p in parts:
            with p.open("rb") as src:
                shutil.copyfileobj(src, out)
    if dest.stat().st_size != size:
        raise RuntimeError(f"{dest.name}: size {dest.stat().st_size} != expected {size}")
    for p in parts:
        p.unlink()


def extract_images(archive: Path, image_dir: Path) -> int:
    """Extract JPEG members of ``archive`` flat into ``image_dir``; return count."""
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name
            if member.is_dir() or not name.lower().endswith(".jpg") or name.startswith("._"):
                continue
            target = image_dir / name
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst)
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, default=Path("data/HAM10000"))
    parser.add_argument("--keep-zips", action="store_true", help="keep archives after extraction")
    parser.add_argument("--workers", type=int, default=8, help="parallel HTTP range connections per archive")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    out: Path = args.output
    image_dir = out / "images"
    zip_dir = out / "_archives"
    image_dir.mkdir(parents=True, exist_ok=True)
    zip_dir.mkdir(parents=True, exist_ok=True)

    records = {f["dataFile"]["filename"]: f["dataFile"] for f in list_files()}
    missing = [n for n in WANTED if n not in records]
    if missing:
        log.error("Dataverse listing is missing expected files: %s", missing)
        return 1

    for name, kind in WANTED.items():
        rec = records[name]
        if kind == "metadata":
            dest = out / "HAM10000_metadata.csv"
            log.info("Downloading metadata (original format) -> %s", dest)
            download_small(rec["id"], dest, original=True)
            continue
        dest = zip_dir / name
        log.info("Downloading %s (%.1f MB) with %d connections", name, rec["filesize"] / 1e6, args.workers)
        download_parallel(rec["id"], dest, size=rec["filesize"], workers=args.workers)
        expected_md5 = (rec.get("checksum") or {}).get("value") or rec.get("md5")
        if expected_md5:
            actual = md5sum(dest)
            if actual != expected_md5:
                log.error("MD5 mismatch for %s: expected %s got %s", name, expected_md5, actual)
                dest.unlink()
                return 1
            log.info("MD5 verified for %s", name)
        n = extract_images(dest, image_dir)
        log.info("Extracted %d images from %s", n, name)
        if not args.keep_zips:
            dest.unlink()

    if not args.keep_zips:
        shutil.rmtree(zip_dir, ignore_errors=True)
    total = sum(1 for _ in image_dir.glob("*.jpg"))
    log.info("Done. %d images in %s; metadata at %s", total, image_dir, out / "HAM10000_metadata.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
