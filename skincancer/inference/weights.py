"""Fetch published trained checkpoints when they are missing locally.

A fresh clone (or a Streamlit Community Cloud deployment) has no ``checkpoints/``
directory: the weights are too large for git. They are published as GitHub
release assets instead (``configs/default.yaml`` → ``weights``). On first use the
file is streamed to disk, its SHA-256 is checked against the configured digest,
and only then is it moved into place. Locally trained weights are never replaced.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
import threading
from pathlib import Path

from skincancer.config import ProjectConfig
from skincancer.inference.predictor import ModelUnavailableError
from skincancer.utils.logging import get_logger

log = get_logger(__name__)

MAX_DOWNLOAD_BYTES = 500 * 1024 * 1024  # the largest checkpoint is ~97 MB
_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


class WeightsDownloadError(ModelUnavailableError):
    """The published weights could not be downloaded or failed verification."""


def checkpoint_path(cfg: ProjectConfig, model_name: str) -> Path:
    return cfg.checkpoint_dir(model_name) / "best.pt"


def published_url(cfg: ProjectConfig, model_name: str) -> str | None:
    """Download URL of the published checkpoint, or ``None`` if none is configured."""
    w = cfg.weights
    if not w.release_url or model_name not in w.sha256:
        return None
    return f"{w.release_url.rstrip('/')}/{model_name}.pt"


def weights_available(cfg: ProjectConfig, model_name: str) -> bool:
    """True if trained weights exist locally or can be downloaded."""
    return checkpoint_path(cfg, model_name).exists() or published_url(cfg, model_name) is not None


def weights_local(cfg: ProjectConfig, model_name: str) -> bool:
    return checkpoint_path(cfg, model_name).exists()


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(str(path), threading.Lock())


def ensure_weights(cfg: ProjectConfig, model_name: str, timeout: float = 60.0) -> Path:
    """Return the local checkpoint path, downloading and verifying it first if needed.

    Raises ``ModelUnavailableError`` when there are no local weights and none are
    published, and ``WeightsDownloadError`` when the download or verification fails.
    """
    path = checkpoint_path(cfg, model_name)
    if path.exists():
        return path
    url = published_url(cfg, model_name)
    if url is None:
        raise ModelUnavailableError(
            f"No trained weights for '{model_name}'. Train it with `python scripts/train.py --model {model_name}`."
        )
    with _lock_for(path):  # several app sessions may ask for the same model at once
        if path.exists():
            return path
        _download_verified(url, path, cfg.weights.sha256[model_name].lower(), timeout)
    return path


def _download_verified(url: str, dest: Path, expected_sha256: str, timeout: float) -> None:
    import requests  # local import: only needed when weights are missing

    dest.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    size = 0
    fd, tmp_name = tempfile.mkstemp(dir=dest.parent, prefix=f".{dest.name}.", suffix=".download")
    tmp = Path(tmp_name)
    log.info("Downloading trained weights from %s", url)
    try:
        with os.fdopen(fd, "wb") as fh, requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=1 << 20):
                size += len(chunk)
                if size > MAX_DOWNLOAD_BYTES:
                    raise WeightsDownloadError(f"{url} is larger than {MAX_DOWNLOAD_BYTES // 2**20} MB; refusing it")
                fh.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise WeightsDownloadError(
                f"Checksum mismatch for {url}: expected {expected_sha256[:12]}…, got {digest.hexdigest()[:12]}…"
            )
        os.replace(tmp, dest)
        log.info("Saved verified weights to %s (%.1f MB)", dest, size / 1e6)
    except requests.RequestException as exc:
        raise WeightsDownloadError(f"Could not download the trained weights from {url}: {exc}") from exc
    finally:
        tmp.unlink(missing_ok=True)
