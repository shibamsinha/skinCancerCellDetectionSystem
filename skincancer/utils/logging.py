"""Logging configuration shared by scripts, training and the app."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
_configured = False


def configure_logging(level: int | str = logging.INFO, log_file: str | Path | None = None) -> None:
    """Configure root logging once (console, plus an optional file)."""
    global _configured
    root = logging.getLogger()
    if not _configured:
        root.setLevel(level)
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(logging.Formatter(_FORMAT))
        root.addHandler(console)
        for noisy in ("PIL", "matplotlib", "urllib3", "fsspec", "h5py"):
            logging.getLogger(noisy).setLevel(logging.WARNING)
        _configured = True
    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == path.resolve() for h in root.handlers):
            fh = logging.FileHandler(path)
            fh.setFormatter(logging.Formatter(_FORMAT))
            root.addHandler(fh)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
