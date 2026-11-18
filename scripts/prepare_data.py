#!/usr/bin/env python3
"""Validate HAM10000 images and create the reproducible lesion-grouped split.

Usage:
    python scripts/prepare_data.py [--config configs/default.yaml] [--force-resplit]
"""

from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401

from skincancer.config import load_config
from skincancer.data.pipeline import prepare_dataset
from skincancer.data.splits import summarize_splits
from skincancer.utils.logging import configure_logging


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None, help="path to YAML config (default: configs/default.yaml)")
    parser.add_argument("--force-resplit", action="store_true", help="discard the saved split and regenerate it")
    args = parser.parse_args(argv)
    configure_logging()
    cfg = load_config(args.config)
    splits = prepare_dataset(cfg, force_resplit=args.force_resplit)
    print(json.dumps(summarize_splits(splits), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
