#!/usr/bin/env python3
"""Evaluate trained checkpoints on the held-out test split.

Usage:
    python scripts/evaluate.py --model modified_densenet201
    python scripts/evaluate.py --all            # every model that has checkpoints/<model>/best.pt
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from skincancer.config import load_config
from skincancer.utils.logging import configure_logging, get_logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model")
    group.add_argument("--all", action="store_true")
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args(argv)
    configure_logging()
    log = get_logger("evaluate")

    from skincancer.evaluation.evaluate import evaluate_model
    from skincancer.utils.device import resolve_device
    from skincancer.utils.resources import apply_process_limits

    cfg = load_config(args.config)
    res = cfg.resources  # same laptop-friendly limits as training
    apply_process_limits(res.nice, res.torch_threads, res.mps_memory_fraction, resolve_device(cfg.training.device))
    names = list(cfg.models) if args.all else [args.model]
    status = 0
    for name in names:
        if args.all and not (cfg.checkpoint_dir(name) / "best.pt").exists():
            log.info("Skipping %s (no trained checkpoint)", name)
            continue
        try:
            evaluate_model(cfg, name, split=args.split, num_workers=args.num_workers)
        except (FileNotFoundError, ValueError) as exc:
            log.error("%s: %s", name, exc)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
