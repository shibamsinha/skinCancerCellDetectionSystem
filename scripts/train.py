#!/usr/bin/env python3
"""Train one model (phase-wise), then evaluate it on the held-out test split.

Usage:
    python scripts/train.py --model modified_densenet201
    python scripts/train.py --model modified_densenet201 --resume      # continue an interrupted run
    python scripts/train.py --model mobilenet_v2 --no-evaluate
    python scripts/train.py --list
"""

from __future__ import annotations

import argparse
import sys

import _bootstrap  # noqa: F401

from skincancer.config import load_config
from skincancer.utils.logging import configure_logging, get_logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", help="model key from configs/default.yaml (e.g. modified_densenet201)")
    parser.add_argument("--config", default=None)
    parser.add_argument("--resume", action="store_true", help="resume from checkpoints/<model>/last.pt")
    parser.add_argument("--no-evaluate", action="store_true", help="skip test-set evaluation after training")
    parser.add_argument("--num-workers", type=int, default=None, help="override DataLoader workers")
    parser.add_argument("--epochs", type=int, default=None, help="override epochs of every phase (for quick runs)")
    parser.add_argument("--list", action="store_true", help="list available models and exit")
    args = parser.parse_args(argv)
    configure_logging()
    log = get_logger("train")

    cfg = load_config(args.config)
    if args.list or not args.model:
        for name, m in cfg.models.items():
            print(f"{name:22s} {m.display_name}  (input {m.input_size}px, phases: {', '.join(p.name for p in m.phases)})")
        return 0 if args.list else 2
    if args.epochs is not None:
        for p in cfg.model(args.model).phases:
            p.epochs = args.epochs

    from skincancer.evaluation.evaluate import evaluate_model
    from skincancer.training.trainer import Trainer

    try:
        trainer = Trainer(cfg, args.model, num_workers=args.num_workers)
        trainer.fit(resume=args.resume)
        if not args.no_evaluate:
            evaluate_model(cfg, args.model, num_workers=args.num_workers)
    except KeyboardInterrupt:
        log.warning("Interrupted. Resume with: python scripts/train.py --model %s --resume", args.model)
        return 130
    except (FileNotFoundError, ValueError) as exc:
        log.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
