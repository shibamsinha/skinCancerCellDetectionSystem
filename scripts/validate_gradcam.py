#!/usr/bin/env python3
"""Quantitatively validate Grad-CAM for a trained model on held-out test images.

Writes ``results/<model>/gradcam_validation.json``.

Usage:
    python scripts/validate_gradcam.py --model modified_densenet201 --n 200
"""

from __future__ import annotations

import argparse
import json
import sys

import _bootstrap  # noqa: F401

from skincancer.config import load_config
from skincancer.utils.logging import configure_logging, get_logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="modified_densenet201")
    parser.add_argument("--n", type=int, default=200, help="number of test images (stratified by class)")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    configure_logging()
    log = get_logger("validate_gradcam")

    import pandas as pd

    from skincancer.data.pipeline import load_prepared_splits
    from skincancer.explainability.validation import validate_gradcam
    from skincancer.inference import Predictor
    from skincancer.preprocessing import load_image

    from skincancer.utils.device import resolve_device
    from skincancer.utils.resources import apply_process_limits

    cfg = load_config(args.config)
    device = resolve_device()
    res = cfg.resources  # same laptop-friendly limits as training (may run while training continues)
    apply_process_limits(res.nice, res.torch_threads, res.mps_memory_fraction, device)
    predictor = Predictor.for_model(cfg, args.model, device=str(device))
    model = predictor.model
    splits = load_prepared_splits(cfg)
    test = splits[splits["split"] == "test"]
    per_class = args.n // 2
    sample = pd.concat([test[test.label == c].sample(min(per_class, (test.label == c).sum()), random_state=cfg.seed)
                        for c in (0, 1)])
    images_dir = cfg.resolve(cfg.data.images_dir)
    tensors = [predictor.transform(load_image(images_dir / f"{iid}.jpg")).unsqueeze(0).to(predictor.device)
               for iid in sample["image_id"]]
    trained = [name for name in ("additional", "head") if hasattr(model, name) and getattr(model, name) is not None]
    if not trained:  # models trained end-to-end (GoogleNet): randomise the classifier
        trained = ["classifier"]
    log.info("Validating Grad-CAM for %s on %d test images (target %s; randomising %s)",
             args.model, len(tensors), model.gradcam_target_name, trained)
    result = validate_gradcam(model, model.gradcam_target_name, tensors, trained, seed=cfg.seed)
    out = cfg.results_dir(args.model) / "gradcam_validation.json"
    payload = {"model_name": args.model, "target_layer": model.gradcam_target_name,
               "randomised_modules": trained, **result.to_dict()}
    out.write_text(json.dumps(payload, indent=2))
    log.info("Deletion test: CAM drop %s vs random %s; CAM > random in %s of images",
             result.cam_drop_mean, result.random_drop_mean, result.cam_beats_random_fraction)
    log.info("Randomisation test: mean Spearman rho %.3f (median %.3f)",
             result.randomization_spearman_mean, result.randomization_spearman_median)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
