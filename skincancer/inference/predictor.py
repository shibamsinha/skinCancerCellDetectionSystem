"""Inference service used by the web app (and usable from Python).

Loads a trained ``best.pt`` checkpoint and applies **exactly** the evaluation
preprocessing (resize -> contrast stretch -> normalise, no augmentation).
Images are processed in memory and never written to disk.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
import torch
from PIL import Image

from skincancer.config import CLASS_NAMES, ProjectConfig
from skincancer.explainability.gradcam import GradCAM, heatmap_rgb, overlay_heatmap
from skincancer.preprocessing.image_loader import load_image
from skincancer.utils.device import resolve_device


class ModelUnavailableError(FileNotFoundError):
    """Raised when a model has no trained checkpoint."""


@dataclass
class Prediction:
    label: str
    class_index: int
    confidence: float  # softmax probability of the predicted class (model confidence, not clinical certainty)
    probabilities: dict[str, float]
    model_name: str
    display_name: str
    inference_ms: float
    input_size: int
    device: str


@dataclass
class Explanation:
    original: np.ndarray  # uint8 (S, S, 3): the uploaded image resized to the model input size
    model_input: np.ndarray  # uint8 (S, S, 3): what the model sees (resized + contrast-stretched)
    heatmap: np.ndarray  # uint8 (S, S, 3)
    overlay: np.ndarray  # uint8 (S, S, 3): heat map blended over ``original``
    cam: np.ndarray  # float32 (S, S) in [0, 1]
    explained_class: str
    target_layer: str
    feature_map_size: tuple[int, int]
    seconds: float
    extra: dict[str, Any] = field(default_factory=dict)


class Predictor:
    """A trained model plus its preprocessing, ready for single-image inference."""

    def __init__(self, checkpoint_path: str | Path, device: str | torch.device = "auto") -> None:
        # Local import to avoid a circular import at package import time.
        from skincancer.evaluation.evaluate import load_trained_model

        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise ModelUnavailableError(f"No trained weights at {self.checkpoint_path}")
        self.device = device if isinstance(device, torch.device) else resolve_device(device)
        self.model, self.transform, ckpt = load_trained_model(self.checkpoint_path, self.device)
        self.model_name: str = ckpt["model_name"]
        self.display_name: str = ckpt["model_config"]["display_name"]
        self.class_names: list[str] = list(ckpt.get("class_names", CLASS_NAMES))
        self.metadata = {k: ckpt[k] for k in ("epoch", "phase", "created_at", "val_metrics", "gradcam_target") if k in ckpt}

    def warmup(self) -> None:
        """Run one dummy prediction and Grad-CAM so GPU kernels are compiled before the first real request."""
        dummy = Image.new("RGB", (self.model.input_size, self.model.input_size), (128, 100, 90))
        self.predict(dummy)
        self.explain(dummy)

    @classmethod
    def for_model(cls, cfg: ProjectConfig, model_name: str, device: str = "auto") -> "Predictor":
        return cls(cfg.checkpoint_dir(model_name) / "best.pt", device)

    # ---------------------------------------------------------------- core
    def _tensor(self, image: Image.Image) -> torch.Tensor:
        return self.transform(image).unsqueeze(0).to(self.device)

    @staticmethod
    def _as_image(image: Image.Image | str | Path | bytes | BinaryIO) -> Image.Image:
        return image if isinstance(image, Image.Image) else load_image(image)

    @torch.no_grad()
    def predict(self, image: Image.Image | str | Path | bytes | BinaryIO) -> Prediction:
        img = self._as_image(image)
        x = self._tensor(img)
        t0 = time.perf_counter()
        probs = torch.softmax(self.model(x), dim=1)[0].float().cpu().numpy()
        if self.device.type == "mps":
            torch.mps.synchronize()
        elapsed = (time.perf_counter() - t0) * 1000
        idx = int(np.argmax(probs))
        return Prediction(
            label=self.class_names[idx],
            class_index=idx,
            confidence=float(probs[idx]),
            probabilities={c: float(p) for c, p in zip(self.class_names, probs)},
            model_name=self.model_name,
            display_name=self.display_name,
            inference_ms=elapsed,
            input_size=self.model.input_size,
            device=str(self.device),
        )

    def explain(self, image: Image.Image | str | Path | bytes | BinaryIO, class_index: int | None = None) -> Explanation:
        """Grad-CAM for ``class_index`` (default: the predicted class)."""
        img = self._as_image(image)
        t0 = time.perf_counter()
        size = self.model.input_size
        original = np.asarray(img.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.uint8)
        model_input = self.transform.preprocess_display(img)
        x = self._tensor(img)
        with GradCAM(self.model, self.model.gradcam_target_layer) as cam_fn:
            res = cam_fn(x, class_index)
        return Explanation(
            original=original,
            model_input=model_input,
            heatmap=heatmap_rgb(res.cam),
            overlay=overlay_heatmap(original, res.cam),
            cam=res.cam,
            explained_class=self.class_names[res.class_index],
            target_layer=self.model.gradcam_target_name,
            feature_map_size=tuple(res.raw_cam.shape),
            seconds=time.perf_counter() - t0,
        )
