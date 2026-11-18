"""Grad-CAM (Selvaraju et al., ICCV 2017) for the reconstructed models.

For a target class c and a convolutional layer with feature maps A^k:

    alpha_k^c = mean_{i,j} d y^c / d A^k_{ij}        (global-average-pooled gradients)
    L^c       = ReLU( sum_k alpha_k^c * A^k )        (class-discriminative map)

``y^c`` is the pre-softmax score (logit) of class c, as in the original paper.
The default target layer is each model's last convolutional layer before global
average pooling. For Modified DenseNet201 that is **CONV2** (64 x 7 x 7 at 224 px).
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class GradCAMResult:
    cam: np.ndarray  # (H, W) float32 in [0, 1], upsampled to the model input size
    class_index: int  # class the map explains
    probabilities: np.ndarray  # softmax over classes for this input
    raw_cam: np.ndarray  # (h, w) map at the target layer's resolution, before upsampling


class GradCAM:
    """Compute Grad-CAM maps using hooks on ``target_layer``.

    Use as a context manager (or call ``close()``) so the hooks are removed.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self._activations: torch.Tensor | None = None
        self._gradients: torch.Tensor | None = None
        self._handle = target_layer.register_forward_hook(self._forward_hook)

    def _forward_hook(self, module: nn.Module, inputs, output: torch.Tensor) -> None:
        self._activations = output
        if output.requires_grad:  # False for forward passes run under torch.no_grad()
            output.register_hook(self._save_gradient)

    def _save_gradient(self, grad: torch.Tensor) -> None:
        self._gradients = grad

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "GradCAM":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __call__(self, x: torch.Tensor, class_index: int | None = None) -> GradCAMResult:
        """Explain a single image ``x`` of shape ``(1, 3, H, W)``.

        ``class_index`` defaults to the predicted class.
        """
        if x.ndim != 4 or x.shape[0] != 1:
            raise ValueError("GradCAM expects a single image tensor of shape (1, 3, H, W)")
        was_training = self.model.training
        self.model.eval()
        self._activations = self._gradients = None
        try:
            with torch.enable_grad():
                inp = x.detach().clone().requires_grad_(True)  # ensures the graph is recorded even if weights are frozen
                logits = self.model(inp)
                probs = torch.softmax(logits, dim=1)[0].detach().float().cpu().numpy()
                idx = int(np.argmax(probs)) if class_index is None else int(class_index)
                if not 0 <= idx < logits.shape[1]:
                    raise ValueError(f"class_index {idx} out of range for {logits.shape[1]} classes")
                self.model.zero_grad(set_to_none=True)
                logits[0, idx].backward()
            if self._activations is None or self._gradients is None:
                raise RuntimeError("Grad-CAM hooks did not fire; is the target layer part of the forward pass?")
            acts = self._activations.detach()[0]  # (K, h, w)
            grads = self._gradients.detach()[0]  # (K, h, w)
            weights = grads.mean(dim=(1, 2))  # alpha_k
            cam = F.relu((weights[:, None, None] * acts).sum(dim=0))
            raw = cam.float().cpu().numpy()
            up = F.interpolate(cam[None, None].float(), size=x.shape[-2:], mode="bilinear", align_corners=False)[0, 0]
            up = up.cpu().numpy()
            up = _normalize(up)
            return GradCAMResult(cam=up.astype(np.float32), class_index=idx, probabilities=probs, raw_cam=raw)
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)


def _normalize(cam: np.ndarray) -> np.ndarray:
    lo, hi = float(cam.min()), float(cam.max())
    if hi - lo < 1e-12:
        return np.zeros_like(cam)  # no positive evidence for this class anywhere
    return (cam - lo) / (hi - lo)


def heatmap_rgb(cam: np.ndarray, colormap: str = "jet") -> np.ndarray:
    """Map a [0, 1] CAM to a ``uint8`` RGB image using a matplotlib colormap."""
    cmap = matplotlib.colormaps[colormap]
    return (cmap(np.clip(cam, 0, 1))[..., :3] * 255).astype(np.uint8)


def overlay_heatmap(image: np.ndarray, cam: np.ndarray, alpha: float = 0.45, colormap: str = "jet") -> np.ndarray:
    """Blend a CAM over a ``uint8`` RGB image of the same height and width."""
    if image.shape[:2] != cam.shape:
        raise ValueError(f"image {image.shape[:2]} and cam {cam.shape} sizes differ")
    heat = heatmap_rgb(cam, colormap).astype(np.float32)
    out = (1 - alpha) * image.astype(np.float32) + alpha * heat
    return np.clip(out, 0, 255).astype(np.uint8)
