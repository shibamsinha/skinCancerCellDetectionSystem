"""Quantitative sanity checks for Grad-CAM maps.

1. **Deletion test (faithfulness).** Blur the top-k% most important pixels
   according to Grad-CAM and measure the drop in the predicted class'
   probability. Compare with blurring an equally large, spatially coherent
   random region. Faithful maps cause larger drops than random regions.
2. **Model-randomisation test** (Adebayo et al., NeurIPS 2018). Re-initialise
   the trained layers (additional convolutions + head) and recompute the maps.
   If explanations depend on what the model learned, the Spearman correlation
   between trained and randomised maps should be low.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch import nn
from torchvision.transforms.functional import gaussian_blur

from skincancer.explainability.gradcam import GradCAM


@dataclass
class GradCAMValidation:
    n_images: int
    area_fractions: list[float]
    cam_drop_mean: dict[str, float]
    random_drop_mean: dict[str, float]
    cam_beats_random_fraction: dict[str, float]
    randomization_spearman_mean: float
    randomization_spearman_median: float

    def to_dict(self) -> dict:
        return asdict(self)


def _random_coherent_mask(shape: tuple[int, int], fraction: float, rng: np.random.Generator, grid: int = 7) -> np.ndarray:
    """Spatially coherent random mask covering ``fraction`` of the image (upsampled random grid)."""
    noise = torch.from_numpy(rng.random((1, 1, grid, grid)).astype(np.float32))
    up = F.interpolate(noise, size=shape, mode="bilinear", align_corners=False)[0, 0].numpy()
    return up >= np.quantile(up, 1 - fraction)


def _masked_prob(model: nn.Module, x: torch.Tensor, blurred: torch.Tensor, mask: np.ndarray, cls: int) -> float:
    m = torch.from_numpy(mask.astype(np.float32)).to(x.device)[None, None]
    xm = x * (1 - m) + blurred * m
    with torch.no_grad():
        return float(torch.softmax(model(xm), dim=1)[0, cls])


def _reinitialise(module: nn.Module) -> None:
    for m in module.modules():
        if hasattr(m, "reset_parameters") and m is not module:
            m.reset_parameters()


def validate_gradcam(
    model: nn.Module,
    target_layer_name: str,
    images: list[torch.Tensor],
    trained_modules: list[str],
    area_fractions: tuple[float, ...] = (0.1, 0.2, 0.4),
    random_repeats: int = 3,
    seed: int = 42,
) -> GradCAMValidation:
    """Run both checks on ``images``, a list of preprocessed ``(1, 3, H, W)`` tensors on the model's device."""
    rng = np.random.default_rng(seed)
    model.eval()
    target = model.get_submodule(target_layer_name)
    cam_drops = {f: [] for f in area_fractions}
    rnd_drops = {f: [] for f in area_fractions}
    wins = {f: [] for f in area_fractions}
    trained_cams = []

    with GradCAM(model, target) as cam_fn:
        for x in images:
            res = cam_fn(x)
            cls = res.class_index
            p0 = float(res.probabilities[cls])
            blurred = gaussian_blur(x, kernel_size=[51, 51], sigma=[12.0, 12.0])
            trained_cams.append(res.cam)
            for f in area_fractions:
                cam_mask = res.cam >= np.quantile(res.cam, 1 - f)
                d_cam = p0 - _masked_prob(model, x, blurred, cam_mask, cls)
                d_rnd = float(np.mean([
                    p0 - _masked_prob(model, x, blurred, _random_coherent_mask(res.cam.shape, f, rng), cls)
                    for _ in range(random_repeats)
                ]))
                cam_drops[f].append(d_cam)
                rnd_drops[f].append(d_rnd)
                wins[f].append(d_cam > d_rnd)

    randomized = copy.deepcopy(model)
    for name in trained_modules:
        _reinitialise(randomized.get_submodule(name))
    rho = []
    with GradCAM(randomized, randomized.get_submodule(target_layer_name)) as cam_fn:
        for x, cam in zip(images, trained_cams):
            rcam = cam_fn(x).cam
            if np.ptp(rcam) == 0 or np.ptp(cam) == 0:
                rho.append(0.0)
                continue
            rho.append(float(spearmanr(cam.ravel(), rcam.ravel()).statistic))

    key = lambda f: f"{int(round(f * 100))}%"  # noqa: E731
    return GradCAMValidation(
        n_images=len(images),
        area_fractions=list(area_fractions),
        cam_drop_mean={key(f): float(np.mean(v)) for f, v in cam_drops.items()},
        random_drop_mean={key(f): float(np.mean(v)) for f, v in rnd_drops.items()},
        cam_beats_random_fraction={key(f): float(np.mean(v)) for f, v in wins.items()},
        randomization_spearman_mean=float(np.mean(rho)),
        randomization_spearman_median=float(np.median(rho)),
    )
