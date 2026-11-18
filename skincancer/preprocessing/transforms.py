"""The full image pipeline: resize, contrast stretch, augment, normalise.

The same ``ImageTransform`` (with ``augmentation=None``) is used for
validation, testing and inference in the web app, which guarantees
train/serve consistency.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

from skincancer.config import ContrastStretchConfig
from skincancer.preprocessing.augmentation import TrainAugmentation
from skincancer.preprocessing.contrast_stretching import contrast_stretch


@dataclass(frozen=True)
class Normalization:
    """Per-channel normalisation expected by a backbone's pretrained weights."""

    mean: tuple[float, float, float]
    std: tuple[float, float, float]


IMAGENET_NORMALIZATION = Normalization((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
INCEPTION_NORMALIZATION = Normalization((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # maps to [-1, 1] (Keras Xception)
UNIT_NORMALIZATION = Normalization((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))  # plain /255 scaling


class ImageTransform:
    """PIL image -> normalised ``(3, S, S)`` float tensor."""

    def __init__(
        self,
        input_size: int,
        normalization: Normalization,
        contrast: ContrastStretchConfig,
        augmentation: TrainAugmentation | None = None,
    ) -> None:
        self.input_size = input_size
        self.normalization = normalization
        self.contrast = contrast
        self.augmentation = augmentation
        self._mean = np.asarray(normalization.mean, dtype=np.float32)
        self._std = np.asarray(normalization.std, dtype=np.float32)

    def preprocess_display(self, image: Image.Image) -> np.ndarray:
        """Resize and contrast-stretch; returns the ``uint8`` image the model sees.

        Images are resized directly to a square (aspect ratio not preserved),
        as Keras ``load_img(target_size=...)`` does.
        """
        img = image.convert("RGB").resize((self.input_size, self.input_size), Image.BILINEAR)
        arr = np.asarray(img, dtype=np.uint8)
        if self.contrast.enabled:
            arr = contrast_stretch(
                arr,
                self.contrast.lower_percentile,
                self.contrast.upper_percentile,
                self.contrast.per_channel,
            )
        return arr

    def __call__(self, image: Image.Image) -> torch.Tensor:
        x = self.preprocess_display(image).astype(np.float32) / 255.0
        if self.augmentation is not None:
            x = self.augmentation(x)
        x = (x - self._mean) / self._std
        return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))
