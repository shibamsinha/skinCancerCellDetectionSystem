"""Training-time data augmentation: horizontal flip, rotation, Gaussian noise.

These are the three techniques documented in the report (§5, Fig. 4):

* **Horizontal flip** [DOC]
* **Rotation**, by +/-15 and +/-45 degrees [SRC: Rehman et al. 2022 §3.1]
* **Gaussian noise**, variance 0.1 [DOC]

The original work enlarged the dataset offline before splitting it, which lets
augmented copies of one image land in different splits. Here augmentation is
applied **online and only to training images**, after a lesion-grouped split.
Every time a training image is drawn it receives an independent random
combination of the documented operations. In expectation this is equivalent
to offline enlargement, without the leakage.

Images are float32 arrays in [0, 1] with shape ``(H, W, 3)``.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from skincancer.config import AugmentationConfig


def horizontal_flip(image: np.ndarray) -> np.ndarray:
    return image[:, ::-1, :].copy()


def rotate(image: np.ndarray, angle_degrees: float) -> np.ndarray:
    """Rotate about the centre, keeping the size; borders use nearest fill.

    ``nearest`` matches the default ``fill_mode`` of Keras'
    ``ImageDataGenerator``, the augmentation tool of the original stack.
    """
    if angle_degrees == 0:
        return image
    out = ndimage.rotate(image, angle_degrees, axes=(1, 0), reshape=False, order=1, mode="nearest")
    return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)


def add_gaussian_noise(image: np.ndarray, variance: float, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise with the given variance (intensity scale [0, 1])."""
    if variance <= 0:
        return image
    noise = rng.normal(0.0, np.sqrt(variance), size=image.shape).astype(np.float32)
    return np.clip(image + noise, 0.0, 1.0)


class TrainAugmentation:
    """Randomly apply the documented augmentation operations to one image."""

    def __init__(self, cfg: AugmentationConfig, seed: int | None = None) -> None:
        self.cfg = cfg
        self._seed = seed
        self._rng: np.random.Generator | None = None

    @property
    def rng(self) -> np.random.Generator:
        # Created lazily so each DataLoader worker draws from its own stream,
        # seeded from NumPy's global state (set per worker in worker_init_fn).
        if self._rng is None:
            seed = self._seed if self._seed is not None else int(np.random.randint(0, 2**31 - 1))
            self._rng = np.random.default_rng(seed)
        return self._rng

    def __call__(self, image: np.ndarray) -> np.ndarray:
        if image.dtype != np.float32 or image.ndim != 3:
            raise ValueError("TrainAugmentation expects a float32 (H, W, C) image in [0, 1]")
        rng = self.rng
        out = image
        angles = self.cfg.rotation_angles
        if angles:
            out = rotate(out, float(angles[rng.integers(len(angles))]))
        if rng.random() < self.cfg.horizontal_flip_p:
            out = horizontal_flip(out)
        noise = self.cfg.gaussian_noise
        if rng.random() < noise.p:
            out = add_gaussian_noise(out, noise.variance, rng)
        return out
