"""Contrast stretching, the first preprocessing stage of the original pipeline.

The report states only that contrast stretching was applied "for enhanced
visual clarity" (report §5, Fig. 4); the algorithm is not documented. This
module implements the standard linear percentile stretch (Gonzalez & Woods;
the scikit-image "contrast stretching" recipe): intensities between the
``lower`` and ``upper`` percentiles are mapped linearly onto the full
[0, 255] range and values outside are clipped.

By default the percentiles are computed over all three channels jointly
(``per_channel=False``), which preserves the lesion's hue. Per-channel
stretching is available but changes colour balance.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-6


def contrast_stretch(
    image: np.ndarray,
    lower_percentile: float = 2.0,
    upper_percentile: float = 98.0,
    per_channel: bool = False,
) -> np.ndarray:
    """Linearly stretch an RGB ``uint8`` image between two intensity percentiles.

    Args:
        image: ``(H, W, 3)`` or ``(H, W)`` array of dtype ``uint8``.
        lower_percentile: intensity percentile mapped to 0.
        upper_percentile: intensity percentile mapped to 255.
        per_channel: compute percentiles independently for each channel.

    Returns:
        A new ``uint8`` array with the same shape. Images with (near) constant
        intensity are returned unchanged because they cannot be stretched.
    """
    if image.dtype != np.uint8:
        raise TypeError(f"contrast_stretch expects uint8 input, got {image.dtype}")
    if image.ndim not in (2, 3):
        raise ValueError(f"expected (H, W) or (H, W, C) image, got shape {image.shape}")
    if not 0 <= lower_percentile < upper_percentile <= 100:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")

    img = image.astype(np.float32)
    if per_channel and img.ndim == 3:
        lo = np.percentile(img, lower_percentile, axis=(0, 1), keepdims=True)
        hi = np.percentile(img, upper_percentile, axis=(0, 1), keepdims=True)
        span = hi - lo
        # Channels without dynamic range are left untouched.
        flat = span < _EPS
        span = np.where(flat, 1.0, span)
        out = (img - lo) / span * 255.0
        out = np.where(flat, img, out)
    else:
        lo, hi = np.percentile(img, [lower_percentile, upper_percentile])
        if hi - lo < _EPS:
            return image.copy()
        out = (img - lo) / (hi - lo) * 255.0
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)
