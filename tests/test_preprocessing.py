import io

import numpy as np
import pytest
import torch
from PIL import Image

from skincancer.config import AugmentationConfig, ContrastStretchConfig, GaussianNoiseConfig
from skincancer.preprocessing import (
    IMAGENET_NORMALIZATION,
    ImageLoadError,
    ImageTransform,
    TrainAugmentation,
    contrast_stretch,
    load_image,
)
from skincancer.preprocessing.augmentation import add_gaussian_noise, horizontal_flip, rotate


# ------------------------------------------------------------ contrast stretching
def test_contrast_stretch_expands_low_contrast_image():
    rng = np.random.default_rng(0)
    img = rng.integers(100, 140, size=(64, 64, 3)).astype(np.uint8)
    out = contrast_stretch(img, 2, 98)
    assert out.dtype == np.uint8 and out.shape == img.shape
    assert out.min() == 0 and out.max() == 255
    assert out.std() > img.std() * 3


def test_contrast_stretch_is_monotonic():
    img = np.arange(256, dtype=np.uint8).reshape(16, 16)
    out = contrast_stretch(img, 2, 98).ravel()
    assert np.all(np.diff(out.astype(int)) >= 0)


def test_contrast_stretch_constant_image_unchanged():
    img = np.full((8, 8, 3), 77, dtype=np.uint8)
    assert np.array_equal(contrast_stretch(img), img)


def test_contrast_stretch_global_preserves_channel_order():
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = 150, 100, 50  # R > G > B
    img[0, 0] = [160, 110, 60]
    out = contrast_stretch(img, 0, 100, per_channel=False)
    assert out[5, 5, 0] > out[5, 5, 1] > out[5, 5, 2]


def test_contrast_stretch_rejects_bad_input():
    with pytest.raises(TypeError):
        contrast_stretch(np.zeros((4, 4, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        contrast_stretch(np.zeros((4, 4, 3), dtype=np.uint8), 50, 10)


# ------------------------------------------------------------------ augmentation
def test_horizontal_flip():
    img = np.zeros((4, 6, 3), dtype=np.float32)
    img[:, 0] = 1.0
    out = horizontal_flip(img)
    assert out[:, -1].min() == 1.0 and out[:, 0].max() == 0.0


def test_rotation_keeps_shape_and_range():
    img = np.random.default_rng(1).random((32, 32, 3), dtype=np.float32)
    assert rotate(img, 0) is img
    for angle in (15, -15, 45, -45):
        out = rotate(img, angle)
        assert out.shape == img.shape and out.min() >= 0 and out.max() <= 1


def test_gaussian_noise_variance():
    rng = np.random.default_rng(2)
    img = np.full((256, 256, 3), 0.5, dtype=np.float32)
    out = add_gaussian_noise(img, 0.01, rng)  # small variance -> negligible clipping
    assert abs(float((out - img).var()) - 0.01) < 0.001


def test_documented_noise_variance_is_default():
    assert AugmentationConfig().gaussian_noise.variance == 0.1


def test_train_augmentation_is_seeded_and_bounded():
    cfg = AugmentationConfig(gaussian_noise=GaussianNoiseConfig(p=1.0, variance=0.1))
    img = np.random.default_rng(3).random((32, 32, 3), dtype=np.float32)
    a = TrainAugmentation(cfg, seed=7)(img)
    b = TrainAugmentation(cfg, seed=7)(img)
    assert np.array_equal(a, b)
    assert a.dtype == np.float32 and a.min() >= 0 and a.max() <= 1
    assert not np.array_equal(a, img)


# ------------------------------------------------------------------- transforms
def _pil(size=(120, 90), color=(180, 120, 100)):
    return Image.new("RGB", size, color)


def test_eval_transform_shape_and_determinism():
    tf = ImageTransform(64, IMAGENET_NORMALIZATION, ContrastStretchConfig(), None)
    img = _pil()
    x1, x2 = tf(img), tf(img)
    assert x1.shape == (3, 64, 64) and x1.dtype == torch.float32
    assert torch.equal(x1, x2)  # no augmentation at evaluation/inference time


def test_train_transform_changes_output():
    rng = np.random.default_rng(4)
    img = Image.fromarray(rng.integers(0, 255, (90, 120, 3)).astype(np.uint8))
    aug = TrainAugmentation(AugmentationConfig(horizontal_flip_p=1.0), seed=1)
    tf_train = ImageTransform(64, IMAGENET_NORMALIZATION, ContrastStretchConfig(), aug)
    tf_eval = ImageTransform(64, IMAGENET_NORMALIZATION, ContrastStretchConfig(), None)
    assert not torch.equal(tf_train(img), tf_eval(img))


def test_preprocess_display_is_uint8_square():
    tf = ImageTransform(224, IMAGENET_NORMALIZATION, ContrastStretchConfig(), None)
    out = tf.preprocess_display(_pil((600, 450)))
    assert out.shape == (224, 224, 3) and out.dtype == np.uint8


# ----------------------------------------------------------------- image loading
def _bytes(img: Image.Image, fmt: str) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return buf.getvalue()


def test_load_png_with_alpha_becomes_rgb():
    img = Image.new("RGBA", (64, 64), (255, 0, 0, 0))
    out = load_image(_bytes(img, "PNG"))
    assert out.mode == "RGB" and out.getpixel((0, 0)) == (255, 255, 255)


def test_load_grayscale_jpeg_becomes_rgb():
    out = load_image(_bytes(Image.new("L", (64, 64), 128), "JPEG"))
    assert out.mode == "RGB"


@pytest.mark.parametrize(
    "payload",
    [b"", b"not an image at all", _bytes(Image.new("RGB", (64, 64)), "GIF"), _bytes(Image.new("RGB", (8, 8)), "PNG")],
    ids=["empty", "garbage", "gif", "too-small"],
)
def test_invalid_uploads_raise_user_readable_error(payload):
    with pytest.raises(ImageLoadError):
        load_image(payload)


def test_missing_file_raises(tmp_path):
    with pytest.raises(ImageLoadError):
        load_image(tmp_path / "nope.jpg")
