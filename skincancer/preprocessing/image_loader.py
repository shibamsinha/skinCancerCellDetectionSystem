"""Safe image loading for training files and user uploads."""

from __future__ import annotations

import io
from pathlib import Path
from typing import BinaryIO

from PIL import Image, ImageOps, UnidentifiedImageError

# Guard against decompression bombs (default PIL limit is ~89 MP).
Image.MAX_IMAGE_PIXELS = 50_000_000

SUPPORTED_FORMATS = frozenset({"JPEG", "PNG", "MPO"})  # MPO = multi-picture JPEG from some cameras
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")
MIN_SIDE = 32
MAX_UPLOAD_BYTES = 20 * 1024 * 1024


class ImageLoadError(ValueError):
    """Raised when an image cannot be read or fails validation."""


def load_image(source: str | Path | bytes | BinaryIO) -> Image.Image:
    """Load an image as RGB, applying EXIF orientation and validating it.

    Accepts a filesystem path, raw bytes, or a binary file-like object (for
    example a Streamlit ``UploadedFile``). Transparent PNGs are composited onto
    a white background. Raises ``ImageLoadError`` with a user-readable message
    on any problem.
    """
    try:
        if isinstance(source, (str, Path)):
            img = Image.open(source)
        else:
            data = source if isinstance(source, bytes) else source.read()
            if len(data) == 0:
                raise ImageLoadError("The file is empty.")
            if len(data) > MAX_UPLOAD_BYTES:
                raise ImageLoadError(f"The file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB.")
            img = Image.open(io.BytesIO(data))
        fmt = img.format
        if fmt not in SUPPORTED_FORMATS:
            raise ImageLoadError(f"Unsupported image format '{fmt}'. Please upload a JPEG or PNG image.")
        img.load()
    except ImageLoadError:
        raise
    except FileNotFoundError as exc:
        raise ImageLoadError(f"Image file not found: {source}") from exc
    except Image.DecompressionBombError as exc:
        raise ImageLoadError("The image is too large to process safely.") from exc
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ImageLoadError("The file could not be read as an image. It may be corrupted.") from exc

    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba)
    img = img.convert("RGB")
    if min(img.size) < MIN_SIDE:
        raise ImageLoadError(f"The image is too small ({img.size[0]}x{img.size[1]}); at least {MIN_SIDE}x{MIN_SIDE} pixels are required.")
    return img
