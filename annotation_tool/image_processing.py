from __future__ import annotations

from PIL import Image, ImageOps
import numpy as np


def normalize_image_to_display_rgb(image: Image.Image) -> Image.Image:
    """Return an RGB image suitable for browser/model display."""
    pixels = _array_or_none(image)
    if pixels is not None and pixels.dtype in (np.uint8, np.bool_):
        return image.convert("RGB")
    normalized = _normalize_grayscale_array(_grayscale_pixels(image), upper_percentile=99.5)
    return normalized.convert("RGB")


def enhance_xray_image(image: Image.Image) -> Image.Image:
    """Return a display-oriented grayscale enhancement without changing geometry."""
    pixels = _grayscale_pixels(image)
    if pixels.size == 0:
        return Image.new("RGB", image.size)

    enhanced = _normalize_grayscale_array(pixels, upper_percentile=99.4)
    enhanced = ImageOps.autocontrast(enhanced, cutoff=0.5)
    enhanced = ImageOps.equalize(enhanced)
    return enhanced.convert("RGB")


def _array_or_none(image: Image.Image) -> np.ndarray | None:
    try:
        return np.asarray(image)
    except Exception:
        return None


def _grayscale_pixels(image: Image.Image) -> np.ndarray:
    pixels = _array_or_none(image)
    if pixels is None:
        return np.asarray(image.convert("L"), dtype=np.float32)
    if pixels.dtype in (np.uint8, np.bool_):
        return np.asarray(image.convert("L"), dtype=np.float32)

    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.ndim == 2:
        return pixels
    if pixels.ndim == 3 and pixels.shape[-1] >= 3:
        return pixels[..., 0] * 0.299 + pixels[..., 1] * 0.587 + pixels[..., 2] * 0.114
    if pixels.ndim == 3 and pixels.shape[-1] >= 1:
        return pixels[..., 0]
    return np.asarray(image.convert("L"), dtype=np.float32)


def _normalize_grayscale_array(
    pixels: np.ndarray,
    *,
    lower_percentile: float = 1.0,
    upper_percentile: float,
) -> Image.Image:
    pixels = np.asarray(pixels, dtype=np.float32)
    if pixels.size == 0:
        return Image.fromarray(np.zeros(pixels.shape, dtype=np.uint8), mode="L")

    finite = pixels[np.isfinite(pixels)]
    if finite.size == 0:
        return Image.fromarray(np.zeros(pixels.shape, dtype=np.uint8), mode="L")

    low = float(np.percentile(finite, lower_percentile))
    high = float(np.percentile(finite, upper_percentile))
    if high <= low:
        low = float(np.min(finite))
        high = float(np.max(finite))
    if high > low:
        normalized = np.clip((pixels - low) / (high - low), 0.0, 1.0) * 255.0
    else:
        normalized = np.clip(pixels, 0.0, 255.0)

    normalized = np.nan_to_num(normalized, nan=0.0, posinf=255.0, neginf=0.0)
    return Image.fromarray(normalized.astype(np.uint8), mode="L")
