from __future__ import annotations

from PIL import Image, ImageOps
import numpy as np


def enhance_xray_image(image: Image.Image) -> Image.Image:
    """Return a display-oriented grayscale enhancement without changing geometry."""
    gray = image.convert("L")
    pixels = np.asarray(gray, dtype=np.float32)
    if pixels.size == 0:
        return gray.convert("RGB")

    low = float(np.percentile(pixels, 1.0))
    high = float(np.percentile(pixels, 99.4))
    if high > low:
        normalized = np.clip((pixels - low) / (high - low), 0.0, 1.0) * 255.0
    else:
        normalized = pixels

    enhanced = Image.fromarray(normalized.astype(np.uint8), mode="L")
    enhanced = ImageOps.autocontrast(enhanced, cutoff=0.5)
    enhanced = ImageOps.equalize(enhanced)
    return enhanced.convert("RGB")
