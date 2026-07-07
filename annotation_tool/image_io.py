from __future__ import annotations

from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

from .dicom_utils import is_dicom_path, read_dicom_image
from .image_processing import normalize_image_to_display_rgb


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


def is_supported_image_path(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS or is_dicom_path(path)


def read_supported_image(path: Path, *, normalize: bool = True) -> tuple[Image.Image, dict[str, Any]]:
    if is_dicom_path(path):
        result = read_dicom_image(path)
        return result.image, result.metadata
    try:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image)
        image.load()
    except Exception as exc:
        raise ValueError(f"Unable to read image: {path.name}") from exc
    if normalize:
        image = normalize_image_to_display_rgb(image)
    return image, {"source_format": path.suffix.lower().lstrip(".") or "image"}
