from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


DICOM_EXTENSIONS = {".dcm", ".dicom"}


@dataclass(frozen=True)
class DicomImage:
    image: Image.Image
    metadata: dict[str, Any]


def has_dicom_magic(path: Path) -> bool:
    try:
        with open(path, "rb") as handle:
            handle.seek(128)
            return handle.read(4) == b"DICM"
    except OSError:
        return False


def is_dicom_path(path: Path) -> bool:
    return path.suffix.lower() in DICOM_EXTENSIONS or (not path.suffix and has_dicom_magic(path))


def read_dicom_image(path: Path) -> DicomImage:
    try:
        import pydicom
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise ValueError("DICOM support requires pydicom.") from exc

    warnings: list[str] = []
    try:
        dataset = pydicom.dcmread(str(path), force=True)
    except Exception as exc:
        raise ValueError(f"Unable to parse DICOM: {exc}") from exc

    if not hasattr(dataset, "PixelData"):
        raise ValueError("DICOM file does not contain PixelData.")

    try:
        pixels = dataset.pixel_array
    except Exception as exc:
        raise ValueError(f"Unable to decode DICOM pixels: {exc}") from exc

    pixels = np.asarray(pixels)
    if pixels.ndim >= 3 and pixels.shape[-1] not in (3, 4):
        warnings.append("多帧 DICOM 仅显示第一帧。")
        pixels = pixels[0]

    if pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
        rgb = pixels[..., :3].astype(np.float32)
        image = _normalize_rgb(rgb)
    else:
        image = _normalize_grayscale(pixels.astype(np.float32), dataset)

    row_spacing, col_spacing, spacing_source = _pixel_spacing(dataset)
    pixel_spacing_mm = None
    if row_spacing is not None and col_spacing is not None:
        pixel_spacing_mm = round(float((row_spacing + col_spacing) / 2.0), 6)

    metadata = {
        "source_format": "dicom",
        "pixel_spacing_mm": pixel_spacing_mm,
        "pixel_spacing_row_mm": row_spacing,
        "pixel_spacing_col_mm": col_spacing,
        "pixel_spacing_source": spacing_source,
        "dicom_warnings": warnings,
    }
    return DicomImage(image=image.convert("RGB"), metadata=metadata)


def dicom_png_response_bytes(path: Path, *, enhanced: bool = False) -> tuple[bytes, dict[str, Any]]:
    from .image_processing import enhance_xray_image

    result = read_dicom_image(path)
    image = enhance_xray_image(result.image) if enhanced else result.image
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue(), result.metadata


def _normalize_rgb(pixels: np.ndarray) -> Image.Image:
    channels = []
    for idx in range(3):
        channels.append(_normalize_uint8(pixels[..., idx]))
    return Image.merge("RGB", [Image.fromarray(channel, mode="L") for channel in channels])


def _normalize_grayscale(pixels: np.ndarray, dataset: Any) -> Image.Image:
    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    pixels = pixels * slope + intercept

    center = _first_float(getattr(dataset, "WindowCenter", None))
    width = _first_float(getattr(dataset, "WindowWidth", None))
    if center is not None and width is not None and width > 0:
        low = center - width / 2.0
        high = center + width / 2.0
        pixels = np.clip(pixels, low, high)

    if str(getattr(dataset, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
        pixels = float(np.max(pixels)) + float(np.min(pixels)) - pixels

    return Image.fromarray(_normalize_uint8(pixels), mode="L").convert("RGB")


def _normalize_uint8(pixels: np.ndarray) -> np.ndarray:
    if pixels.size == 0:
        return pixels.astype(np.uint8)
    low = float(np.percentile(pixels, 1.0))
    high = float(np.percentile(pixels, 99.5))
    if high <= low:
        low = float(np.min(pixels))
        high = float(np.max(pixels))
    if high <= low:
        return np.zeros(pixels.shape, dtype=np.uint8)
    normalized = np.clip((pixels - low) / (high - low), 0.0, 1.0) * 255.0
    return normalized.astype(np.uint8)


def _first_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if isinstance(value, (list, tuple)) or hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
            value = list(value)[0]
        return float(value)
    except Exception:
        return None


def _pixel_spacing(dataset: Any) -> tuple[float | None, float | None, str]:
    for attr in ("PixelSpacing", "ImagerPixelSpacing"):
        value = getattr(dataset, attr, None)
        if value is None:
            continue
        try:
            row, col = list(value)[:2]
            return round(float(row), 6), round(float(col), 6), attr
        except Exception:
            continue
    return None, None, ""
