from __future__ import annotations

from typing import Any

from PIL import Image

from .heuristics import AutoAnnotationResult, estimate_keypoints_from_image
from .image_processing import enhance_xray_image
from .scan_like import map_result_from_scan, warp_for_scan_transform


def preprocess_label(*, use_enhanced: bool, use_scan: bool = False, use_roi: bool = False) -> str:
    parts: list[str] = []
    if use_roi:
        parts.append("roi_crop")
    elif use_scan:
        parts.append("scan_like")
    parts.append("hip_demo_enhanced" if use_enhanced else "original")
    return "+".join(parts)


def image_for_auto_detection(image: Image.Image, *, use_enhanced: bool) -> Image.Image:
    return enhance_xray_image(image) if use_enhanced else image


def normalized_roi_crop(roi_crop: dict[str, Any] | None, image: Image.Image) -> dict[str, Any] | None:
    if not isinstance(roi_crop, dict) or not roi_crop.get("enabled"):
        return None
    try:
        x = float(roi_crop.get("x"))
        y = float(roi_crop.get("y"))
        width = float(roi_crop.get("width"))
        height = float(roi_crop.get("height"))
    except (TypeError, ValueError):
        return None
    image_width, image_height = image.size
    x = max(0.0, min(float(image_width), x))
    y = max(0.0, min(float(image_height), y))
    width = max(0.0, min(float(image_width) - x, width))
    height = max(0.0, min(float(image_height) - y, height))
    if width < 8 or height < 8:
        return None
    return {
        "enabled": True,
        "x": x,
        "y": y,
        "width": width,
        "height": height,
    }


def crop_for_roi(image: Image.Image, roi_crop: dict[str, Any] | None) -> tuple[Image.Image, dict[str, Any] | None]:
    roi = normalized_roi_crop(roi_crop, image)
    if roi is None:
        return image, None
    left = int(round(roi["x"]))
    top = int(round(roi["y"]))
    right = int(round(roi["x"] + roi["width"]))
    bottom = int(round(roi["y"] + roi["height"]))
    right = max(left + 1, min(image.width, right))
    bottom = max(top + 1, min(image.height, bottom))
    roi.update({"x": float(left), "y": float(top), "width": float(right - left), "height": float(bottom - top)})
    return image.crop((left, top, right, bottom)), roi


def map_result_from_roi(result: AutoAnnotationResult, roi_used: dict[str, Any] | None) -> AutoAnnotationResult:
    if roi_used is None:
        return result
    offset_x = float(roi_used["x"])
    offset_y = float(roi_used["y"])
    for point in result.keypoints.values():
        if not point.visible or point.x is None or point.y is None:
            continue
        point.x = round(float(point.x) + offset_x, 2)
        point.y = round(float(point.y) + offset_y, 2)
    return result


def estimate_keypoints_with_preprocessing(
    image: Image.Image,
    *,
    use_enhanced: bool,
    use_scan: bool = True,
    min_visible_keypoints: int | None = None,
    include_partial: bool = False,
    roi_crop: dict[str, Any] | None = None,
    scan_transform: dict[str, Any] | None = None,
) -> tuple[AutoAnnotationResult, str, dict[str, Any] | None, dict[str, Any] | None]:
    kwargs: dict[str, Any] = {"include_partial": include_partial}
    if min_visible_keypoints is not None:
        kwargs["min_visible_keypoints"] = min_visible_keypoints
    detection_image, roi_used = crop_for_roi(image, roi_crop)
    scan_warp = None
    scan_used = None
    if roi_used is None:
        scan_warp = warp_for_scan_transform(image, scan_transform) if use_scan else None
    if scan_warp is not None:
        detection_image = scan_warp.image
        scan_used = scan_warp.used
    result = estimate_keypoints_from_image(image_for_auto_detection(detection_image, use_enhanced=use_enhanced), **kwargs)
    if scan_warp is not None:
        result = map_result_from_scan(result, scan_warp.inverse_matrix)
    else:
        result = map_result_from_roi(result, roi_used)
    return result, preprocess_label(use_enhanced=use_enhanced, use_scan=scan_used is not None, use_roi=roi_used is not None), roi_used, scan_used
