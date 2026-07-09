from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from .heuristics import AutoAnnotationResult, estimate_keypoints_from_image
from .image_processing import enhance_xray_image
from .scan_like import map_result_from_scan, warp_for_scan_transform
from .schema import (
    LANDMARK_DEFS,
    REQUIRED_LANDMARK_DEFS,
    SIDES,
    Annotation,
    empty_keypoint,
    fill_inferred_femoral_neck_axis_proximal,
    key_for,
)


AUTO_DETECT_POLICY_LABEL = "original_then_enhanced_if_needed"


@dataclass(frozen=True)
class PreprocessedAutoDetection:
    result: AutoAnnotationResult
    image_preprocess: str
    roi_used: dict[str, Any] | None
    scan_used: dict[str, Any] | None
    preprocess_attempts: list[dict[str, Any]]
    usable: bool


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


def blank_keypoint_template() -> dict[str, Any]:
    return {key_for(side, item.name): empty_keypoint(side, item) for side in SIDES for item in LANDMARK_DEFS}


def required_side_visible_counts(keypoints: dict[str, Any]) -> dict[str, int]:
    counts = {side: 0 for side in SIDES}
    for side in SIDES:
        for landmark in REQUIRED_LANDMARK_DEFS:
            point = keypoints.get(key_for(side, landmark.name))
            if point is not None and point.visible and point.x is not None and point.y is not None:
                counts[side] += 1
    return counts


def has_required_keypoint_each_side(keypoints: dict[str, Any]) -> bool:
    counts = required_side_visible_counts(keypoints)
    return all(counts[side] > 0 for side in SIDES)


def has_any_required_keypoint_side(keypoints: dict[str, Any]) -> bool:
    counts = required_side_visible_counts(keypoints)
    return any(counts[side] > 0 for side in SIDES)


def has_manual_keypoints(annotation: Annotation) -> bool:
    return any(
        point.visible and point.x is not None and point.y is not None and point.source in {"manual", "imported_label"}
        for point in (annotation.keypoints or {}).values()
    )


def has_manual_completion(annotation: Annotation) -> bool:
    review = annotation.review if isinstance(annotation.review, dict) else {}
    keypoints = review.get("manual_keypoints_complete") if isinstance(review, dict) else None
    shenton = review.get("manual_shenton_complete") if isinstance(review, dict) else None
    return (
        (isinstance(keypoints, dict) and keypoints.get("status") == "confirmed")
        or (isinstance(shenton, dict) and shenton.get("status") == "confirmed")
    )


def should_retry_auto_detection(annotation: Annotation) -> bool:
    return (
        not has_manual_keypoints(annotation)
        and not has_manual_completion(annotation)
        and not has_required_keypoint_each_side(annotation.keypoints or {})
    )


def clear_non_manual_keypoints(annotation: Annotation) -> Annotation:
    annotator = annotation.annotator.user_id if annotation.annotator else ""
    for side in SIDES:
        for landmark in LANDMARK_DEFS:
            key = key_for(side, landmark.name)
            point = annotation.keypoints.get(key)
            if (
                point is not None
                and point.visible
                and point.x is not None
                and point.y is not None
                and point.source in {"manual", "imported_label"}
            ):
                continue
            annotation.keypoints[key] = empty_keypoint(side, landmark, annotator=annotator)
    return annotation


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
    fill_inferred_femoral_neck_axis_proximal(result.keypoints)
    return result, preprocess_label(use_enhanced=use_enhanced, use_scan=scan_used is not None, use_roi=roi_used is not None), roi_used, scan_used


def estimate_keypoints_original_then_enhanced(
    image: Image.Image,
    *,
    use_scan: bool = True,
    min_visible_keypoints: int | None = 1,
    include_partial: bool = True,
    roi_crop: dict[str, Any] | None = None,
    scan_transform: dict[str, Any] | None = None,
) -> PreprocessedAutoDetection:
    original = estimate_keypoints_with_preprocessing(
        image,
        use_enhanced=False,
        use_scan=use_scan,
        min_visible_keypoints=min_visible_keypoints,
        include_partial=include_partial,
        roi_crop=roi_crop,
        scan_transform=scan_transform,
    )
    original_record = _preprocess_attempt_record(*original)
    if original_record["usable"]:
        return PreprocessedAutoDetection(*original, preprocess_attempts=[original_record], usable=True)

    enhanced = estimate_keypoints_with_preprocessing(
        image,
        use_enhanced=True,
        use_scan=use_scan,
        min_visible_keypoints=min_visible_keypoints,
        include_partial=include_partial,
        roi_crop=roi_crop,
        scan_transform=scan_transform,
    )
    enhanced_record = _preprocess_attempt_record(*enhanced)
    preprocess_attempts = [original_record, enhanced_record]
    if enhanced_record["usable"] or _preprocess_attempt_has_any_side(enhanced_record):
        return PreprocessedAutoDetection(*enhanced, preprocess_attempts=preprocess_attempts, usable=True)
    if _preprocess_attempt_has_any_side(original_record):
        return PreprocessedAutoDetection(*original, preprocess_attempts=preprocess_attempts, usable=True)

    original_result, original_label, original_roi, original_scan = original
    enhanced_result, enhanced_label, enhanced_roi, enhanced_scan = enhanced
    warnings = [
        *original_result.warnings,
        *enhanced_result.warnings,
        "neither original nor enhanced detection produced usable required keypoints; no model keypoints applied.",
    ]
    result = AutoAnnotationResult(
        keypoints=blank_keypoint_template(),
        warnings=warnings,
        model_available=original_result.model_available or enhanced_result.model_available,
        source="model-no-result",
        attempts=[
            {"image_preprocess": original_label, "model_attempts": original_result.attempts},
            {"image_preprocess": enhanced_label, "model_attempts": enhanced_result.attempts},
        ],
        strategy="original_then_enhanced_no_result",
    )
    return PreprocessedAutoDetection(
        result=result,
        image_preprocess=f"{original_label}+{enhanced_label}",
        roi_used=enhanced_roi or original_roi,
        scan_used=enhanced_scan or original_scan,
        preprocess_attempts=preprocess_attempts,
        usable=False,
    )


def _preprocess_attempt_record(
    result: AutoAnnotationResult,
    image_preprocess: str,
    roi_used: dict[str, Any] | None,
    scan_used: dict[str, Any] | None,
) -> dict[str, Any]:
    counts = required_side_visible_counts(result.keypoints)
    return {
        "image_preprocess": image_preprocess,
        "source": result.source,
        "strategy": result.strategy,
        "visible_count": result.visible_count,
        "required_side_visible_counts": counts,
        "usable": all(counts[side] > 0 for side in SIDES),
        "roi_crop_used": roi_used is not None,
        "scan_transform_used": scan_used is not None,
    }


def _preprocess_attempt_has_any_side(record: dict[str, Any]) -> bool:
    counts = record.get("required_side_visible_counts")
    if not isinstance(counts, dict):
        return False
    return any(int(counts.get(side, 0) or 0) > 0 for side in SIDES)
