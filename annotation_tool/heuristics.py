from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from .paths import resource_path
from .schema import LANDMARK_DEFS, SIDES, Keypoint, empty_keypoint, fill_inferred_femoral_neck_axis_proximal, key_for, make_keypoint


SOURCE_NAME = "pose11_side"
DEFAULT_MODEL_NAME = "yolo11n-best.pt"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMGSZ = 800
DEFAULT_MIN_VISIBLE_KEYPOINTS = 6


@dataclass(frozen=True)
class AutoAnnotationResult:
    keypoints: Dict[str, Keypoint]
    warnings: list[str]
    model_available: bool
    source: str = "model-unavailable"
    attempts: list[dict[str, Any]] = field(default_factory=list)
    strategy: str = ""

    @property
    def retuve_available(self) -> bool:
        """Backward-compatible alias for older route code/tests."""
        return self.model_available

    @property
    def visible_count(self) -> int:
        return sum(1 for point in self.keypoints.values() if point.visible and point.x is not None and point.y is not None)


_model_lock = threading.Lock()
_model_cache: dict[str, Any] = {}


def estimate_keypoints_from_image(
    image: Image.Image,
    *,
    min_visible_keypoints: int | None = None,
    include_partial: bool = False,
) -> AutoAnnotationResult:
    image = image.convert("RGB")
    warnings: list[str] = []
    keypoints = _empty_template()
    attempts: list[dict[str, Any]] = []

    model_path = model_path_from_env()
    if not model_path.exists():
        warnings.append(f"Model unavailable: {model_path}")
        return AutoAnnotationResult(keypoints=keypoints, warnings=warnings, model_available=False, source="model-unavailable")

    try:
        model = _load_model(model_path)
    except ImportError as exc:
        warnings.append(f"Ultralytics unavailable: {exc}")
        return AutoAnnotationResult(keypoints=keypoints, warnings=warnings, model_available=False, source="model-unavailable")
    except Exception as exc:
        warnings.append(f"yolo11n-best load failed: {exc}")
        return AutoAnnotationResult(keypoints=keypoints, warnings=warnings, model_available=False, source="model-unavailable")

    min_visible = max(1, int(min_visible_keypoints if min_visible_keypoints is not None else _min_visible_keypoints()))
    best_decoded = keypoints
    best_visible = 0
    best_strategy = ""
    failed_attempts = 0

    for attempt in _prediction_attempts():
        attempt_image = _preprocess_for_attempt(image, str(attempt["preprocess"]))
        record = {
            "strategy": attempt["name"],
            "imgsz": attempt["imgsz"],
            "conf": attempt["conf"],
            "preprocess": attempt["preprocess"],
            "visible_count": 0,
            "success": False,
        }
        try:
            predictions = _run_model(model, attempt_image, imgsz=int(attempt["imgsz"]), conf=float(attempt["conf"]))
            decoded = decode_side11_result(predictions[0] if predictions else None)
        except Exception as exc:
            failed_attempts += 1
            record["error"] = str(exc)
            attempts.append(record)
            continue

        visible_count = _visible_count(decoded)
        record["visible_count"] = visible_count
        record["success"] = visible_count >= min_visible
        attempts.append(record)
        if visible_count > best_visible:
            best_decoded = decoded
            best_visible = visible_count
            best_strategy = str(attempt["name"])
        if visible_count >= min_visible:
            keypoints.update(decoded)
            return AutoAnnotationResult(
                keypoints=keypoints,
                warnings=warnings,
                model_available=True,
                source="yolo11n-best-side11",
                attempts=attempts,
                strategy=str(attempt["name"]),
            )

    if best_visible > 0:
        if include_partial:
            warnings.append(
                f"yolo11n-best returned only {best_visible} visible side11 keypoints; "
                "using partial model result for manual review."
            )
            keypoints.update(best_decoded)
            return AutoAnnotationResult(
                keypoints=keypoints,
                warnings=warnings,
                model_available=True,
                source="yolo11n-best-partial",
                attempts=attempts,
                strategy=best_strategy or "partial",
            )
        else:
            warnings.append(
                f"yolo11n-best returned only {best_visible} visible side11 keypoints; "
                f"need at least {min_visible}, keeping blank manual template."
            )
    else:
        warnings.append("yolo11n-best returned no visible side11 keypoints after fallback attempts.")
    if failed_attempts and failed_attempts == len(attempts):
        warnings.append("All yolo11n-best prediction attempts failed.")
        return AutoAnnotationResult(
            keypoints=keypoints,
            warnings=warnings,
            model_available=False,
            source="model-unavailable",
            attempts=attempts,
        )
    return AutoAnnotationResult(
        keypoints=keypoints,
        warnings=warnings,
        model_available=True,
        source="model-no-result",
        attempts=attempts,
        strategy="no-result",
    )


def model_path_from_env() -> Path:
    configured = os.environ.get("HIP22_MODEL_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return resource_path("models", DEFAULT_MODEL_NAME)


def _empty_template() -> Dict[str, Keypoint]:
    return {key_for(side, item.name): empty_keypoint(side, item) for side in SIDES for item in LANDMARK_DEFS}


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return default
    return float(value)


def _fallback_enabled() -> bool:
    value = os.environ.get("HIP22_AUTO_FALLBACK", os.environ.get("HIP22_FALLBACK", "1")).strip().lower()
    return value not in {"0", "false", "no", "off"}


def _min_visible_keypoints() -> int:
    return max(1, _env_int("HIP22_MIN_VISIBLE", DEFAULT_MIN_VISIBLE_KEYPOINTS))


def _prediction_attempts() -> list[dict[str, int | float | str]]:
    first = {
        "name": "default",
        "imgsz": _env_int("HIP22_IMGSZ", DEFAULT_IMGSZ),
        "conf": _env_float("HIP22_CONF", DEFAULT_CONFIDENCE),
        "preprocess": "none",
    }
    if not _fallback_enabled():
        return [first]
    return [
        first,
        {"name": "large_1024_low_conf", "imgsz": 1024, "conf": 0.15, "preprocess": "none"},
        {"name": "large_1280_low_conf", "imgsz": 1280, "conf": 0.15, "preprocess": "none"},
        {"name": "low_conf", "imgsz": DEFAULT_IMGSZ, "conf": 0.05, "preprocess": "none"},
        {"name": "contrast_1024_low_conf", "imgsz": 1024, "conf": 0.15, "preprocess": "autocontrast"},
    ]


def _preprocess_for_attempt(image: Image.Image, preprocess: str) -> Image.Image:
    if preprocess != "autocontrast":
        return image
    gray = ImageOps.grayscale(image)
    contrasted = ImageOps.autocontrast(gray)
    contrasted = ImageEnhance.Contrast(contrasted).enhance(1.35)
    return Image.merge("RGB", (contrasted, contrasted, contrasted))


def _visible_count(keypoints: Dict[str, Keypoint]) -> int:
    return sum(1 for point in keypoints.values() if point.visible and point.x is not None and point.y is not None)


def _preferred_device() -> str:
    override = os.environ.get("HIP22_DEVICE") or os.environ.get("HIP22_MODEL_DEVICE") or ""
    override = override.strip()
    if override and override != "auto":
        return override
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _load_model(model_path: Path) -> Any:
    cache_key = str(model_path)
    with _model_lock:
        if cache_key in _model_cache:
            return _model_cache[cache_key]
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        _model_cache[cache_key] = model
        return model


def _run_model(model: Any, image: Image.Image, *, imgsz: int, conf: float) -> list[Any]:
    return model.predict(
        source=np.asarray(image),
        imgsz=imgsz,
        conf=conf,
        device=_preferred_device(),
        verbose=False,
    )


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    return list(value)


def _image_size_from_result(result: Any) -> tuple[int, int]:
    if result is None:
        return 0, 0
    shape = getattr(result, "orig_shape", None)
    if shape is None and hasattr(result, "boxes"):
        shape = getattr(result.boxes, "orig_shape", None)
    if shape is None:
        return 0, 0
    height, width = shape[:2]
    return int(width), int(height)


def _box_centers_x(boxes: Any) -> list[float]:
    xywh = getattr(boxes, "xywh", None)
    if xywh is not None:
        return [float(item[0]) for item in _as_list(xywh)]
    xyxy = getattr(boxes, "xyxy", None)
    if xyxy is not None:
        return [(float(item[0]) + float(item[2])) / 2 for item in _as_list(xyxy)]
    return []


def _detections_from_result(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    boxes = getattr(result, "boxes", None)
    keypoints_obj = getattr(result, "keypoints", None)
    if boxes is None or keypoints_obj is None:
        return []

    classes = _as_list(getattr(boxes, "cls", []))
    confidences = _as_list(getattr(boxes, "conf", []))
    centers_x = _box_centers_x(boxes)
    keypoint_data = getattr(keypoints_obj, "data", None)
    if keypoint_data is None:
        keypoint_data = getattr(keypoints_obj, "xy", None)
    keypoints = _as_list(keypoint_data)

    detections: list[dict[str, Any]] = []
    for index, (class_value, confidence, points) in enumerate(zip(classes, confidences, keypoints)):
        if int(class_value) != 0:
            continue
        center_x = centers_x[index] if index < len(centers_x) else 0.0
        detections.append(
            {
                "confidence": float(confidence),
                "center_x": float(center_x),
                "points": points,
            }
        )
    return detections


def _assign_side_detections(detections: list[dict[str, Any]], *, width: int) -> dict[str, dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    if width > 0:
        for detection in detections:
            side = "left" if detection["center_x"] < width / 2 else "right"
            current = selected.get(side)
            if current is None or detection["confidence"] > current["confidence"]:
                selected[side] = detection
        return selected

    ordered = sorted(detections, key=lambda item: item["center_x"])
    if ordered:
        selected["left"] = ordered[0]
    if len(ordered) > 1:
        selected["right"] = ordered[-1]
    return selected


def decode_side11_result(result: Any) -> Dict[str, Keypoint]:
    width, _ = _image_size_from_result(result)
    detections = _detections_from_result(result)
    selected = _assign_side_detections(detections, width=width)
    decoded = _empty_template()

    for side in SIDES:
        detection = selected.get(side)
        if detection is None:
            continue
        # Current bundled model predicts the original 11 landmarks per side.
        # #12 is inferred below from the decoded #3 and #7 points.
        for index, landmark in enumerate(LANDMARK_DEFS[:11]):
            points = detection["points"]
            point = points[index] if len(points) > index else [0, 0, 0]
            x = float(point[0]) if len(point) > 0 else 0.0
            y = float(point[1]) if len(point) > 1 else 0.0
            point_confidence = float(point[2]) if len(point) > 2 else detection["confidence"]
            visible = (x > 0 or y > 0) and point_confidence > 0
            if not visible:
                continue
            decoded[key_for(side, landmark.name)] = make_keypoint(
                side,
                landmark.name,
                x,
                y,
                source=SOURCE_NAME,
                confidence=min(float(detection["confidence"]), point_confidence),
            )
    fill_inferred_femoral_neck_axis_proximal(decoded)
    return decoded
