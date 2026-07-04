from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np
from PIL import Image

from .paths import resource_path
from .schema import LANDMARK_DEFS, SIDES, Keypoint, empty_keypoint, key_for, make_keypoint


SOURCE_NAME = "pose11_side"
DEFAULT_MODEL_NAME = "yolo11n-best.pt"
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IMGSZ = 800


@dataclass(frozen=True)
class AutoAnnotationResult:
    keypoints: Dict[str, Keypoint]
    warnings: list[str]
    model_available: bool

    @property
    def retuve_available(self) -> bool:
        """Backward-compatible alias for older route code/tests."""
        return self.model_available


_model_lock = threading.Lock()
_model_cache: dict[str, Any] = {}


def estimate_keypoints_from_image(image: Image.Image) -> AutoAnnotationResult:
    image = image.convert("RGB")
    warnings: list[str] = []
    keypoints = _empty_template()

    model_path = model_path_from_env()
    if not model_path.exists():
        warnings.append(f"Model unavailable: {model_path}")
        return AutoAnnotationResult(keypoints=keypoints, warnings=warnings, model_available=False)

    try:
        model = _load_model(model_path)
        predictions = _run_model(model, image)
    except ImportError as exc:
        warnings.append(f"Ultralytics unavailable: {exc}")
        return AutoAnnotationResult(keypoints=keypoints, warnings=warnings, model_available=False)
    except Exception as exc:
        warnings.append(f"yolo11n-best prediction failed: {exc}")
        return AutoAnnotationResult(keypoints=keypoints, warnings=warnings, model_available=False)

    decoded = decode_side11_result(predictions[0] if predictions else None)
    if not any(point.visible for point in decoded.values()):
        warnings.append("yolo11n-best returned no visible side11 keypoints.")
    keypoints.update(decoded)
    return AutoAnnotationResult(keypoints=keypoints, warnings=warnings, model_available=True)


def model_path_from_env() -> Path:
    configured = os.environ.get("HIP22_MODEL_PATH", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return resource_path("models", DEFAULT_MODEL_NAME)


def _empty_template() -> Dict[str, Keypoint]:
    return {key_for(side, item.name): empty_keypoint(side, item) for side in SIDES for item in LANDMARK_DEFS}


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


def _run_model(model: Any, image: Image.Image) -> list[Any]:
    imgsz = int(os.environ.get("HIP22_IMGSZ", DEFAULT_IMGSZ))
    conf = float(os.environ.get("HIP22_CONF", DEFAULT_CONFIDENCE))
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
        for index, landmark in enumerate(LANDMARK_DEFS):
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
    return decoded
