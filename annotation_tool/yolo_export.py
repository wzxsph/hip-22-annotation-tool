from __future__ import annotations

from typing import Iterable

from .schema import LANDMARK_DEFS, Annotation, Keypoint, create_blank_annotation, key_for, make_keypoint


MIN_BOX_SIZE = 0.03
PADDING_FRACTION = 0.02


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _format_float(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def _is_visible(kp: Keypoint | None) -> bool:
    return bool(kp and kp.visible and kp.visibility > 0 and kp.x is not None and kp.y is not None)


def _kpt_values(kp: Keypoint | None, width: int, height: int) -> tuple[float, float, int]:
    if not _is_visible(kp):
        return 0.0, 0.0, 0
    assert kp is not None and kp.x is not None and kp.y is not None
    visibility = int(kp.visibility or 2)
    return _clamp(float(kp.x) / width), _clamp(float(kp.y) / height), visibility


def _expand_min_box(cx: float, cy: float, w: float, h: float) -> tuple[float, float, float, float]:
    w = max(w, MIN_BOX_SIZE)
    h = max(h, MIN_BOX_SIZE)
    x1 = _clamp(cx - w / 2)
    y1 = _clamp(cy - h / 2)
    x2 = _clamp(cx + w / 2)
    y2 = _clamp(cy + h / 2)
    if x2 - x1 < MIN_BOX_SIZE:
        if x1 <= 0:
            x2 = min(1.0, MIN_BOX_SIZE)
        elif x2 >= 1:
            x1 = max(0.0, 1.0 - MIN_BOX_SIZE)
    if y2 - y1 < MIN_BOX_SIZE:
        if y1 <= 0:
            y2 = min(1.0, MIN_BOX_SIZE)
        elif y2 >= 1:
            y1 = max(0.0, 1.0 - MIN_BOX_SIZE)
    return x1, y1, x2, y2


def _bbox_for_pair(left: Keypoint | None, right: Keypoint | None, width: int, height: int) -> tuple[float, float, float, float]:
    points = [kp for kp in (left, right) if _is_visible(kp)]
    if not points:
        return 0.5, 0.5, MIN_BOX_SIZE, MIN_BOX_SIZE

    xs = [float(kp.x) for kp in points if kp.x is not None]
    ys = [float(kp.y) for kp in points if kp.y is not None]
    x1 = _clamp((min(xs) - width * PADDING_FRACTION) / width)
    x2 = _clamp((max(xs) + width * PADDING_FRACTION) / width)
    y1 = _clamp((min(ys) - height * PADDING_FRACTION) / height)
    y2 = _clamp((max(ys) + height * PADDING_FRACTION) / height)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    x1, y1, x2, y2 = _expand_min_box(cx, cy, x2 - x1, y2 - y1)
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def annotation_to_yolo_lines(annotation: Annotation) -> list[str]:
    width = max(1, int(annotation.image.width))
    height = max(1, int(annotation.image.height))
    lines: list[str] = []
    for class_id, landmark in enumerate(LANDMARK_DEFS):
        left = annotation.keypoints.get(key_for("left", landmark.name))
        right = annotation.keypoints.get(key_for("right", landmark.name))
        cx, cy, box_w, box_h = _bbox_for_pair(left, right, width, height)
        lx, ly, lv = _kpt_values(left, width, height)
        rx, ry, rv = _kpt_values(right, width, height)
        values: Iterable[str] = (
            str(class_id),
            _format_float(cx),
            _format_float(cy),
            _format_float(box_w),
            _format_float(box_h),
            _format_float(lx),
            _format_float(ly),
            str(lv),
            _format_float(rx),
            _format_float(ry),
            str(rv),
        )
        lines.append(" ".join(values))
    return lines


def annotation_to_yolo_text(annotation: Annotation) -> str:
    return "\n".join(annotation_to_yolo_lines(annotation)) + "\n"


def annotation_from_yolo_text(
    text: str,
    filename: str,
    width: int,
    height: int,
    *,
    annotator: str = "default",
    source: str = "imported_label",
) -> Annotation:
    annotation = create_blank_annotation(filename, width, height, annotator=annotator)
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 11:
            continue
        try:
            class_id = int(float(parts[0]))
        except ValueError:
            continue
        if class_id < 0 or class_id >= len(LANDMARK_DEFS):
            continue
        landmark = LANDMARK_DEFS[class_id]
        for side, offset in (("left", 5), ("right", 8)):
            try:
                x_norm = float(parts[offset])
                y_norm = float(parts[offset + 1])
                visibility = int(float(parts[offset + 2]))
            except ValueError:
                continue
            key = key_for(side, landmark.name)
            if visibility <= 0:
                point = annotation.keypoints[key]
                point.visible = False
                point.visibility = 0
                point.source = "missing"
                point.confidence = 0.0
                point.x = None
                point.y = None
                continue
            annotation.keypoints[key] = make_keypoint(
                side,
                landmark.name,
                _clamp(x_norm) * width,
                _clamp(y_norm) * height,
                source=source,
                confidence=1.0,
                annotator=annotator,
                visibility=visibility,
            )
    annotation.auto_initialization = {
        "source": source,
        "warnings": [],
        "created_at": annotation.annotator.created_at,
    }
    return annotation


def data_yaml_text(path: str = ".", *, train: str = ".", val: str = ".") -> str:
    lines = [
        f"path: {path}",
        f"train: {train}",
        f"val: {val}",
        "",
        "kpt_shape: [2, 3]",
        "flip_idx: [1, 0]",
        "",
        "names:",
    ]
    for class_id, landmark in enumerate(LANDMARK_DEFS):
        lines.append(f"  {class_id}: {landmark.name}")
    return "\n".join(lines) + "\n"
