from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - pydantic v1 compatibility
    ConfigDict = None


SIDES = ("left", "right")
VALID_SPLITS = ("train", "val")


@dataclass(frozen=True)
class LandmarkDef:
    number: int
    name: str
    label_zh: str
    description: str


LANDMARK_DEFS: tuple[LandmarkDef, ...] = (
    LandmarkDef(1, "acetabular_outer", "髋臼外上缘", "髋臼顶最外上方的骨性凸起"),
    LandmarkDef(2, "triradiate_center", "Y 形软骨中心", "髋臼中心小凹陷或髂骨最下缘"),
    LandmarkDef(3, "femoral_head_center", "股骨头中心", "股骨头几何中心"),
    LandmarkDef(4, "teardrop_lower", "泪滴下缘", "泪滴形阴影底端尖点"),
    LandmarkDef(5, "femoral_shaft_prox", "股骨干轴近端中心", "小转子下方股骨干横截面中心"),
    LandmarkDef(6, "femoral_shaft_dist", "股骨干轴远端中心", "股骨干远端可见段横截面中心"),
    LandmarkDef(7, "femoral_neck_axis_center", "股骨颈轴中心远端", "股骨颈轴靠近股骨干侧的中心点"),
    LandmarkDef(8, "femoral_head_medial", "股骨头最内侧缘", "股骨头轮廓最靠近身体中线的点"),
    LandmarkDef(9, "femoral_head_lateral", "股骨头最外侧缘", "股骨头轮廓最远离身体中线的点"),
    LandmarkDef(10, "obturator_upper", "闭孔上缘", "闭孔透亮区顶部弧线最高点"),
    LandmarkDef(11, "femoral_neck_inner_lower", "股骨颈内下缘", "股骨颈内下方与股骨干交界处"),
    LandmarkDef(12, "femoral_neck_axis_proximal", "股骨颈轴中心近端", "股骨颈轴靠近股骨头侧的中心点"),
)

LANDMARK_NAMES = tuple(item.name for item in LANDMARK_DEFS)
LANDMARK_BY_NAME = {item.name: item for item in LANDMARK_DEFS}
OPTIONAL_LANDMARK_NUMBERS = frozenset({10, 11})
REQUIRED_LANDMARK_DEFS = tuple(item for item in LANDMARK_DEFS if item.number not in OPTIONAL_LANDMARK_NUMBERS)
OPTIONAL_LANDMARK_DEFS = tuple(item for item in LANDMARK_DEFS if item.number in OPTIONAL_LANDMARK_NUMBERS)
FEMORAL_HEAD_CENTER_NAME = "femoral_head_center"
FEMORAL_NECK_AXIS_DISTAL_NAME = "femoral_neck_axis_center"
FEMORAL_NECK_AXIS_PROXIMAL_NAME = "femoral_neck_axis_proximal"


def is_optional_landmark_name(name: str) -> bool:
    landmark = LANDMARK_BY_NAME.get(name)
    return bool(landmark and landmark.number in OPTIONAL_LANDMARK_NUMBERS)


def fill_inferred_femoral_neck_axis_proximal(keypoints: dict[str, "Keypoint"]) -> int:
    filled = 0
    for side in SIDES:
        head = keypoints.get(key_for(side, FEMORAL_HEAD_CENTER_NAME))
        distal = keypoints.get(key_for(side, FEMORAL_NECK_AXIS_DISTAL_NAME))
        target_key = key_for(side, FEMORAL_NECK_AXIS_PROXIMAL_NAME)
        current = keypoints.get(target_key)
        if (
            current is not None
            and current.visible
            and current.x is not None
            and current.y is not None
            and current.source == "manual"
        ):
            continue
        if not (
            head is not None
            and distal is not None
            and head.visible
            and distal.visible
            and head.x is not None
            and head.y is not None
            and distal.x is not None
            and distal.y is not None
        ):
            continue
        keypoints[target_key] = make_keypoint(
            side,
            FEMORAL_NECK_AXIS_PROXIMAL_NAME,
            (float(head.x) + float(distal.x)) / 2.0,
            (float(head.y) + float(distal.y)) / 2.0,
            source="estimated",
            confidence=min(float(head.confidence or 1.0), float(distal.confidence or 1.0)),
            annotator=current.annotator if current is not None else "",
        )
        filled += 1
    return filled


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def key_for(side: str, landmark_name: str) -> str:
    return f"{side}_{landmark_name}"


def side_label(side: str) -> str:
    return "左" if side == "left" else "右"


def point_label(side: str, landmark: LandmarkDef) -> str:
    return f"{side_label(side)} #{landmark.number} {landmark.label_zh}"


def all_keypoint_ids() -> list[str]:
    return [key_for(side, item.name) for side in SIDES for item in LANDMARK_DEFS]


def landmark_name_by_number(number: int) -> str:
    return LANDMARK_DEFS[number - 1].name


class ExtraModel(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(extra="allow")
    else:  # pragma: no cover - pydantic v1 compatibility
        class Config:
            extra = "allow"


class Keypoint(ExtraModel):
    id: str
    number: int
    name: str
    side: str
    label: str
    x: Optional[float] = None
    y: Optional[float] = None
    visible: bool = False
    visibility: int = 0
    source: str = "missing"
    confidence: float = 0.0
    updated_at: str = Field(default_factory=utc_now)
    annotator: str = ""


class Connection(ExtraModel):
    id: str
    point_a: str
    point_b: str
    label: str = ""
    source: str = "default"
    visible: bool = True
    updated_at: str = Field(default_factory=utc_now)
    annotator: str = ""


def default_shenton_segment(*, annotator: str = "") -> dict[str, Any]:
    return {
        "type": "polyline",
        "points": [],
        "source": "manual",
        "updated_at": utc_now(),
        "annotator": annotator,
    }


def default_shenton_curves(*, annotator: str = "") -> dict[str, Any]:
    return {
        side: {
            "obturator_upper_curve": default_shenton_segment(annotator=annotator),
            "femoral_neck_inner_lower_curve": default_shenton_segment(annotator=annotator),
        }
        for side in SIDES
    }


def default_shenton_review(*, annotator: str = "") -> dict[str, Any]:
    return {
        side: {
            "status": "not_reviewed",
            "updated_at": utc_now(),
            "annotator": annotator,
        }
        for side in SIDES
    }


def default_shenton_adjustments(*, annotator: str = "") -> dict[str, Any]:
    return {
        side: {
            "extension_intersection": {
                "enabled": False,
                "x": None,
                "y": None,
                "source": "manual",
                "updated_at": utc_now(),
                "annotator": annotator,
            }
        }
        for side in SIDES
    }


def default_roi_crop(*, annotator: str = "") -> dict[str, Any]:
    return {
        "enabled": False,
        "x": None,
        "y": None,
        "width": None,
        "height": None,
        "source": "manual",
        "updated_at": utc_now(),
        "annotator": annotator,
    }


def default_scan_transform(*, annotator: str = "") -> dict[str, Any]:
    return {
        "enabled": False,
        "corners": [],
        "mode": "manual_four_corners",
        "source": "manual",
        "updated_at": utc_now(),
        "annotator": annotator,
    }


def default_display_settings() -> dict[str, Any]:
    return {
        "brightness": 100,
        "contrast": 100,
    }


class ImageInfo(ExtraModel):
    filename: str
    width: int
    height: int
    split: str = "train"
    source_format: str = "image"
    pixel_spacing_mm: Optional[float] = None
    pixel_spacing_row_mm: Optional[float] = None
    pixel_spacing_col_mm: Optional[float] = None
    pixel_spacing_source: str = ""
    dicom_warnings: List[str] = Field(default_factory=list)
    patient_position: str = "AP"
    side_convention: str = "image-left/right; left key is image-left"


class AnnotatorInfo(ExtraModel):
    user_id: str = "default"
    created_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    duration_seconds: int = 0


class Annotation(ExtraModel):
    schema_version: str = "1.0.0"
    tool: str = "hip-22-point-annotator"
    image: ImageInfo
    annotator: AnnotatorInfo = Field(default_factory=AnnotatorInfo)
    keypoints: Dict[str, Keypoint] = Field(default_factory=dict)
    connections: List[Connection] = Field(default_factory=list)
    shenton_curves: Dict[str, Any] = Field(default_factory=default_shenton_curves)
    shenton_review: Dict[str, Any] = Field(default_factory=default_shenton_review)
    shenton_adjustments: Dict[str, Any] = Field(default_factory=default_shenton_adjustments)
    roi_crop: Dict[str, Any] = Field(default_factory=default_roi_crop)
    scan_transform: Dict[str, Any] = Field(default_factory=default_scan_transform)
    display_settings: Dict[str, Any] = Field(default_factory=default_display_settings)
    auto_initialization: Dict[str, Any] = Field(default_factory=dict)
    measurements_snapshot: Dict[str, Any] = Field(default_factory=dict)
    review: Dict[str, Any] = Field(default_factory=dict)


class ManifestImage(ExtraModel):
    id: str
    image_path: str
    annotation_path: str
    status: str = "pending"
    keypoint_status: str = "pending"
    shenton_status: str = "pending"
    status_detail: str = ""
    keypoint_visible_count: int = 0
    keypoint_manual_count: int = 0
    shenton_complete_sides: int = 0
    shenton_started_sides: int = 0
    annotation_mtime_ns: int = 0
    progress_status_version: int = 0
    annotator: str = ""
    completed_at: Optional[str] = None
    review_status: str = "pending"
    checksum_sha256: Optional[str] = None


class Manifest(ExtraModel):
    schema_version: str = "1.0.0"
    task_name: str = "hip-22-point-annotation"
    created_at: str = Field(default_factory=utc_now)
    images: List[ManifestImage] = Field(default_factory=list)


def model_to_dict(model: BaseModel) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def empty_keypoint(side: str, landmark: LandmarkDef, *, annotator: str = "") -> Keypoint:
    return Keypoint(
        id=key_for(side, landmark.name),
        number=landmark.number,
        name=landmark.name,
        side=side,
        label=point_label(side, landmark),
        annotator=annotator,
    )


def connection_id(point_a: str, point_b: str, prefix: str = "default") -> str:
    return f"{prefix}_{point_a}__{point_b}"


def make_connection(
    point_a: str,
    point_b: str,
    *,
    label: str = "",
    source: str = "default",
    visible: bool = True,
    annotator: str = "",
    id_prefix: str = "default",
) -> Connection:
    return Connection(
        id=connection_id(point_a, point_b, id_prefix),
        point_a=point_a,
        point_b=point_b,
        label=label,
        source=source,
        visible=visible,
        annotator=annotator,
    )


def default_connections(*, annotator: str = "") -> list[Connection]:
    connections: list[Connection] = []
    seen: set[frozenset[str]] = set()

    def add(point_a: str, point_b: str, label: str, prefix: str) -> None:
        edge = frozenset((point_a, point_b))
        if edge in seen:
            return
        seen.add(edge)
        connections.append(
            make_connection(
                point_a,
                point_b,
                label=label,
                source="default",
                annotator=annotator,
                id_prefix=prefix,
            )
        )

    def side_key(side: str, number: int) -> str:
        return key_for(side, landmark_name_by_number(number))

    for side in SIDES:
        prefix = "左" if side == "left" else "右"
        side_id = f"default_{side}"
        for a, b in ((1, 2), (2, 3), (3, 1)):
            add(side_key(side, a), side_key(side, b), f"{prefix} #1-#2-#3", f"{side_id}_triangle")
        for a, b in ((9, 3), (3, 8)):
            add(side_key(side, a), side_key(side, b), f"{prefix} #9-#3-#8", f"{side_id}_head")
        for a, b in ((1, 3), (3, 7), (7, 5), (5, 6), (6, 11), (11, 1)):
            add(side_key(side, a), side_key(side, b), f"{prefix} #1-#3-#7-#5-#6-#11", f"{side_id}_loop")
        add(side_key(side, 8), side_key(side, 9), f"{prefix} #8-#9", f"{side_id}_head_width")
        add(side_key(side, 12), side_key(side, 7), f"{prefix} #12-#7", f"{side_id}_neck_axis")

    image_path = [
        side_key("left", 4),
        side_key("left", 10),
        side_key("right", 10),
        side_key("right", 4),
    ]
    for point_a, point_b in zip(image_path, image_path[1:]):
        add(point_a, point_b, "图像左 #4-#10 到图像右 #10-#4", "default_cross_image")
    return connections


def make_keypoint(
    side: str,
    landmark_name: str,
    x: float,
    y: float,
    *,
    source: str,
    confidence: float,
    annotator: str = "",
    visibility: int = 2,
) -> Keypoint:
    landmark = LANDMARK_BY_NAME[landmark_name]
    return Keypoint(
        id=key_for(side, landmark.name),
        number=landmark.number,
        name=landmark.name,
        side=side,
        label=point_label(side, landmark),
        x=round(float(x), 2),
        y=round(float(y), 2),
        visible=visibility > 0,
        visibility=visibility,
        source=source,
        confidence=round(float(confidence), 3),
        updated_at=utc_now(),
        annotator=annotator,
    )


def create_blank_annotation(filename: str, width: int, height: int, *, annotator: str = "default") -> Annotation:
    annotation = Annotation(
        image=ImageInfo(filename=filename, width=int(width), height=int(height)),
        annotator=AnnotatorInfo(user_id=annotator),
    )
    ensure_keypoint_template(annotation)
    annotation.connections = default_connections(annotator=annotator)
    return annotation


def normalize_split(split: str | None) -> str:
    return split if split in VALID_SPLITS else "train"


def normalize_connections(annotation: Annotation) -> Annotation:
    annotator = annotation.annotator.user_id if annotation.annotator else ""
    normalized: list[Connection] = []
    seen_ids: set[str] = set()
    for item in annotation.connections or []:
        payload = model_to_dict(item) if isinstance(item, BaseModel) else dict(item)
        point_a = payload.get("point_a") or payload.get("from") or payload.get("a")
        point_b = payload.get("point_b") or payload.get("to") or payload.get("b")
        if point_a not in annotation.keypoints or point_b not in annotation.keypoints or point_a == point_b:
            continue
        payload.update(
            {
                "id": payload.get("id") or connection_id(point_a, point_b, str(payload.get("source") or "manual")),
                "point_a": point_a,
                "point_b": point_b,
                "label": payload.get("label") or "",
                "source": payload.get("source") or "manual",
                "visible": bool(payload.get("visible", payload.get("enabled", True))),
                "updated_at": payload.get("updated_at") or utc_now(),
                "annotator": payload.get("annotator") or annotator,
            }
        )
        if payload["id"] in seen_ids:
            payload["id"] = f"{payload['id']}_{len(seen_ids)}"
        seen_ids.add(payload["id"])
        normalized.append(Connection(**payload))
    annotation.connections = normalized
    return annotation


def normalize_shenton(annotation: Annotation) -> Annotation:
    annotator = annotation.annotator.user_id if annotation.annotator else ""
    raw_curves = annotation.shenton_curves if isinstance(annotation.shenton_curves, dict) else {}
    normalized_curves = default_shenton_curves(annotator=annotator)
    for side in SIDES:
        raw_side = raw_curves.get(side, {}) if isinstance(raw_curves.get(side, {}), dict) else {}
        for segment_key in ("obturator_upper_curve", "femoral_neck_inner_lower_curve"):
            raw_segment = raw_side.get(segment_key, {}) if isinstance(raw_side.get(segment_key, {}), dict) else {}
            segment = default_shenton_segment(annotator=annotator)
            points = raw_segment.get("points", [])
            segment.update(
                {
                    "type": raw_segment.get("type") or "polyline",
                    "points": points if isinstance(points, list) else [],
                    "source": raw_segment.get("source") or "manual",
                    "updated_at": raw_segment.get("updated_at") or utc_now(),
                    "annotator": raw_segment.get("annotator") or annotator,
                }
            )
            normalized_curves[side][segment_key] = segment
    annotation.shenton_curves = normalized_curves

    raw_review = annotation.shenton_review if isinstance(annotation.shenton_review, dict) else {}
    normalized_review = default_shenton_review(annotator=annotator)
    allowed_status = {"continuous", "discontinuous", "uncertain", "not_reviewed"}
    for side in SIDES:
        raw_side = raw_review.get(side, {}) if isinstance(raw_review.get(side, {}), dict) else {}
        status = raw_side.get("status") or "not_reviewed"
        if status not in allowed_status:
            status = "not_reviewed"
        normalized_review[side] = {
            "status": status,
            "updated_at": raw_side.get("updated_at") or utc_now(),
            "annotator": raw_side.get("annotator") or annotator,
        }
    annotation.shenton_review = normalized_review

    raw_adjustments = annotation.shenton_adjustments if isinstance(annotation.shenton_adjustments, dict) else {}
    normalized_adjustments = default_shenton_adjustments(annotator=annotator)
    image_width = max(1.0, float(annotation.image.width or 1))
    image_height = max(1.0, float(annotation.image.height or 1))
    for side in SIDES:
        raw_side = raw_adjustments.get(side, {}) if isinstance(raw_adjustments.get(side, {}), dict) else {}
        raw_intersection = raw_side.get("extension_intersection", {})
        if not isinstance(raw_intersection, dict):
            raw_intersection = {}
        x = raw_intersection.get("x")
        y = raw_intersection.get("y")
        try:
            x_float = float(x)
            y_float = float(y)
            has_point = True
        except (TypeError, ValueError):
            x_float = None
            y_float = None
            has_point = False
        enabled = bool(raw_intersection.get("enabled")) and has_point
        if enabled:
            x_float = max(0.0, min(image_width, x_float))
            y_float = max(0.0, min(image_height, y_float))
        normalized_adjustments[side]["extension_intersection"] = {
            "enabled": enabled,
            "x": x_float if enabled else None,
            "y": y_float if enabled else None,
            "source": raw_intersection.get("source") or "manual",
            "updated_at": raw_intersection.get("updated_at") or utc_now(),
            "annotator": raw_intersection.get("annotator") or annotator,
        }
    annotation.shenton_adjustments = normalized_adjustments
    return annotation


def normalize_roi_crop(annotation: Annotation) -> Annotation:
    annotator = annotation.annotator.user_id if annotation.annotator else ""
    raw = annotation.roi_crop if isinstance(annotation.roi_crop, dict) else {}
    normalized = default_roi_crop(annotator=annotator)
    normalized.update(
        {
            "source": raw.get("source") or "manual",
            "updated_at": raw.get("updated_at") or utc_now(),
            "annotator": raw.get("annotator") or annotator,
        }
    )
    enabled = bool(raw.get("enabled", False))
    try:
        x = float(raw.get("x"))
        y = float(raw.get("y"))
        width = float(raw.get("width"))
        height = float(raw.get("height"))
    except (TypeError, ValueError):
        enabled = False
    else:
        image_width = max(1.0, float(annotation.image.width or 1))
        image_height = max(1.0, float(annotation.image.height or 1))
        x = max(0.0, min(image_width, x))
        y = max(0.0, min(image_height, y))
        width = max(0.0, min(image_width - x, width))
        height = max(0.0, min(image_height - y, height))
        enabled = enabled and width >= 8 and height >= 8
        normalized.update(
            {
                "x": round(x, 3),
                "y": round(y, 3),
                "width": round(width, 3),
                "height": round(height, 3),
            }
        )
    normalized["enabled"] = enabled
    if not enabled:
        normalized.update({"x": None, "y": None, "width": None, "height": None})
    annotation.roi_crop = normalized
    return annotation


def normalize_scan_transform(annotation: Annotation) -> Annotation:
    annotator = annotation.annotator.user_id if annotation.annotator else ""
    raw = annotation.scan_transform if isinstance(annotation.scan_transform, dict) else {}
    normalized = default_scan_transform(annotator=annotator)
    normalized.update(
        {
            "mode": raw.get("mode") or "manual_four_corners",
            "source": raw.get("source") or "manual",
            "updated_at": raw.get("updated_at") or utc_now(),
            "annotator": raw.get("annotator") or annotator,
        }
    )
    image_width = max(1.0, float(annotation.image.width or 1))
    image_height = max(1.0, float(annotation.image.height or 1))
    corners: list[dict[str, float]] = []
    raw_corners = raw.get("corners", [])
    if isinstance(raw_corners, dict):
        raw_corners = [raw_corners.get(key) for key in ("top_left", "top_right", "bottom_right", "bottom_left")]
    if isinstance(raw_corners, list):
        for item in raw_corners[:4]:
            if not isinstance(item, dict):
                continue
            try:
                x = float(item.get("x"))
                y = float(item.get("y"))
            except (TypeError, ValueError):
                continue
            x = max(0.0, min(image_width, x))
            y = max(0.0, min(image_height, y))
            corners.append({"x": round(x, 3), "y": round(y, 3)})
    enabled = bool(raw.get("enabled", False)) and len(corners) == 4 and _polygon_area(corners) >= 64.0
    normalized["enabled"] = enabled
    normalized["corners"] = corners if enabled or corners else []
    annotation.scan_transform = normalized
    return annotation


def _polygon_area(points: list[dict[str, float]]) -> float:
    if len(points) < 3:
        return 0.0
    area = 0.0
    for index, point in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        area += point["x"] * nxt["y"] - nxt["x"] * point["y"]
    return abs(area) / 2.0


def ensure_keypoint_template(annotation: Annotation) -> Annotation:
    annotator = annotation.annotator.user_id if annotation.annotator else ""
    normalized: Dict[str, Keypoint] = {}
    existing = annotation.keypoints or {}
    for side in SIDES:
        for landmark in LANDMARK_DEFS:
            key = key_for(side, landmark.name)
            current = existing.get(key)
            if current is None:
                normalized[key] = empty_keypoint(side, landmark, annotator=annotator)
                continue
            payload = model_to_dict(current) if isinstance(current, BaseModel) else dict(current)
            x = payload.get("x")
            y = payload.get("y")
            visible = bool(payload.get("visible", x is not None and y is not None))
            visibility = int(payload.get("visibility", 2 if visible else 0) or 0)
            if x is None or y is None:
                visible = False
                visibility = 0
            payload.update(
                {
                    "id": key,
                    "number": landmark.number,
                    "name": landmark.name,
                    "side": side,
                    "label": payload.get("label") or point_label(side, landmark),
                    "visible": visible,
                    "visibility": visibility,
                    "source": payload.get("source") or ("manual" if visible else "missing"),
                    "confidence": float(payload.get("confidence") or (1.0 if visible else 0.0)),
                    "updated_at": payload.get("updated_at") or utc_now(),
                    "annotator": payload.get("annotator") or annotator,
                }
            )
            normalized[key] = Keypoint(**payload)
    annotation.keypoints = normalized
    fill_inferred_femoral_neck_axis_proximal(annotation.keypoints)
    annotation.image.split = normalize_split(getattr(annotation.image, "split", "train"))
    normalize_connections(annotation)
    normalize_shenton(annotation)
    normalize_roi_crop(annotation)
    normalize_scan_transform(annotation)
    return annotation


def annotation_from_dict(payload: dict[str, Any]) -> Annotation:
    payload = dict(payload)
    had_connections = "connections" in payload
    raw_keypoints = dict(payload.get("keypoints") or {})
    normalized_keypoints: Dict[str, Any] = {}
    for side in SIDES:
        for landmark in LANDMARK_DEFS:
            key = key_for(side, landmark.name)
            current = dict(raw_keypoints.get(key) or {})
            current.setdefault("id", key)
            current.setdefault("number", landmark.number)
            current.setdefault("name", landmark.name)
            current.setdefault("side", side)
            current.setdefault("label", point_label(side, landmark))
            if current.get("x") is not None and current.get("y") is not None:
                current.setdefault("visible", True)
                current.setdefault("visibility", 2)
            else:
                current.setdefault("visible", False)
                current.setdefault("visibility", 0)
            normalized_keypoints[key] = current
    payload["keypoints"] = normalized_keypoints
    if not had_connections:
        annotator = dict(payload.get("annotator") or {}).get("user_id") or ""
        payload["connections"] = [model_to_dict(item) for item in default_connections(annotator=annotator)]
    annotation = Annotation(**payload)
    return ensure_keypoint_template(annotation)
