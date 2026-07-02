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
    LandmarkDef(7, "femoral_neck_axis_center", "股骨颈轴中心", "股骨头与股骨干之间的股骨颈中心"),
    LandmarkDef(8, "femoral_head_medial", "股骨头最内侧缘", "股骨头轮廓最靠近身体中线的点"),
    LandmarkDef(9, "femoral_head_lateral", "股骨头最外侧缘", "股骨头轮廓最远离身体中线的点"),
    LandmarkDef(10, "obturator_upper", "闭孔上缘", "闭孔透亮区顶部弧线最高点"),
    LandmarkDef(11, "femoral_neck_inner_lower", "股骨颈内下缘", "股骨颈内下方与股骨干交界处"),
)

LANDMARK_NAMES = tuple(item.name for item in LANDMARK_DEFS)
LANDMARK_BY_NAME = {item.name: item for item in LANDMARK_DEFS}


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


class ImageInfo(ExtraModel):
    filename: str
    width: int
    height: int
    split: str = "train"
    pixel_spacing_mm: Optional[float] = None
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
    auto_initialization: Dict[str, Any] = Field(default_factory=dict)
    measurements_snapshot: Dict[str, Any] = Field(default_factory=dict)
    review: Dict[str, Any] = Field(default_factory=dict)


class ManifestImage(ExtraModel):
    id: str
    image_path: str
    annotation_path: str
    status: str = "pending"
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

    image_path = [
        side_key("left", 1),
        side_key("left", 4),
        side_key("left", 10),
        side_key("right", 10),
        side_key("right", 4),
        side_key("right", 1),
    ]
    for point_a, point_b in zip(image_path, image_path[1:]):
        add(point_a, point_b, "图像左 #1-#4-#10 到图像右 #10-#4-#1", "default_cross_image")
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
    annotation.image.split = normalize_split(getattr(annotation.image, "split", "train"))
    normalize_connections(annotation)
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
