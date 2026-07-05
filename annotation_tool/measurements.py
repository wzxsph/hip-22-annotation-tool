from __future__ import annotations

import math
from typing import Any

from .schema import Annotation


SHENTON_GAP_THRESHOLD_PX = 8.0
SHENTON_TANGENT_THRESHOLD_DEG = 15.0
SHENTON_MAX_EXTENSION_FRACTION = 0.12
SHENTON_MIN_EXTENSION_THRESHOLD_PX = 32.0
SHENTON_MAX_EXTENSION_THRESHOLD_PX = 120.0


def compute_measurements(annotation: Annotation) -> dict[str, Any]:
    return {
        "shenton": {
            "left": compute_shenton_side(annotation, "left"),
            "right": compute_shenton_side(annotation, "right"),
            "disclaimer": "Shenton 线结果仅作研究辅助，最终连续性由医生复核确认。",
        },
        "clinical_parameters": {
            "left": compute_clinical_parameters_side(annotation, "left"),
            "right": compute_clinical_parameters_side(annotation, "right"),
            "disclaimer": "Measurements are research aids based on current keypoints and require doctor review.",
        },
        "acetabular_depth": {
            "left": compute_acetabular_depth_side(annotation, "left"),
            "right": compute_acetabular_depth_side(annotation, "right"),
            "disclaimer": "Acetabular depth uses current keypoints; teardrop_lower is used as the available teardrop-edge proxy.",
        },
    }


def compute_shenton_side(annotation: Annotation, side: str) -> dict[str, Any]:
    curves = (annotation.shenton_curves or {}).get(side, {})
    obturator = _clean_points(curves.get("obturator_upper_curve", {}).get("points", []))
    femoral = _clean_points(curves.get("femoral_neck_inner_lower_curve", {}).get("points", []))
    warnings: list[str] = []
    if len(obturator) < 3:
        warnings.append("闭孔上缘曲线至少需要 3 个点。")
    if len(femoral) < 3:
        warnings.append("股骨颈内下缘曲线至少需要 3 个点。")
    if warnings:
        return {
            "status": "unavailable",
            "available": False,
            "continuous_candidate": None,
            "gap_px": None,
            "gap_mm": None,
            "tangent_angle_deg": None,
            "warnings": warnings,
        }

    end_a, idx_a, end_b, idx_b = _closest_endpoints(obturator, femoral)
    tangent_a = _endpoint_tangent(obturator, idx_a, target=end_b)
    tangent_b = _endpoint_tangent(femoral, idx_b, target=end_a)
    endpoint_gap_px = _distance(end_a, end_b)
    tangent_angle = _undirected_angle(tangent_a, tangent_b)
    endpoint_gap_mm = _physical_distance(annotation, end_a, end_b)
    extension = _shenton_extension_geometry(annotation, end_a, tangent_a, end_b, tangent_b)
    max_extension_threshold = _max_shenton_extension_distance(annotation)
    continuous = (
        extension["gap_px"] <= SHENTON_GAP_THRESHOLD_PX
        and tangent_angle <= SHENTON_TANGENT_THRESHOLD_DEG
        and extension["max_extension_distance_px"] <= max_extension_threshold
    )
    result_warnings = ["临床连续性阈值待学校团队确认。"]
    if endpoint_gap_mm is None:
        result_warnings.append("缺少有效 DICOM PixelSpacing，当前仅输出像素距离。")
    return {
        "status": "computed",
        "available": True,
        "continuous_candidate": continuous,
        "gap_px": round(endpoint_gap_px, 2),
        "gap_mm": round(endpoint_gap_mm, 3) if endpoint_gap_mm is not None else None,
        "endpoint_gap_px": round(endpoint_gap_px, 2),
        "endpoint_gap_mm": round(endpoint_gap_mm, 3) if endpoint_gap_mm is not None else None,
        "extension_gap_px": round(extension["gap_px"], 2),
        "extension_gap_mm": round(extension["gap_mm"], 3) if extension["gap_mm"] is not None else None,
        "extension_points_px": extension["points_px"],
        "extension_distance_px": {
            "obturator_upper_curve": round(extension["distance_a_px"], 2),
            "femoral_neck_inner_lower_curve": round(extension["distance_b_px"], 2),
            "max": round(extension["max_extension_distance_px"], 2),
            "threshold": round(max_extension_threshold, 2),
        },
        "extension_projection": extension["projection"],
        "tangent_angle_deg": round(tangent_angle, 2),
        "thresholds": {
            "gap_px": SHENTON_GAP_THRESHOLD_PX,
            "extension_gap_px": SHENTON_GAP_THRESHOLD_PX,
            "tangent_angle_deg": SHENTON_TANGENT_THRESHOLD_DEG,
            "max_extension_distance_px": round(max_extension_threshold, 2),
            "clinical_threshold_status": "pending",
        },
        "warnings": result_warnings,
    }


def compute_clinical_parameters_side(annotation: Annotation, side: str) -> dict[str, Any]:
    warnings: list[str] = []
    ai_angle = _ai_angle(annotation, side, warnings)
    sharp_angle = _sharp_angle(annotation, side, warnings)
    ce_angle = _ce_angle(annotation, side, warnings)
    neck_shaft_angle = _neck_shaft_angle(annotation, side, warnings)
    available = any(value is not None for value in (ai_angle, sharp_angle, ce_angle, neck_shaft_angle))
    return {
        "status": "computed" if available else "unavailable",
        "available": available,
        "ai_tonnis_angle_deg": _round_optional(ai_angle),
        "sharp_angle_deg": _round_optional(sharp_angle),
        "ce_angle_deg": _round_optional(ce_angle),
        "neck_shaft_angle_deg": _round_optional(neck_shaft_angle),
        "warnings": warnings,
    }


def compute_acetabular_depth_side(annotation: Annotation, side: str) -> dict[str, Any]:
    warnings: list[str] = []
    acetabular_outer = _keypoint(annotation, side, "acetabular_outer")
    teardrop = _keypoint(annotation, side, "teardrop_lower")
    femoral_head = _keypoint(annotation, side, "femoral_head_center")
    if acetabular_outer is None or teardrop is None or femoral_head is None:
        return {
            "status": "unavailable",
            "available": False,
            "value_px": None,
            "value_mm": None,
            "warnings": ["需要同侧髋臼外上缘、泪滴下缘和股骨头中心点。"],
        }
    depth_px = _point_to_line_distance(femoral_head, teardrop, acetabular_outer)
    teardrop_mm = _measurement_point(annotation, teardrop)
    acetabular_mm = _measurement_point(annotation, acetabular_outer)
    femoral_mm = _measurement_point(annotation, femoral_head)
    depth_mm = None
    if teardrop_mm and acetabular_mm and femoral_mm:
        depth_mm = _point_to_line_distance(femoral_mm, teardrop_mm, acetabular_mm)
    else:
        warnings.append("缺少有效 DICOM PixelSpacing，当前仅输出像素距离。")
    warnings.append("当前 22 点 schema 无独立泪滴外缘点，暂以 teardrop_lower 作为泪滴外缘代理。")
    return {
        "status": "computed",
        "available": True,
        "value_px": round(depth_px, 2),
        "value_mm": round(depth_mm, 3) if depth_mm is not None else None,
        "reference_points": {
            "acetabular_outer": "A",
            "teardrop_lower_proxy": "B",
            "femoral_head_center": "C",
        },
        "warnings": warnings,
    }


def _clean_points(raw_points: Any) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    if not isinstance(raw_points, list):
        return points
    for item in raw_points:
        if not isinstance(item, dict):
            continue
        try:
            x = float(item.get("x"))
            y = float(item.get("y"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(x) and math.isfinite(y):
            points.append({"x": x, "y": y})
    return points


def _closest_endpoints(
    first: list[dict[str, float]],
    second: list[dict[str, float]],
) -> tuple[dict[str, float], int, dict[str, float], int]:
    candidates = [(0, 0), (0, len(second) - 1), (len(first) - 1, 0), (len(first) - 1, len(second) - 1)]
    idx_a, idx_b = min(candidates, key=lambda pair: _distance(first[pair[0]], second[pair[1]]))
    return first[idx_a], idx_a, second[idx_b], idx_b


def _endpoint_tangent(
    points: list[dict[str, float]],
    endpoint_index: int,
    *,
    target: dict[str, float] | None = None,
) -> tuple[float, float]:
    endpoint = points[endpoint_index]
    window_size = min(4, len(points))
    if endpoint_index == 0:
        local = points[:window_size]
        vec = (local[0]["x"] - local[-1]["x"], local[0]["y"] - local[-1]["y"])
    else:
        local = points[-window_size:]
        vec = (local[-1]["x"] - local[0]["x"], local[-1]["y"] - local[0]["y"])
    if math.hypot(*vec) == 0:
        neighbor = points[1] if endpoint_index == 0 else points[-2]
        vec = (endpoint["x"] - neighbor["x"], endpoint["y"] - neighbor["y"])
    if target is not None:
        toward_target = (target["x"] - endpoint["x"], target["y"] - endpoint["y"])
        if _dot(vec, toward_target) < 0:
            vec = (-vec[0], -vec[1])
    return vec


def _shenton_extension_geometry(
    annotation: Annotation,
    first: dict[str, float],
    first_tangent: tuple[float, float],
    second: dict[str, float],
    second_tangent: tuple[float, float],
) -> dict[str, Any]:
    first_dir = _unit(first_tangent)
    second_dir = _unit(second_tangent)
    if first_dir is None or second_dir is None:
        gap_px = _distance(first, second)
        return {
            "gap_px": gap_px,
            "gap_mm": _physical_distance(annotation, first, second),
            "distance_a_px": 0.0,
            "distance_b_px": 0.0,
            "max_extension_distance_px": 0.0,
            "projection": "endpoint_fallback",
            "points_px": {
                "obturator_upper_curve": _round_point(first),
                "femoral_neck_inner_lower_curve": _round_point(second),
                "intersection": None,
            },
        }

    closest = _closest_points_on_forward_rays(first, first_dir, second, second_dir)
    point_a = closest["point_a"]
    point_b = closest["point_b"]
    gap_px = _distance(point_a, point_b)
    return {
        "gap_px": gap_px,
        "gap_mm": _physical_distance(annotation, point_a, point_b),
        "distance_a_px": closest["distance_a_px"],
        "distance_b_px": closest["distance_b_px"],
        "max_extension_distance_px": max(closest["distance_a_px"], closest["distance_b_px"]),
        "projection": closest["projection"],
        "points_px": {
            "obturator_upper_curve": _round_point(point_a),
            "femoral_neck_inner_lower_curve": _round_point(point_b),
            "intersection": _round_point(point_a) if gap_px <= 1e-6 else None,
        },
    }


def _closest_points_on_forward_rays(
    first: dict[str, float],
    first_dir: tuple[float, float],
    second: dict[str, float],
    second_dir: tuple[float, float],
) -> dict[str, Any]:
    intersection = _ray_intersection(first, first_dir, second, second_dir)
    if intersection is not None:
        point, distance_a, distance_b = intersection
        return {
            "point_a": point,
            "point_b": point,
            "distance_a_px": distance_a,
            "distance_b_px": distance_b,
            "projection": "forward_intersection",
        }

    candidates: list[dict[str, Any]] = []
    projection, distance = _project_point_to_ray(first, second, second_dir)
    candidates.append(
        {
            "point_a": first,
            "point_b": projection,
            "distance_a_px": 0.0,
            "distance_b_px": distance,
            "projection": "obturator_endpoint_to_femoral_ray",
        }
    )
    projection, distance = _project_point_to_ray(second, first, first_dir)
    candidates.append(
        {
            "point_a": projection,
            "point_b": second,
            "distance_a_px": distance,
            "distance_b_px": 0.0,
            "projection": "femoral_endpoint_to_obturator_ray",
        }
    )
    candidates.append(
        {
            "point_a": first,
            "point_b": second,
            "distance_a_px": 0.0,
            "distance_b_px": 0.0,
            "projection": "endpoint_gap",
        }
    )
    return min(candidates, key=lambda item: _distance(item["point_a"], item["point_b"]))


def _ray_intersection(
    first: dict[str, float],
    first_dir: tuple[float, float],
    second: dict[str, float],
    second_dir: tuple[float, float],
) -> tuple[dict[str, float], float, float] | None:
    denom = _cross(first_dir, second_dir)
    if abs(denom) < 1e-9:
        return None
    delta = (second["x"] - first["x"], second["y"] - first["y"])
    distance_a = _cross(delta, second_dir) / denom
    distance_b = _cross(delta, first_dir) / denom
    if distance_a < -1e-6 or distance_b < -1e-6:
        return None
    point = {"x": first["x"] + first_dir[0] * distance_a, "y": first["y"] + first_dir[1] * distance_a}
    return point, max(0.0, distance_a), max(0.0, distance_b)


def _project_point_to_ray(
    point: dict[str, float],
    origin: dict[str, float],
    direction: tuple[float, float],
) -> tuple[dict[str, float], float]:
    delta = (point["x"] - origin["x"], point["y"] - origin["y"])
    distance = max(0.0, _dot(delta, direction))
    projected = {"x": origin["x"] + direction[0] * distance, "y": origin["y"] + direction[1] * distance}
    return projected, distance


def _undirected_angle(first: tuple[float, float], second: tuple[float, float]) -> float:
    angle = _angle_between(first, second)
    return min(angle, abs(180.0 - angle))


def _angle_between(first: tuple[float, float], second: tuple[float, float]) -> float:
    ax, ay = first
    bx, by = second
    norm = math.hypot(ax, ay) * math.hypot(bx, by)
    if norm == 0:
        return 180.0
    cosine = max(-1.0, min(1.0, (ax * bx + ay * by) / norm))
    return math.degrees(math.acos(cosine))


def _distance(first: dict[str, float], second: dict[str, float]) -> float:
    return math.hypot(second["x"] - first["x"], second["y"] - first["y"])


def _dot(first: tuple[float, float], second: tuple[float, float]) -> float:
    return first[0] * second[0] + first[1] * second[1]


def _cross(first: tuple[float, float], second: tuple[float, float]) -> float:
    return first[0] * second[1] - first[1] * second[0]


def _unit(vector: tuple[float, float]) -> tuple[float, float] | None:
    norm = math.hypot(vector[0], vector[1])
    if norm <= 1e-9:
        return None
    return vector[0] / norm, vector[1] / norm


def _round_point(point: dict[str, float]) -> dict[str, float]:
    return {"x": round(float(point["x"]), 2), "y": round(float(point["y"]), 2)}


def _max_shenton_extension_distance(annotation: Annotation) -> float:
    width = max(1.0, float(getattr(annotation.image, "width", 1) or 1))
    height = max(1.0, float(getattr(annotation.image, "height", 1) or 1))
    dynamic = math.hypot(width, height) * SHENTON_MAX_EXTENSION_FRACTION
    return max(SHENTON_MIN_EXTENSION_THRESHOLD_PX, min(SHENTON_MAX_EXTENSION_THRESHOLD_PX, dynamic))


def _physical_distance(annotation: Annotation, first: dict[str, float], second: dict[str, float]) -> float | None:
    row_spacing = getattr(annotation.image, "pixel_spacing_row_mm", None) or getattr(annotation.image, "pixel_spacing_mm", None)
    col_spacing = getattr(annotation.image, "pixel_spacing_col_mm", None) or getattr(annotation.image, "pixel_spacing_mm", None)
    if row_spacing is None or col_spacing is None:
        return None
    try:
        dx = (second["x"] - first["x"]) * float(col_spacing)
        dy = (second["y"] - first["y"]) * float(row_spacing)
    except (TypeError, ValueError):
        return None
    return math.hypot(dx, dy)


def _keypoint(annotation: Annotation, side: str, name: str) -> dict[str, float] | None:
    point = annotation.keypoints.get(f"{side}_{name}")
    if point is None or not point.visible or point.x is None or point.y is None:
        return None
    return {"x": float(point.x), "y": float(point.y)}


def _measurement_point(annotation: Annotation, point: dict[str, float]) -> dict[str, float] | None:
    row_spacing = getattr(annotation.image, "pixel_spacing_row_mm", None) or getattr(annotation.image, "pixel_spacing_mm", None)
    col_spacing = getattr(annotation.image, "pixel_spacing_col_mm", None) or getattr(annotation.image, "pixel_spacing_mm", None)
    if row_spacing is None or col_spacing is None:
        return None
    try:
        return {"x": point["x"] * float(col_spacing), "y": point["y"] * float(row_spacing)}
    except (TypeError, ValueError):
        return None


def _geometry_point(annotation: Annotation, point: dict[str, float]) -> dict[str, float]:
    return _measurement_point(annotation, point) or point


def _pelvic_horizontal(annotation: Annotation) -> tuple[dict[str, float], dict[str, float]] | None:
    left = _keypoint(annotation, "left", "triradiate_center")
    right = _keypoint(annotation, "right", "triradiate_center")
    if left is None or right is None:
        return None
    return _geometry_point(annotation, left), _geometry_point(annotation, right)


def _teardrop_baseline(annotation: Annotation) -> tuple[dict[str, float], dict[str, float]] | None:
    left = _keypoint(annotation, "left", "teardrop_lower")
    right = _keypoint(annotation, "right", "teardrop_lower")
    if left is None or right is None:
        return None
    return _geometry_point(annotation, left), _geometry_point(annotation, right)


def _ai_angle(annotation: Annotation, side: str, warnings: list[str]) -> float | None:
    horizontal = _pelvic_horizontal(annotation)
    triradiate = _keypoint(annotation, side, "triradiate_center")
    acetabular = _keypoint(annotation, side, "acetabular_outer")
    if horizontal is None or triradiate is None or acetabular is None:
        warnings.append("AI/Tonnis 角需要双侧 Y 形软骨中心和同侧髋臼外上缘。")
        return None
    h_vec = _vector(horizontal[0], horizontal[1])
    roof_vec = _vector(_geometry_point(annotation, triradiate), _geometry_point(annotation, acetabular))
    return _undirected_angle(h_vec, roof_vec)


def _sharp_angle(annotation: Annotation, side: str, warnings: list[str]) -> float | None:
    baseline = _teardrop_baseline(annotation)
    teardrop = _keypoint(annotation, side, "teardrop_lower")
    acetabular = _keypoint(annotation, side, "acetabular_outer")
    if baseline is None or teardrop is None or acetabular is None:
        warnings.append("Sharp 角需要双侧泪滴下缘和同侧髋臼外上缘。")
        return None
    base_vec = _vector(baseline[0], baseline[1])
    sharp_vec = _vector(_geometry_point(annotation, teardrop), _geometry_point(annotation, acetabular))
    return _undirected_angle(base_vec, sharp_vec)


def _ce_angle(annotation: Annotation, side: str, warnings: list[str]) -> float | None:
    horizontal = _pelvic_horizontal(annotation)
    femoral_head = _keypoint(annotation, side, "femoral_head_center")
    acetabular = _keypoint(annotation, side, "acetabular_outer")
    if horizontal is None or femoral_head is None or acetabular is None:
        warnings.append("CE 角需要双侧 Y 形软骨中心、同侧股骨头中心和髋臼外上缘。")
        return None
    h_vec = _vector(horizontal[0], horizontal[1])
    vertical_vec = (-h_vec[1], h_vec[0])
    cover_vec = _vector(_geometry_point(annotation, femoral_head), _geometry_point(annotation, acetabular))
    return _undirected_angle(vertical_vec, cover_vec)


def _neck_shaft_angle(annotation: Annotation, side: str, warnings: list[str]) -> float | None:
    head = _keypoint(annotation, side, "femoral_head_center")
    neck = _keypoint(annotation, side, "femoral_neck_axis_center")
    shaft_prox = _keypoint(annotation, side, "femoral_shaft_prox")
    shaft_dist = _keypoint(annotation, side, "femoral_shaft_dist")
    if head is None or neck is None or shaft_prox is None or shaft_dist is None:
        warnings.append("颈干角需要股骨头中心、股骨颈轴中心和股骨干近/远端中心。")
        return None
    neck_vec = _vector(_geometry_point(annotation, neck), _geometry_point(annotation, head))
    shaft_vec = _vector(_geometry_point(annotation, shaft_prox), _geometry_point(annotation, shaft_dist))
    angle = _angle_between(neck_vec, shaft_vec)
    return max(angle, 180.0 - angle)


def _vector(first: dict[str, float], second: dict[str, float]) -> tuple[float, float]:
    return second["x"] - first["x"], second["y"] - first["y"]


def _point_to_line_distance(point: dict[str, float], line_a: dict[str, float], line_b: dict[str, float]) -> float:
    ax, ay = line_a["x"], line_a["y"]
    bx, by = line_b["x"], line_b["y"]
    px, py = point["x"], point["y"]
    dx = bx - ax
    dy = by - ay
    denom = math.hypot(dx, dy)
    if denom == 0:
        return math.hypot(px - ax, py - ay)
    return abs(dy * px - dx * py + bx * ay - by * ax) / denom


def _round_optional(value: float | None) -> float | None:
    return round(value, 2) if value is not None else None
