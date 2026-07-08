from __future__ import annotations

from typing import Any

from .schema import (
    Annotation,
    LANDMARK_DEFS,
    OPTIONAL_LANDMARK_DEFS,
    REQUIRED_LANDMARK_DEFS,
    SIDES,
    key_for,
)


KEYPOINT_TOTAL = len(SIDES) * len(LANDMARK_DEFS)
REQUIRED_KEYPOINT_TOTAL = len(SIDES) * len(REQUIRED_LANDMARK_DEFS)
OPTIONAL_KEYPOINT_TOTAL = len(SIDES) * len(OPTIONAL_LANDMARK_DEFS)
PROGRESS_STATUS_VERSION = 2
SHENTON_SEGMENTS = ("obturator_upper_curve", "femoral_neck_inner_lower_curve")
SHENTON_DONE_STATUSES = {"continuous", "discontinuous", "uncertain"}


def annotation_progress(annotation: Annotation) -> dict[str, Any]:
    keypoints = keypoint_progress(annotation)
    shenton = shenton_progress(annotation)
    status = _overall_status(keypoints, shenton)
    return {
        "status": status,
        "keypoint_status": keypoints["status"],
        "shenton_status": shenton["status"],
        "status_detail": (
            f"关键点 {keypoints['visible']}/{keypoints['total']}；"
            f"可选 {keypoints['optional_visible']}/{keypoints['optional_total']}；"
            f"Shenton {shenton['complete_sides']}/{shenton['total_sides']}"
        ),
        "keypoints": keypoints,
        "shenton": shenton,
    }


def keypoint_progress(annotation: Annotation) -> dict[str, Any]:
    visible = 0
    manual = 0
    optional_visible = 0
    optional_manual = 0
    for side in SIDES:
        for landmark in LANDMARK_DEFS:
            point = annotation.keypoints.get(key_for(side, landmark.name))
            is_visible = bool(point and point.visible and point.x is not None and point.y is not None)
            if not is_visible:
                continue
            if landmark in OPTIONAL_LANDMARK_DEFS:
                optional_visible += 1
                if point.source == "manual":
                    optional_manual += 1
                continue
            visible += 1
            if point.source == "manual":
                manual += 1
    complete = _manual_keypoints_confirmed(annotation)
    if complete:
        status = "complete"
    elif visible == 0 and optional_visible == 0:
        status = "pending"
    elif visible >= REQUIRED_KEYPOINT_TOTAL and manual == 0:
        status = "auto"
    else:
        status = "in_progress"
    return {
        "status": status,
        "visible": visible,
        "manual": manual,
        "total": REQUIRED_KEYPOINT_TOTAL,
        "optional_visible": optional_visible,
        "optional_manual": optional_manual,
        "optional_total": OPTIONAL_KEYPOINT_TOTAL,
        "all_visible": visible + optional_visible,
        "all_total": KEYPOINT_TOTAL,
        "complete": complete,
    }


def _manual_keypoints_confirmed(annotation: Annotation) -> bool:
    review = annotation.review if isinstance(annotation.review, dict) else {}
    manual = review.get("manual_keypoints_complete") if isinstance(review, dict) else None
    return isinstance(manual, dict) and manual.get("status") == "confirmed"


def _manual_shenton_confirmed(annotation: Annotation) -> bool:
    review = annotation.review if isinstance(annotation.review, dict) else {}
    manual = review.get("manual_shenton_complete") if isinstance(review, dict) else None
    return isinstance(manual, dict) and manual.get("status") == "confirmed"


def shenton_progress(annotation: Annotation) -> dict[str, Any]:
    sides: dict[str, Any] = {}
    complete_sides = 0
    started_sides = 0
    for side in SIDES:
        side_progress = shenton_side_progress(annotation, side)
        sides[side] = side_progress
        if side_progress["started"]:
            started_sides += 1
        if side_progress["complete"]:
            complete_sides += 1
    total_sides = len(SIDES)
    curves_complete = complete_sides == total_sides
    confirmed = _manual_shenton_confirmed(annotation)
    if confirmed:
        status = "complete"
        complete = True
    elif curves_complete:
        status = "awaiting_confirmation"
        complete = False
    elif started_sides:
        status = "in_progress"
        complete = False
    else:
        status = "pending"
        complete = False
    return {
        "status": status,
        "complete": complete,
        "complete_sides": complete_sides,
        "started_sides": started_sides,
        "total_sides": total_sides,
        "confirmed": confirmed,
        "sides": sides,
    }


def shenton_side_progress(annotation: Annotation, side: str) -> dict[str, Any]:
    curves = (annotation.shenton_curves or {}).get(side, {})
    review = (annotation.shenton_review or {}).get(side, {})
    review_status = str(review.get("status") or "not_reviewed")
    segment_counts = {
        segment: len(_clean_points((curves.get(segment) or {}).get("points") or []))
        for segment in SHENTON_SEGMENTS
    }
    curves_complete = all(count >= 3 for count in segment_counts.values())
    reviewed = review_status in SHENTON_DONE_STATUSES
    started = any(count > 0 for count in segment_counts.values()) or review_status != "not_reviewed"
    complete = curves_complete and reviewed
    if complete:
        status = "complete"
    elif started:
        status = "in_progress"
    else:
        status = "pending"
    return {
        "status": status,
        "complete": complete,
        "started": started,
        "curves_complete": curves_complete,
        "reviewed": reviewed,
        "review_status": review_status,
        "segment_counts": segment_counts,
    }


def _overall_status(keypoints: dict[str, Any], shenton: dict[str, Any]) -> str:
    if keypoints["complete"] and shenton["complete"]:
        return "done"
    if keypoints["complete"]:
        return "keypoint_complete"
    if shenton["complete"]:
        return "shenton_complete"
    if keypoints["visible"] == 0 and not shenton["started_sides"]:
        return "pending"
    if keypoints["status"] == "auto" and not shenton["started_sides"]:
        return "auto"
    if shenton["status"] == "awaiting_confirmation":
        return "shenton_awaiting_confirmation"
    return "in_progress"


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
        points.append({"x": x, "y": y})
    return points
