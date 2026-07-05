from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .heuristics import AutoAnnotationResult


@dataclass(frozen=True)
class ScanWarp:
    image: Image.Image
    used: dict[str, Any]
    inverse_matrix: np.ndarray


def scan_transform_corners(scan_transform: dict[str, Any] | None, image: Image.Image) -> np.ndarray | None:
    if not isinstance(scan_transform, dict) or not scan_transform.get("enabled"):
        return None
    raw_corners = scan_transform.get("corners")
    if not isinstance(raw_corners, list) or len(raw_corners) != 4:
        return None
    points: list[tuple[float, float]] = []
    for item in raw_corners:
        if not isinstance(item, dict):
            return None
        try:
            x = max(0.0, min(float(image.width), float(item.get("x"))))
            y = max(0.0, min(float(image.height), float(item.get("y"))))
        except (TypeError, ValueError):
            return None
        points.append((x, y))
    ordered = order_quad_points(np.asarray(points, dtype=np.float32))
    if polygon_area(ordered) < 64.0:
        return None
    return ordered


def order_quad_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    sums = points.sum(axis=1)
    diffs = np.diff(points, axis=1).reshape(4)
    ordered = np.zeros((4, 2), dtype=np.float32)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(diffs)]
    ordered[3] = points[np.argmax(diffs)]
    return ordered


def polygon_area(points: np.ndarray) -> float:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def warp_scan_like_image(image: Image.Image, corners: np.ndarray) -> ScanWarp:
    ordered = order_quad_points(corners)
    top_width = np.linalg.norm(ordered[1] - ordered[0])
    bottom_width = np.linalg.norm(ordered[2] - ordered[3])
    left_height = np.linalg.norm(ordered[3] - ordered[0])
    right_height = np.linalg.norm(ordered[2] - ordered[1])
    output_width = max(16, int(round(max(top_width, bottom_width))))
    output_height = max(16, int(round(max(left_height, right_height))))
    destination = np.asarray(
        [
            [0, 0],
            [output_width - 1, 0],
            [output_width - 1, output_height - 1],
            [0, output_height - 1],
        ],
        dtype=np.float32,
    )
    matrix = cv2.getPerspectiveTransform(ordered.astype(np.float32), destination)
    inverse_matrix = cv2.getPerspectiveTransform(destination, ordered.astype(np.float32))
    pixels = np.asarray(image.convert("RGB"))
    warped = cv2.warpPerspective(pixels, matrix, (output_width, output_height), borderMode=cv2.BORDER_REPLICATE)
    used = {
        "enabled": True,
        "mode": "manual_four_corners",
        "corners": [{"x": round(float(x), 3), "y": round(float(y), 3)} for x, y in ordered],
        "output_width": output_width,
        "output_height": output_height,
    }
    return ScanWarp(image=Image.fromarray(warped).convert("RGB"), used=used, inverse_matrix=inverse_matrix)


def warp_for_scan_transform(image: Image.Image, scan_transform: dict[str, Any] | None) -> ScanWarp | None:
    corners = scan_transform_corners(scan_transform, image)
    if corners is None:
        return None
    return warp_scan_like_image(image, corners)


def map_result_from_scan(result: AutoAnnotationResult, inverse_matrix: np.ndarray | None) -> AutoAnnotationResult:
    if inverse_matrix is None:
        return result
    visible_items = [
        (key, point)
        for key, point in result.keypoints.items()
        if point.visible and point.x is not None and point.y is not None
    ]
    if not visible_items:
        return result
    points = np.asarray([[[float(point.x), float(point.y)]] for _, point in visible_items], dtype=np.float32)
    mapped = cv2.perspectiveTransform(points, inverse_matrix).reshape(-1, 2)
    for (_, point), (x, y) in zip(visible_items, mapped):
        point.x = round(float(x), 2)
        point.y = round(float(y), 2)
    return result


def detect_scan_quad(image: Image.Image) -> np.ndarray | None:
    pixels = np.asarray(image.convert("RGB"))
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    kernel = np.ones((5, 5), dtype=np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    contours, _hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    image_area = float(image.width * image.height)
    candidates: list[tuple[float, np.ndarray]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < image_area * 0.08:
            continue
        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(approx) != 4:
            continue
        points = approx.reshape(4, 2).astype(np.float32)
        candidates.append((float(area), order_quad_points(points)))
    if not candidates:
        return None
    return max(candidates, key=lambda item: item[0])[1]
