from __future__ import annotations

from .schema import Keypoint, make_keypoint


TEMPLATE_VERSION = "demo-median-2026-07-05"
TEMPLATE_SOURCE = "template_guess"
TEMPLATE_CONFIDENCE = 0.05

# Normalized median point locations extracted from local MTDDH-style demo annotations.
# No source images or original annotation files are included in the repository.
NORMALIZED_TEMPLATE: dict[str, tuple[float, float]] = {
    "left_acetabular_outer": (0.251922, 0.520707),
    "left_femoral_head_center": (0.285443, 0.615142),
    "left_femoral_head_lateral": (0.256843, 0.600594),
    "left_femoral_head_medial": (0.313732, 0.620762),
    "left_femoral_neck_axis_center": (0.260717, 0.708475),
    "left_femoral_neck_inner_lower": (0.294561, 0.769361),
    "left_femoral_shaft_dist": (0.237065, 0.985905),
    "left_femoral_shaft_prox": (0.237593, 0.857619),
    "left_obturator_upper": (0.431661, 0.727755),
    "left_teardrop_lower": (0.383311, 0.706163),
    "left_triradiate_center": (0.337676, 0.561693),
    "right_acetabular_outer": (0.763330, 0.478830),
    "right_femoral_head_center": (0.741758, 0.564381),
    "right_femoral_head_lateral": (0.771748, 0.551320),
    "right_femoral_head_medial": (0.704314, 0.572435),
    "right_femoral_neck_axis_center": (0.767001, 0.657673),
    "right_femoral_neck_inner_lower": (0.735646, 0.720082),
    "right_femoral_shaft_dist": (0.792392, 0.981020),
    "right_femoral_shaft_prox": (0.788824, 0.820476),
    "right_obturator_upper": (0.567568, 0.684762),
    "right_teardrop_lower": (0.634608, 0.663075),
    "right_triradiate_center": (0.682316, 0.523374),
}


def visible_keypoint_count(keypoints: dict[str, Keypoint]) -> int:
    return sum(1 for point in keypoints.values() if point.visible and point.x is not None and point.y is not None)


def template_keypoints_for_image(width: int | float, height: int | float, *, annotator: str = "") -> dict[str, Keypoint]:
    width = max(1.0, float(width))
    height = max(1.0, float(height))
    keypoints: dict[str, Keypoint] = {}
    for key, (x_norm, y_norm) in NORMALIZED_TEMPLATE.items():
        side, name = key.split("_", 1)
        keypoints[key] = make_keypoint(
            side,
            name,
            max(0.0, min(width, x_norm * width)),
            max(0.0, min(height, y_norm * height)),
            source=TEMPLATE_SOURCE,
            confidence=TEMPLATE_CONFIDENCE,
            annotator=annotator,
        )
    return keypoints


