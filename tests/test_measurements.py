import pytest

from annotation_tool.measurements import compute_measurements
from annotation_tool.schema import create_blank_annotation, make_keypoint


def _set_curve(annotation, side, segment, points):
    annotation.shenton_curves[side][segment]["points"] = [{"x": x, "y": y} for x, y in points]


def _set_point(annotation, side, name, x, y):
    annotation.keypoints[f"{side}_{name}"] = make_keypoint(side, name, x, y, source="manual", confidence=1)


def test_shenton_measurement_requires_three_points_per_curve():
    annotation = create_blank_annotation("case.png", 100, 100)
    _set_curve(annotation, "left", "obturator_upper_curve", [(10, 20), (20, 20)])
    _set_curve(annotation, "left", "femoral_neck_inner_lower_curve", [(25, 20), (35, 20), (45, 20)])

    measurement = compute_measurements(annotation)["shenton"]["left"]

    assert measurement["status"] == "unavailable"
    assert "continuous_candidate" not in measurement
    assert "tangent_angle_deg" not in measurement
    assert measurement["warnings"]


def test_shenton_measurement_outputs_endpoint_gap():
    annotation = create_blank_annotation("case.png", 100, 100)
    _set_curve(annotation, "left", "obturator_upper_curve", [(10, 20), (20, 20), (30, 20)])
    _set_curve(annotation, "left", "femoral_neck_inner_lower_curve", [(36, 20), (46, 20), (56, 20)])

    measurement = compute_measurements(annotation)["shenton"]["left"]

    assert measurement["status"] == "computed"
    assert measurement["gap_px"] == 6.0
    assert measurement["endpoint_gap_px"] == 6.0
    assert "tangent_angle_deg" not in measurement
    assert "continuous_candidate" not in measurement
    assert measurement["gap_mm"] is None


def test_shenton_measurement_uses_pixel_spacing_when_available():
    annotation = create_blank_annotation("case.dcm", 100, 100)
    annotation.image.pixel_spacing_row_mm = 0.2
    annotation.image.pixel_spacing_col_mm = 0.3
    _set_curve(annotation, "right", "obturator_upper_curve", [(10, 20), (20, 20), (30, 20)])
    _set_curve(annotation, "right", "femoral_neck_inner_lower_curve", [(36, 20), (46, 20), (56, 20)])

    measurement = compute_measurements(annotation)["shenton"]["right"]

    assert measurement["gap_px"] == 6.0
    assert measurement["gap_mm"] == 1.8


def test_shenton_measurement_accepts_many_curve_points():
    annotation = create_blank_annotation("case.png", 200, 120)
    obturator = [(10 + idx * 5, 40 + (idx % 3)) for idx in range(10)]
    femoral = [(70 + idx * 5, 41 + (idx % 2)) for idx in range(10)]
    _set_curve(annotation, "left", "obturator_upper_curve", obturator)
    _set_curve(annotation, "left", "femoral_neck_inner_lower_curve", femoral)

    measurement = compute_measurements(annotation)["shenton"]["left"]

    assert measurement["status"] == "computed"
    assert measurement["gap_px"] is not None
    assert "tangent_angle_deg" not in measurement


def test_shenton_measurement_does_not_bridge_endpoint_gap_by_extension():
    annotation = create_blank_annotation("case.png", 220, 140)
    _set_curve(annotation, "left", "obturator_upper_curve", [(10, 40), (20, 35), (30, 30)])
    _set_curve(annotation, "left", "femoral_neck_inner_lower_curve", [(50, 20), (60, 15), (70, 10)])

    measurement = compute_measurements(annotation)["shenton"]["left"]

    assert measurement["gap_px"] > 8
    assert "extension_gap_px" not in measurement
    assert "extension_projection" not in measurement
    assert "extension_points_px" not in measurement
    assert "continuous_candidate" not in measurement


def test_shenton_measurement_ignores_legacy_manual_intersection():
    annotation = create_blank_annotation("case.png", 220, 140)
    _set_curve(annotation, "left", "obturator_upper_curve", [(10, 40), (20, 35), (30, 30)])
    _set_curve(annotation, "left", "femoral_neck_inner_lower_curve", [(50, 20), (60, 15), (70, 10)])
    annotation.shenton_adjustments["left"]["extension_intersection"] = {
        "enabled": True,
        "x": 42,
        "y": 26,
        "source": "manual",
        "updated_at": "2026-07-05T00:00:00Z",
        "annotator": "doctor-a",
    }

    measurement = compute_measurements(annotation)["shenton"]["left"]

    assert "extension_projection" not in measurement
    assert "extension_source" not in measurement
    assert "extension_gap_px" not in measurement
    assert "extension_points_px" not in measurement
    assert not any("人工调整" in warning for warning in measurement["warnings"])


def test_shenton_measurement_does_not_compute_tangent_angle():
    annotation = create_blank_annotation("case.png", 220, 140)
    _set_curve(annotation, "left", "obturator_upper_curve", [(10, 20), (20, 20), (30, 20)])
    _set_curve(annotation, "left", "femoral_neck_inner_lower_curve", [(30, 55), (30, 45), (30, 35)])

    measurement = compute_measurements(annotation)["shenton"]["left"]

    assert measurement["gap_px"] == pytest.approx(15.0, abs=0.01)
    assert "tangent_angle_deg" not in measurement
    assert "continuous_candidate" not in measurement


def test_clinical_parameters_compute_ai_and_sharp_angles():
    annotation = create_blank_annotation("case.png", 200, 200)
    _set_point(annotation, "left", "triradiate_center", 50, 100)
    _set_point(annotation, "right", "triradiate_center", 150, 100)
    _set_point(annotation, "left", "acetabular_outer", 30, 80)
    _set_point(annotation, "right", "acetabular_outer", 170, 80)
    _set_point(annotation, "left", "teardrop_lower", 60, 140)
    _set_point(annotation, "right", "teardrop_lower", 140, 140)
    _set_point(annotation, "left", "femoral_head_center", 50, 130)
    _set_point(annotation, "left", "femoral_neck_axis_center", 45, 150)
    _set_point(annotation, "left", "femoral_shaft_prox", 45, 155)
    _set_point(annotation, "left", "femoral_shaft_dist", 45, 190)

    parameters = compute_measurements(annotation)["clinical_parameters"]["left"]

    assert parameters["status"] == "computed"
    assert parameters["ai_tonnis_angle_deg"] == 45.0
    assert parameters["sharp_angle_deg"] == pytest.approx(63.43, abs=0.01)
    assert parameters["ce_angle_deg"] == pytest.approx(21.8, abs=0.01)
    assert parameters["neck_shaft_angle_deg"] is not None


def test_acetabular_depth_uses_pixel_spacing_when_available():
    annotation = create_blank_annotation("case.dcm", 200, 200)
    annotation.image.pixel_spacing_row_mm = 0.2
    annotation.image.pixel_spacing_col_mm = 0.3
    _set_point(annotation, "left", "acetabular_outer", 30, 80)
    _set_point(annotation, "left", "teardrop_lower", 60, 140)
    _set_point(annotation, "left", "femoral_head_center", 50, 130)

    depth = compute_measurements(annotation)["acetabular_depth"]["left"]

    assert depth["status"] == "computed"
    assert depth["value_px"] == pytest.approx(4.47, abs=0.01)
    assert depth["value_mm"] == pytest.approx(1.2, abs=0.001)
    assert "teardrop_lower" in depth["warnings"][0] or "teardrop_lower" in depth["warnings"][1]
