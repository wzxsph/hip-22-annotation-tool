from annotation_tool.schema import LANDMARK_DEFS, all_keypoint_ids, annotation_from_dict, create_blank_annotation


def test_template_contains_22_keypoints_in_hospital_order():
    annotation = create_blank_annotation("case.png", 1000, 800)

    assert len(LANDMARK_DEFS) == 11
    assert [item.name for item in LANDMARK_DEFS] == [
        "acetabular_outer",
        "triradiate_center",
        "femoral_head_center",
        "teardrop_lower",
        "femoral_shaft_prox",
        "femoral_shaft_dist",
        "femoral_neck_axis_center",
        "femoral_head_medial",
        "femoral_head_lateral",
        "obturator_upper",
        "femoral_neck_inner_lower",
    ]
    assert len(annotation.keypoints) == 22
    assert list(annotation.keypoints) == all_keypoint_ids()


def test_blank_points_are_saved_as_missing_not_removed():
    annotation = create_blank_annotation("case.png", 1000, 800)
    point = annotation.keypoints["left_acetabular_outer"]

    assert point.x is None
    assert point.y is None
    assert point.visible is False
    assert point.visibility == 0
    assert point.source == "missing"


def test_blank_annotation_has_default_editable_connections():
    annotation = create_blank_annotation("case.png", 1000, 800)

    assert len(annotation.connections) == 23
    assert all(item.source == "default" for item in annotation.connections)
    pairs = {frozenset((item.point_a, item.point_b)) for item in annotation.connections}
    assert frozenset(("left_acetabular_outer", "left_triradiate_center")) in pairs
    assert frozenset(("left_femoral_head_lateral", "left_femoral_head_center")) in pairs
    assert frozenset(("right_obturator_upper", "left_obturator_upper")) in pairs
    assert frozenset(("left_acetabular_outer", "left_teardrop_lower")) not in pairs
    assert frozenset(("right_acetabular_outer", "right_teardrop_lower")) not in pairs
    assert frozenset(("left_teardrop_lower", "right_teardrop_lower")) not in pairs


def test_legacy_annotation_without_connections_gets_defaults_but_empty_list_is_preserved():
    legacy = annotation_from_dict({"image": {"filename": "case.png", "width": 100, "height": 100}, "keypoints": {}})
    explicit_empty = annotation_from_dict(
        {"image": {"filename": "case.png", "width": 100, "height": 100}, "keypoints": {}, "connections": []}
    )

    assert len(legacy.connections) == 23
    assert explicit_empty.connections == []


def test_legacy_annotation_gets_shenton_defaults():
    legacy = annotation_from_dict({"image": {"filename": "case.png", "width": 100, "height": 100}, "keypoints": {}})

    assert legacy.shenton_curves["left"]["obturator_upper_curve"]["points"] == []
    assert legacy.shenton_curves["right"]["femoral_neck_inner_lower_curve"]["points"] == []
    assert legacy.shenton_review["left"]["status"] == "not_reviewed"
    assert legacy.shenton_adjustments["left"]["extension_intersection"]["enabled"] is False


def test_shenton_manual_extension_intersection_is_normalized():
    annotation = annotation_from_dict(
        {
            "image": {"filename": "case.png", "width": 100, "height": 80},
            "keypoints": {},
            "shenton_adjustments": {
                "left": {
                    "extension_intersection": {
                        "enabled": True,
                        "x": 120,
                        "y": 40,
                        "source": "manual",
                    }
                }
            },
        }
    )

    point = annotation.shenton_adjustments["left"]["extension_intersection"]
    assert point["enabled"] is True
    assert point["x"] == 100
    assert point["y"] == 40


def test_legacy_annotation_gets_roi_crop_default():
    legacy = annotation_from_dict({"image": {"filename": "case.png", "width": 100, "height": 100}, "keypoints": {}})

    assert legacy.roi_crop["enabled"] is False
    assert legacy.roi_crop["x"] is None


def test_legacy_annotation_gets_scan_transform_default():
    legacy = annotation_from_dict({"image": {"filename": "case.png", "width": 100, "height": 100}, "keypoints": {}})

    assert legacy.scan_transform["enabled"] is False
    assert legacy.scan_transform["corners"] == []


def test_scan_transform_normalizes_four_original_image_corners():
    annotation = annotation_from_dict(
        {
            "image": {"filename": "case.png", "width": 100, "height": 80},
            "keypoints": {},
            "scan_transform": {
                "enabled": True,
                "corners": [
                    {"x": -5, "y": 3},
                    {"x": 95, "y": 4},
                    {"x": 105, "y": 90},
                    {"x": 4, "y": 75},
                ],
            },
        }
    )

    assert annotation.scan_transform["enabled"] is True
    assert annotation.scan_transform["corners"][0] == {"x": 0.0, "y": 3.0}
    assert annotation.scan_transform["corners"][2] == {"x": 100.0, "y": 80.0}
