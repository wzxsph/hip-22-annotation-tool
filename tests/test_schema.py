from annotation_tool.schema import (
    LANDMARK_DEFS,
    OPTIONAL_LANDMARK_DEFS,
    REQUIRED_LANDMARK_DEFS,
    all_keypoint_ids,
    annotation_from_dict,
    create_blank_annotation,
    fill_inferred_femoral_neck_axis_proximal,
    is_optional_landmark_name,
    make_keypoint,
)


def test_template_contains_24_keypoints_in_hospital_order():
    annotation = create_blank_annotation("case.png", 1000, 800)

    assert len(LANDMARK_DEFS) == 12
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
        "femoral_neck_axis_proximal",
    ]
    assert LANDMARK_DEFS[6].label_zh == "股骨颈轴中心远端"
    assert LANDMARK_DEFS[11].label_zh == "股骨颈轴中心近端"
    assert len(annotation.keypoints) == 24
    assert list(annotation.keypoints) == all_keypoint_ids()
    assert all_keypoint_ids().count("left_femoral_neck_axis_proximal") == 1
    assert all_keypoint_ids().count("right_femoral_neck_axis_proximal") == 1
    assert [item.number for item in OPTIONAL_LANDMARK_DEFS] == [10, 11]
    assert len(REQUIRED_LANDMARK_DEFS) == 10
    assert is_optional_landmark_name("obturator_upper") is True
    assert is_optional_landmark_name("femoral_neck_inner_lower") is True
    assert is_optional_landmark_name("femoral_neck_axis_proximal") is False


def test_blank_points_are_saved_as_missing_not_removed():
    annotation = create_blank_annotation("case.png", 1000, 800)
    point = annotation.keypoints["left_acetabular_outer"]

    assert point.x is None
    assert point.y is None
    assert point.visible is False
    assert point.visibility == 0
    assert point.source == "missing"


def test_inferred_12_uses_midpoint_of_3_and_7_without_overwriting_manual_point():
    annotation = create_blank_annotation("case.png", 1000, 800)
    annotation.keypoints["left_femoral_head_center"] = make_keypoint(
        "left", "femoral_head_center", 100, 200, source="pose11_side", confidence=0.8
    )
    annotation.keypoints["left_femoral_neck_axis_center"] = make_keypoint(
        "left", "femoral_neck_axis_center", 140, 260, source="pose11_side", confidence=0.7
    )
    annotation.keypoints["right_femoral_head_center"] = make_keypoint(
        "right", "femoral_head_center", 300, 400, source="pose11_side", confidence=0.8
    )
    annotation.keypoints["right_femoral_neck_axis_center"] = make_keypoint(
        "right", "femoral_neck_axis_center", 360, 420, source="pose11_side", confidence=0.7
    )
    annotation.keypoints["right_femoral_neck_axis_proximal"] = make_keypoint(
        "right", "femoral_neck_axis_proximal", 999, 888, source="manual", confidence=1.0
    )

    filled = fill_inferred_femoral_neck_axis_proximal(annotation.keypoints)

    left = annotation.keypoints["left_femoral_neck_axis_proximal"]
    right = annotation.keypoints["right_femoral_neck_axis_proximal"]
    assert filled == 1
    assert left.visible is True
    assert left.x == 120
    assert left.y == 230
    assert left.source == "estimated"
    assert left.confidence == 0.7
    assert right.x == 999
    assert right.y == 888
    assert right.source == "manual"


def test_legacy_non_manual_12_is_recomputed_from_current_3_and_7():
    annotation = annotation_from_dict(
        {
            "image": {"filename": "case.png", "width": 1000, "height": 800},
            "keypoints": {
                "left_femoral_head_center": {
                    "x": 10,
                    "y": 20,
                    "visible": True,
                    "visibility": 2,
                    "source": "pose11_side",
                    "confidence": 0.8,
                },
                "left_femoral_neck_axis_center": {
                    "x": 30,
                    "y": 60,
                    "visible": True,
                    "visibility": 2,
                    "source": "pose11_side",
                    "confidence": 0.7,
                },
                "left_femoral_neck_axis_proximal": {
                    "x": 99,
                    "y": 99,
                    "visible": True,
                    "visibility": 2,
                    "source": "estimated",
                    "confidence": 0.5,
                },
            },
        }
    )

    point = annotation.keypoints["left_femoral_neck_axis_proximal"]
    assert point.x == 20
    assert point.y == 40
    assert point.source == "estimated"


def test_blank_annotation_has_default_editable_connections():
    annotation = create_blank_annotation("case.png", 1000, 800)

    assert len(annotation.connections) == 11
    assert all(item.source == "default" for item in annotation.connections)
    pairs = {frozenset((item.point_a, item.point_b)) for item in annotation.connections}
    assert frozenset(("left_acetabular_outer", "left_triradiate_center")) in pairs
    assert frozenset(("left_acetabular_outer", "left_femoral_head_center")) in pairs
    assert frozenset(("right_acetabular_outer", "right_femoral_head_center")) in pairs
    assert frozenset(("left_femoral_head_medial", "left_femoral_head_lateral")) in pairs
    assert frozenset(("left_femoral_neck_axis_proximal", "left_femoral_neck_axis_center")) in pairs
    assert frozenset(("right_femoral_head_medial", "right_femoral_head_lateral")) in pairs
    assert frozenset(("right_femoral_neck_axis_proximal", "right_femoral_neck_axis_center")) in pairs
    assert frozenset(("left_triradiate_center", "right_triradiate_center")) in pairs
    assert frozenset(("left_femoral_head_center", "left_femoral_head_medial")) not in pairs
    assert frozenset(("left_femoral_head_center", "left_femoral_head_lateral")) not in pairs
    assert frozenset(("left_femoral_head_center", "left_femoral_neck_axis_center")) not in pairs
    assert frozenset(("right_femoral_head_center", "right_femoral_head_medial")) not in pairs
    assert frozenset(("right_femoral_head_center", "right_femoral_head_lateral")) not in pairs
    assert frozenset(("right_femoral_head_center", "right_femoral_neck_axis_center")) not in pairs
    assert frozenset(("left_acetabular_outer", "left_teardrop_lower")) not in pairs
    assert frozenset(("right_acetabular_outer", "right_teardrop_lower")) not in pairs
    assert frozenset(("left_teardrop_lower", "right_teardrop_lower")) not in pairs


def test_legacy_annotation_without_connections_gets_defaults_but_empty_list_is_preserved():
    legacy = annotation_from_dict({"image": {"filename": "case.png", "width": 100, "height": 100}, "keypoints": {}})
    explicit_empty = annotation_from_dict(
        {"image": {"filename": "case.png", "width": 100, "height": 100}, "keypoints": {}, "connections": []}
    )

    assert len(legacy.connections) == 11
    assert explicit_empty.connections == []


def test_legacy_default_connections_remove_retired_edges_but_keep_manual_edges():
    legacy = annotation_from_dict(
        {
            "image": {"filename": "case.png", "width": 100, "height": 100},
            "keypoints": {},
            "connections": [
                {
                    "id": "old_default_left_3_8",
                    "point_a": "left_femoral_head_center",
                    "point_b": "left_femoral_head_medial",
                    "source": "default",
                    "visible": True,
                },
                {
                    "id": "old_default_left_3_9",
                    "point_a": "left_femoral_head_center",
                    "point_b": "left_femoral_head_lateral",
                    "source": "default",
                    "visible": True,
                },
                {
                    "id": "old_default_left_3_7",
                    "point_a": "left_femoral_head_center",
                    "point_b": "left_femoral_neck_axis_center",
                    "source": "default",
                    "visible": True,
                },
                {
                    "id": "manual_left_3_7",
                    "point_a": "left_femoral_head_center",
                    "point_b": "left_femoral_neck_axis_center",
                    "source": "manual",
                    "visible": True,
                },
                {
                    "id": "keep_default_left_1_3",
                    "point_a": "left_acetabular_outer",
                    "point_b": "left_femoral_head_center",
                    "source": "default",
                    "visible": True,
                },
            ],
        }
    )

    pairs_by_source = {(item.source, frozenset((item.point_a, item.point_b))) for item in legacy.connections}
    assert (
        "manual",
        frozenset(("left_femoral_head_center", "left_femoral_neck_axis_center")),
    ) in pairs_by_source
    assert (
        "default",
        frozenset(("left_acetabular_outer", "left_femoral_head_center")),
    ) in pairs_by_source
    assert (
        "default",
        frozenset(("left_femoral_head_center", "left_femoral_head_medial")),
    ) not in pairs_by_source
    assert (
        "default",
        frozenset(("left_femoral_head_center", "left_femoral_head_lateral")),
    ) not in pairs_by_source
    assert (
        "default",
        frozenset(("left_femoral_head_center", "left_femoral_neck_axis_center")),
    ) not in pairs_by_source


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
