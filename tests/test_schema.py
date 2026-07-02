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

    assert len(annotation.connections) == 25
    assert all(item.source == "default" for item in annotation.connections)
    pairs = {frozenset((item.point_a, item.point_b)) for item in annotation.connections}
    assert frozenset(("left_acetabular_outer", "left_triradiate_center")) in pairs
    assert frozenset(("left_femoral_head_lateral", "left_femoral_head_center")) in pairs
    assert frozenset(("right_obturator_upper", "left_obturator_upper")) in pairs


def test_legacy_annotation_without_connections_gets_defaults_but_empty_list_is_preserved():
    legacy = annotation_from_dict({"image": {"filename": "case.png", "width": 100, "height": 100}, "keypoints": {}})
    explicit_empty = annotation_from_dict(
        {"image": {"filename": "case.png", "width": 100, "height": 100}, "keypoints": {}, "connections": []}
    )

    assert len(legacy.connections) == 25
    assert explicit_empty.connections == []
