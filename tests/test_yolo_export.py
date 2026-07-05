from annotation_tool.schema import create_blank_annotation, make_keypoint
from annotation_tool.template_points import template_keypoints_for_image
from annotation_tool.yolo_export import annotation_to_yolo_lines, data_yaml_text


def test_yolo_export_emits_11_rows_with_left_right_keypoints():
    annotation = create_blank_annotation("case.png", 1000, 500)
    annotation.keypoints["left_acetabular_outer"] = make_keypoint(
        "left", "acetabular_outer", 600, 200, source="manual", confidence=1
    )
    annotation.keypoints["right_acetabular_outer"] = make_keypoint(
        "right", "acetabular_outer", 400, 220, source="manual", confidence=1
    )
    annotation.keypoints["left_triradiate_center"] = make_keypoint(
        "left", "triradiate_center", 620, 260, source="manual", confidence=1
    )

    lines = annotation_to_yolo_lines(annotation)

    assert len(lines) == 11
    first = lines[0].split()
    assert first[0] == "0"
    assert len(first) == 11
    assert first[5:8] == ["0.6", "0.4", "2"]
    assert first[8:11] == ["0.4", "0.44", "2"]

    second = lines[1].split()
    assert second[0] == "1"
    assert second[5:8] == ["0.62", "0.52", "2"]
    assert second[8:11] == ["0", "0", "0"]


def test_data_yaml_matches_11_class_pose_contract():
    text = data_yaml_text(".")

    assert "kpt_shape: [2, 3]" in text
    assert "flip_idx: [1, 0]" in text
    assert "train: ." in text
    assert "val: ." in text
    assert "0: acetabular_outer" in text
    assert "10: femoral_neck_inner_lower" in text


def test_template_guess_points_export_as_valid_yolo_coordinates():
    annotation = create_blank_annotation("case.png", 1000, 500)
    annotation.keypoints.update(template_keypoints_for_image(1000, 500))

    lines = annotation_to_yolo_lines(annotation)

    assert len(lines) == 11
    for line in lines:
        values = [float(value) for value in line.split()[1:]]
        for value in values:
            assert 0 <= value <= 2


def test_export_zip_data_yaml_can_keep_standard_image_paths():
    text = data_yaml_text(".", train="images/train", val="images/val")

    assert "train: images/train" in text
    assert "val: images/val" in text
