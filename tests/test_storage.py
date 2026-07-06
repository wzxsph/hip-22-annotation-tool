import json

from annotation_tool.schema import create_blank_annotation, make_keypoint
from annotation_tool.storage import annotation_status, label_path, load_annotation, load_annotation_from_yolo_label, load_manifest, save_annotation
from annotation_tool.yolo_export import annotation_to_yolo_text


def test_annotation_round_trip_preserves_template_and_extra_fields(tmp_path):
    annotation = create_blank_annotation("case.png", 640, 480, annotator="doctor-a")
    point = annotation.keypoints["left_acetabular_outer"]
    point.x = 123.4
    point.y = 55.6
    point.visible = True
    point.visibility = 2
    point.source = "manual"
    point.confidence = 1.0
    annotation.review["note"] = "checked"

    save_annotation(annotation, tmp_path)
    loaded = load_annotation("case.png", tmp_path)

    assert loaded is not None
    assert len(loaded.keypoints) == 22
    assert loaded.keypoints["left_acetabular_outer"].x == 123.4
    assert loaded.keypoints["left_acetabular_outer"].source == "manual"
    assert loaded.review["note"] == "checked"


def test_shenton_round_trip_preserves_more_than_six_curve_points(tmp_path):
    annotation = create_blank_annotation("case.png", 640, 480, annotator="doctor-a")
    points = [{"x": float(idx * 10), "y": float(100 + idx)} for idx in range(9)]
    annotation.shenton_curves["left"]["obturator_upper_curve"]["points"] = points

    save_annotation(annotation, tmp_path)
    loaded = load_annotation("case.png", tmp_path)

    assert loaded is not None
    assert loaded.shenton_curves["left"]["obturator_upper_curve"]["points"] == points


def test_shenton_manual_extension_intersection_round_trip(tmp_path):
    annotation = create_blank_annotation("case.png", 640, 480, annotator="doctor-a")
    annotation.shenton_adjustments["left"]["extension_intersection"] = {
        "enabled": True,
        "x": 123.5,
        "y": 88.25,
        "source": "manual",
        "updated_at": "2026-07-05T00:00:00Z",
        "annotator": "doctor-a",
    }

    save_annotation(annotation, tmp_path)
    loaded = load_annotation("case.png", tmp_path)

    assert loaded is not None
    point = loaded.shenton_adjustments["left"]["extension_intersection"]
    assert point["enabled"] is True
    assert point["x"] == 123.5
    assert point["y"] == 88.25


def test_annotation_status_requires_keypoints_and_shenton():
    annotation = create_blank_annotation("case.png", 640, 480, annotator="doctor-a")
    for point in annotation.keypoints.values():
        annotation.keypoints[point.id] = make_keypoint(point.side, point.name, 10, 10, source="manual", confidence=1)

    assert annotation_status(annotation) == "in_progress"
    annotation.review["manual_keypoints_complete"] = {
        "status": "confirmed",
        "updated_at": "2026-07-06T00:00:00Z",
        "annotator": "doctor-a",
    }
    assert annotation_status(annotation) == "keypoint_complete"

    for side in ("left", "right"):
        annotation.shenton_curves[side]["obturator_upper_curve"]["points"] = [
            {"x": 10.0, "y": 30.0},
            {"x": 20.0, "y": 28.0},
            {"x": 30.0, "y": 26.0},
        ]
        annotation.shenton_curves[side]["femoral_neck_inner_lower_curve"]["points"] = [
            {"x": 30.0, "y": 26.0},
            {"x": 40.0, "y": 25.0},
            {"x": 50.0, "y": 24.0},
        ]
        annotation.shenton_review[side]["status"] = "continuous"

    assert annotation_status(annotation) == "done"


def test_annotation_status_can_mark_shenton_complete_before_keypoints():
    annotation = create_blank_annotation("case.png", 640, 480, annotator="doctor-a")
    for side in ("left", "right"):
        annotation.shenton_curves[side]["obturator_upper_curve"]["points"] = [
            {"x": 10.0, "y": 30.0},
            {"x": 20.0, "y": 28.0},
            {"x": 30.0, "y": 26.0},
        ]
        annotation.shenton_curves[side]["femoral_neck_inner_lower_curve"]["points"] = [
            {"x": 30.0, "y": 26.0},
            {"x": 40.0, "y": 25.0},
            {"x": 50.0, "y": 24.0},
        ]
        annotation.shenton_review[side]["status"] = "continuous"

    assert annotation_status(annotation) == "shenton_complete"


def test_roi_crop_round_trip_preserves_original_image_coordinates(tmp_path):
    annotation = create_blank_annotation("case.png", 100, 80, annotator="doctor-a")
    annotation.roi_crop = {
        "enabled": True,
        "x": 12.5,
        "y": 9.25,
        "width": 50.0,
        "height": 40.0,
        "source": "manual",
        "updated_at": "2026-07-05T00:00:00Z",
        "annotator": "doctor-a",
    }

    save_annotation(annotation, tmp_path)
    loaded = load_annotation("case.png", tmp_path)

    assert loaded is not None
    assert loaded.roi_crop["enabled"] is True
    assert loaded.roi_crop["x"] == 12.5
    assert loaded.roi_crop["width"] == 50.0


def test_legacy_minimal_point_payload_loads_without_losing_known_fields(tmp_path):
    annotations_dir = tmp_path / "annotations"
    annotations_dir.mkdir()
    payload = {
        "image": {"filename": "legacy.png", "width": 500, "height": 400},
        "annotator": {"user_id": "legacy", "created_at": "2026-01-01T00:00:00Z", "updated_at": "2026-01-01T00:00:00Z"},
        "keypoints": {
            "left_acetabular_outer": {
                "x": 10,
                "y": 20,
                "label": "custom label",
                "side": "left",
                "source": "manual",
                "confidence": 0.9,
                "custom_field": "keep-me",
            }
        },
    }
    (annotations_dir / "legacy.json").write_text(json.dumps(payload), encoding="utf-8")

    loaded = load_annotation("legacy.png", tmp_path)

    assert loaded is not None
    assert len(loaded.keypoints) == 22
    point = loaded.keypoints["left_acetabular_outer"]
    assert point.visible is True
    assert point.visibility == 2
    assert point.label == "custom label"
    assert getattr(point, "custom_field") == "keep-me"


def test_save_annotation_creates_dataset_layout_and_synced_yolo_label(tmp_path):
    annotation = create_blank_annotation("case.png", 640, 480, annotator="doctor-a")
    annotation.image.split = "train"
    annotation.keypoints["left_acetabular_outer"] = make_keypoint(
        "left", "acetabular_outer", 320, 240, source="manual", confidence=1
    )

    save_annotation(annotation, tmp_path)

    assert (tmp_path / "data.yaml").exists()
    assert (tmp_path / "annotations").is_dir()
    text = label_path("case.png", tmp_path, "train").read_text(encoding="utf-8")
    assert text.splitlines()[0].startswith("0 ")
    assert label_path("case.png", tmp_path, "train") == tmp_path / "case.txt"


def test_sidecar_yolo_label_loads_as_imported_annotation(tmp_path):
    from PIL import Image

    image_path = tmp_path / "case.png"
    Image.new("RGB", (100, 50), color="black").save(image_path)
    annotation = create_blank_annotation("case.png", 100, 50, annotator="doctor-a")
    annotation.keypoints["left_acetabular_outer"] = make_keypoint(
        "left", "acetabular_outer", 20, 10, source="manual", confidence=1
    )
    (tmp_path / "case.txt").write_text(annotation_to_yolo_text(annotation), encoding="utf-8")

    imported = load_annotation_from_yolo_label(image_path, tmp_path, annotator="doctor-b")

    assert imported is not None
    point = imported.keypoints["left_acetabular_outer"]
    assert point.x == 20
    assert point.y == 10
    assert point.source == "imported_label"


def test_corrupt_manifest_is_quarantined_instead_of_raising(tmp_path):
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text('{"images": [', encoding="utf-8")

    manifest = load_manifest(tmp_path)

    assert manifest.images == []
    assert not manifest_path.exists()
    assert list(tmp_path.glob("manifest.corrupt-*.json"))
