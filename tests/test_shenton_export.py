import json

from PIL import Image

from annotation_tool.schema import create_blank_annotation
from annotation_tool.storage import save_annotation
from scripts.export_shenton_training_set import (
    export_training_set,
    polyline_to_band_polygons,
    shenton_side_roi,
    yolo_seg_lines_for_side,
)


def _set_curve(annotation, side, segment, points):
    annotation.shenton_curves[side][segment]["points"] = [{"x": x, "y": y} for x, y in points]


def test_polyline_to_band_polygons_builds_mask_polygon():
    polygons = polyline_to_band_polygons(
        [{"x": 10, "y": 20}, {"x": 30, "y": 22}, {"x": 50, "y": 20}],
        80,
        60,
        band_width=7,
    )

    assert polygons
    assert len(polygons[0]) >= 3
    assert all(0 <= point["x"] < 80 and 0 <= point["y"] < 60 for point in polygons[0])


def test_yolo_seg_lines_write_normalized_roi_polygons_and_ignore_legacy_intersection():
    annotation = create_blank_annotation("case.png", 100, 100)
    _set_curve(annotation, "left", "obturator_upper_curve", [(10, 20), (20, 22), (30, 20), (40, 18), (50, 20)])
    annotation.shenton_adjustments["left"]["extension_intersection"] = {
        "enabled": True,
        "x": 42,
        "y": 26,
        "source": "manual",
    }
    roi = {"x": 0, "y": 0, "width": 80, "height": 60}

    lines = yolo_seg_lines_for_side(annotation, "left", roi, band_width=7)

    assert len(lines) >= 1
    parts = lines[0].split()
    assert parts[0] == "0"
    assert len(parts) >= 7
    coords = [float(item) for item in parts[1:]]
    assert all(0 <= value <= 1 for value in coords)


def test_shenton_side_roi_uses_curve_points_when_keypoints_are_missing():
    annotation = create_blank_annotation("case.png", 200, 160)
    _set_curve(annotation, "left", "obturator_upper_curve", [(80, 90), (90, 88), (100, 90)])

    roi = shenton_side_roi(annotation, "left")

    assert roi is not None
    assert roi["width"] >= 64
    assert roi["height"] >= 64
    assert roi["x"] <= 80 <= roi["x"] + roi["width"]
    assert roi["y"] <= 90 <= roi["y"] + roi["height"]


def test_export_training_set_writes_jsonl_and_yolo_seg_dataset(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    Image.new("RGB", (100, 100), color="black").save(workspace / "case.png")
    annotation = create_blank_annotation("case.png", 100, 100, annotator="doctor-a")
    _set_curve(annotation, "left", "obturator_upper_curve", [(10, 20), (20, 22), (30, 20)])
    _set_curve(annotation, "left", "femoral_neck_inner_lower_curve", [(36, 20), (46, 22), (56, 20)])
    annotation.shenton_review["left"]["status"] = "continuous"
    save_annotation(annotation, workspace)

    result = export_training_set(workspace, tmp_path / "out")

    assert result["jsonl_records"] == 1
    assert result["roi_images"] == 1
    assert result["yolo_objects"] >= 2
    record = json.loads((tmp_path / "out" / "shenton_seg_records.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["rois"][0]["shenton_review"]["status"] == "continuous"
    assert record["rois"][0]["roi_to_image"] == {"dx": 0, "dy": 0}
    assert "extension_intersection" not in json.dumps(record, ensure_ascii=False)
    assert (tmp_path / "out" / "yolo_seg" / "images" / "train" / "shenton_00001_left.png").exists()
    label = (tmp_path / "out" / "yolo_seg" / "labels" / "train" / "shenton_00001_left.txt").read_text(encoding="utf-8")
    assert len(label.splitlines()) >= 2
    data_yaml = (tmp_path / "out" / "yolo_seg" / "data.yaml").read_text(encoding="utf-8")
    assert "obturator_upper_curve" in data_yaml
    assert "femoral_neck_inner_lower_curve" in data_yaml
