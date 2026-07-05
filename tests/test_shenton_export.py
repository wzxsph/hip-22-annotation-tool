import json

from PIL import Image

from annotation_tool.schema import create_blank_annotation
from annotation_tool.storage import save_annotation
from scripts.export_shenton_training_set import export_training_set, yolo_pose_lines


def _set_curve(annotation, side, segment, points):
    annotation.shenton_curves[side][segment]["points"] = [{"x": x, "y": y} for x, y in points]


def test_yolo_pose_lines_resample_shenton_curves_to_four_control_points():
    annotation = create_blank_annotation("case.png", 100, 100)
    _set_curve(annotation, "left", "obturator_upper_curve", [(10, 20), (20, 22), (30, 20), (40, 18), (50, 20)])

    lines = yolo_pose_lines(annotation)

    assert len(lines) == 1
    parts = lines[0].split()
    assert parts[0] == "0"
    assert len(parts) == 17
    assert parts[7] == "2"
    assert parts[-1] == "2"


def test_export_training_set_writes_jsonl_and_yolo_pose_dataset(tmp_path):
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
    assert result["yolo_images"] == 1
    assert result["yolo_objects"] == 2
    record = json.loads((tmp_path / "out" / "shenton_curves.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["shenton_review"]["left"]["status"] == "continuous"
    assert (tmp_path / "out" / "yolo_pose" / "images" / "train" / "shenton_00001.png").exists()
    label = (tmp_path / "out" / "yolo_pose" / "labels" / "train" / "shenton_00001.txt").read_text(encoding="utf-8")
    assert len(label.splitlines()) == 2
    data_yaml = (tmp_path / "out" / "yolo_pose" / "data.yaml").read_text(encoding="utf-8")
    assert "kpt_shape: [4, 3]" in data_yaml
    assert "obturator_shenton_arc" in data_yaml
