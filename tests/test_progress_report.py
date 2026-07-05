from pathlib import Path

from PIL import Image

from annotation_tool.progress_report import build_progress_payload, build_progress_rows
from annotation_tool.schema import create_blank_annotation, make_keypoint
from annotation_tool.storage import save_annotation, upsert_manifest_image


def _write_image(path: Path) -> None:
    Image.new("RGB", (80, 60), color="black").save(path)


def _fill_all_keypoints(annotation):
    for point in annotation.keypoints.values():
        annotation.keypoints[point.id] = make_keypoint(point.side, point.name, 10, 10, source="manual", confidence=1)


def _complete_shenton(annotation) -> None:
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


def test_progress_reports_are_written_next_to_dataset(tmp_path):
    pending_image = tmp_path / "pending.png"
    done_image = tmp_path / "done.png"
    _write_image(pending_image)
    _write_image(done_image)

    upsert_manifest_image(pending_image, root=tmp_path)

    annotation = create_blank_annotation("done.png", 80, 60, annotator="doctor-a")
    _fill_all_keypoints(annotation)
    _complete_shenton(annotation)
    save_annotation(annotation, tmp_path)
    upsert_manifest_image(done_image, annotation=annotation, root=tmp_path)

    assert (tmp_path / "HIP22_STATUS_DONE_1_TODO_1.txt").exists()
    assert (tmp_path / "HIP22_status_report.html").exists()
    assert (tmp_path / "HIP22_status_report.csv").exists()
    assert (tmp_path / "HIP22_SUBMISSION_README.txt").exists()
    html = (tmp_path / "HIP22_status_report.html").read_text(encoding="utf-8")
    assert "pending.png" in html
    assert "done.png" in html
    assert "badge done'>完成</span>" in html


def test_full_keypoints_without_shenton_are_keypoint_complete(tmp_path):
    image_path = tmp_path / "case.png"
    _write_image(image_path)
    annotation = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    _fill_all_keypoints(annotation)
    save_annotation(annotation, tmp_path)
    manifest = upsert_manifest_image(image_path, annotation=annotation, root=tmp_path)

    rows = build_progress_rows(tmp_path, manifest)
    payload = build_progress_payload(tmp_path, manifest)

    assert rows[0]["status"] == "keypoint_complete"
    assert rows[0]["keypoint_status"] == "complete"
    assert rows[0]["shenton_status"] == "pending"
    assert rows[0]["status_detail"] == "关键点 22/22；Shenton 0/2"
    assert payload["counts"]["done"] == 0
    assert payload["counts"]["keypoint_complete"] == 1
    assert payload["counts"]["needs_review"] == 1


def test_complete_shenton_without_keypoints_is_shenton_complete(tmp_path):
    image_path = tmp_path / "case.png"
    _write_image(image_path)
    annotation = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    _complete_shenton(annotation)
    save_annotation(annotation, tmp_path)
    manifest = upsert_manifest_image(image_path, annotation=annotation, root=tmp_path)

    rows = build_progress_rows(tmp_path, manifest)
    payload = build_progress_payload(tmp_path, manifest)

    assert rows[0]["status"] == "shenton_complete"
    assert rows[0]["keypoint_status"] == "pending"
    assert rows[0]["shenton_status"] == "complete"
    assert payload["counts"]["shenton_complete"] == 1
    assert payload["counts"]["needs_review"] == 1


def test_progress_payload_counts_review_buckets(tmp_path):
    image_path = tmp_path / "auto.png"
    _write_image(image_path)
    annotation = create_blank_annotation("auto.png", 80, 60, annotator="doctor-a")
    annotation.keypoints["left_acetabular_outer"] = make_keypoint(
        "left", "acetabular_outer", 20, 20, source="pose11_side", confidence=0.8
    )
    save_annotation(annotation, tmp_path)
    manifest = upsert_manifest_image(image_path, annotation=annotation, root=tmp_path)

    rows = build_progress_rows(tmp_path, manifest)
    payload = build_progress_payload(tmp_path, manifest)

    assert rows[0]["status"] == "auto"
    assert payload["counts"]["auto"] == 1
    assert payload["counts"]["needs_review"] == 1
