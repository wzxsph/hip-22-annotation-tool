from pathlib import Path

from PIL import Image

from annotation_tool.progress_report import build_progress_payload, build_progress_rows
from annotation_tool.schema import create_blank_annotation, make_keypoint
from annotation_tool.storage import save_annotation, upsert_manifest_image


def _write_image(path: Path) -> None:
    Image.new("RGB", (80, 60), color="black").save(path)


def test_progress_reports_are_written_next_to_dataset(tmp_path):
    pending_image = tmp_path / "pending.png"
    done_image = tmp_path / "done.png"
    _write_image(pending_image)
    _write_image(done_image)

    upsert_manifest_image(pending_image, root=tmp_path)

    annotation = create_blank_annotation("done.png", 80, 60, annotator="doctor-a")
    for point in annotation.keypoints.values():
        annotation.keypoints[point.id] = make_keypoint(point.side, point.name, 10, 10, source="manual", confidence=1)
    save_annotation(annotation, tmp_path)
    upsert_manifest_image(done_image, annotation=annotation, root=tmp_path)

    assert (tmp_path / "HIP22_STATUS_DONE_1_TODO_1.txt").exists()
    assert (tmp_path / "HIP22_status_report.html").exists()
    assert (tmp_path / "HIP22_status_report.csv").exists()
    assert (tmp_path / "HIP22_SUBMISSION_README.txt").exists()
    html = (tmp_path / "HIP22_status_report.html").read_text(encoding="utf-8")
    assert "pending.png" in html
    assert "done.png" in html
    assert "已完成" in html


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
