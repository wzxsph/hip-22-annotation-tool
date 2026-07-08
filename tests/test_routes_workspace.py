from pathlib import Path
import os

from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from annotation_tool import auto_detection as auto_detection_module
from annotation_tool.heuristics import AutoAnnotationResult
from annotation_tool.schema import create_blank_annotation, make_keypoint
from annotation_tool.server import app
from annotation_tool.storage import annotation_path, load_annotation, load_manifest, load_settings, save_annotation, save_settings, upsert_manifest_image
from annotation_tool.yolo_export import annotation_to_yolo_text
from conftest import write_test_dicom


def _write_image(path: Path) -> None:
    Image.new("RGB", (80, 60), color="black").save(path)


def _fill_all_keypoints(annotation) -> None:
    for point in annotation.keypoints.values():
        annotation.keypoints[point.id] = make_keypoint(point.side, point.name, 10, 10, source="manual", confidence=1)


def test_open_folder_switches_workspace_without_mixing_previous_images(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    _write_image(first / "a.png")
    _write_image(second / "b.png")
    try:
        save_settings({"auto_detect": False})

        res = client.post("/api/annotation/open-folder", json={"folder_path": str(first)})
        assert res.status_code == 200
        assert res.json()["total"] == 1

        res = client.post("/api/annotation/open-folder", json={"folder_path": str(second)})
        assert res.status_code == 200
        assert res.json()["total"] == 1

        listed = client.get("/api/annotation/list").json()
        assert [Path(item["image_path"]).name for item in listed["images"]] == ["b.png"]
        assert listed["settings"]["dataset_root"] == str(second.resolve())
    finally:
        save_settings(old_settings)


def test_settings_persist_display_adjustments():
    old_settings = load_settings()
    client = TestClient(app)
    try:
        res = client.post("/api/annotation/settings", json={"display_brightness": 142, "display_contrast": 83})

        assert res.status_code == 200
        payload = res.json()
        assert payload["display_brightness"] == 142
        assert payload["display_contrast"] == 83
        assert load_settings()["display_brightness"] == 142
        assert load_settings()["display_contrast"] == 83

        clamped = client.post("/api/annotation/settings", json={"display_brightness": 999, "display_contrast": 1})
        assert clamped.status_code == 200
        assert clamped.json()["display_brightness"] == 180
        assert clamped.json()["display_contrast"] == 50
    finally:
        save_settings(old_settings)


def test_open_folder_marks_full_manual_keypoints_without_confirmation_as_in_progress(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_image(workspace / "case.png")
    annotation = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    _fill_all_keypoints(annotation)
    save_annotation(annotation, workspace)
    try:
        save_settings({"auto_detect": False})

        res = client.post("/api/annotation/open-folder", json={"folder_path": str(workspace)})
        assert res.status_code == 200

        listed = client.get("/api/annotation/list").json()
        item = listed["images"][0]
        assert item["status"] == "in_progress"
        assert item["keypoint_status"] == "in_progress"
        assert item["shenton_status"] == "pending"
        assert listed["progress"]["counts"]["done"] == 0
        assert listed["progress"]["counts"]["in_progress"] == 1
        assert listed["progress"]["counts"]["needs_review"] == 1
    finally:
        save_settings(old_settings)


def test_delete_image_moves_files_to_same_folder_trash_and_removes_manifest_entry(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    image_dir = workspace / "images" / "train"
    image_dir.mkdir(parents=True)
    image_path = image_dir / "case.png"
    _write_image(image_path)
    annotation = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    root_legacy_json = workspace / "case.json"
    root_legacy_json.write_text(annotation.model_dump_json() if hasattr(annotation, "model_dump_json") else annotation.json(), encoding="utf-8")
    save_annotation(annotation, workspace)
    upsert_manifest_image(image_path, annotation=annotation, root=workspace)
    root_sidecar = workspace / "case.txt"
    root_sidecar.write_text("sidecar", encoding="utf-8")
    try:
        save_settings({"dataset_root": str(workspace), "auto_detect": False})

        res = client.delete("/api/annotation/image/case.png")

        assert res.status_code == 200
        payload = res.json()
        assert payload["filename"] == "case.png"
        assert Path(payload["trash_dir"]) == Path("images/train/trash")
        assert not image_path.exists()
        assert not annotation_path("case.png", workspace).exists()
        assert not root_legacy_json.exists()
        assert not image_path.with_suffix(".txt").exists()
        assert not root_sidecar.exists()
        trash_dir = image_dir / "trash"
        assert (trash_dir / "case.png").exists()
        trashed_sources = {Path(item["source"]) for item in payload["trashed"]}
        assert Path("images/train/case.png") in trashed_sources
        assert Path("annotations/case.json") in trashed_sources
        assert Path("case.json") in trashed_sources
        assert Path("images/train/case.txt") in trashed_sources
        assert Path("case.txt") in trashed_sources
        assert all((workspace / item["trash_path"]).parent == trash_dir for item in payload["trashed"])
        assert all((workspace / item["trash_path"]).exists() for item in payload["trashed"])
        assert load_manifest(workspace).images == []
    finally:
        save_settings(old_settings)


def test_batch_delete_images_moves_each_image_and_reports_missing(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    image_dir = workspace / "images" / "train"
    image_dir.mkdir(parents=True)
    first = image_dir / "first.png"
    second = image_dir / "second.png"
    _write_image(first)
    _write_image(second)
    for image_path in (first, second):
        annotation = create_blank_annotation(image_path.name, 80, 60, annotator="doctor-a")
        save_annotation(annotation, workspace)
        upsert_manifest_image(image_path, annotation=annotation, root=workspace)
        image_path.with_suffix(".txt").write_text("label", encoding="utf-8")
    try:
        save_settings({"dataset_root": str(workspace), "auto_detect": False})

        res = client.post("/api/annotation/images/delete", json={"filenames": ["first.png", "second.png", "missing.png"]})

        assert res.status_code == 200
        payload = res.json()
        assert payload["status"] == "partial"
        assert payload["deleted_count"] == 2
        assert payload["failed_count"] == 1
        assert {item["filename"] for item in payload["results"]} == {"first.png", "second.png"}
        assert payload["errors"] == [{"filename": "missing.png", "detail": "Image not found."}]
        assert not first.exists()
        assert not second.exists()
        assert (image_dir / "trash" / "first.png").exists()
        assert (image_dir / "trash" / "second.png").exists()
        assert load_manifest(workspace).images == []
    finally:
        save_settings(old_settings)


def test_trash_list_and_restore_image_rebuilds_manifest(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    trash_dir = workspace / "images" / "train" / "trash"
    trash_dir.mkdir(parents=True)
    image_path = trash_dir / "case.png"
    _write_image(image_path)
    annotation = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    (trash_dir / "case.json").write_text(
        annotation.model_dump_json() if hasattr(annotation, "model_dump_json") else annotation.json(),
        encoding="utf-8",
    )
    (trash_dir / "case.txt").write_text("label", encoding="utf-8")
    try:
        save_settings({"dataset_root": str(workspace), "auto_detect": False})

        listed = client.get("/api/annotation/trash")

        assert listed.status_code == 200
        payload = listed.json()
        assert payload["total"] == 1
        assert payload["items"][0]["trash_path"] == "images/train/trash/case.png"
        assert payload["items"][0]["sidecar_count"] == 2

        restored = client.post("/api/annotation/trash/restore", json={"trash_paths": ["images/train/trash/case.png"]})

        assert restored.status_code == 200
        restore_payload = restored.json()
        assert restore_payload["status"] == "success"
        assert restore_payload["restored_count"] == 1
        restored_image = workspace / "images" / "train" / "case.png"
        assert restored_image.exists()
        assert (workspace / "images" / "train" / "case.json").exists()
        assert (workspace / "images" / "train" / "case.txt").exists()
        assert not image_path.exists()
        manifest = load_manifest(workspace)
        assert len(manifest.images) == 1
        assert manifest.images[0].image_path == "images/train/case.png"
        assert manifest.images[0].status == "pending"
    finally:
        save_settings(old_settings)


def test_list_refreshes_manifest_when_annotation_json_is_newer(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image_path = workspace / "case.png"
    _write_image(image_path)
    annotation = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    save_annotation(annotation, workspace)
    upsert_manifest_image(image_path, annotation=annotation, root=workspace)
    stored_path = annotation_path("case.png", workspace)
    old_mtime_ns = stored_path.stat().st_mtime_ns

    _fill_all_keypoints(annotation)
    save_annotation(annotation, workspace)
    if stored_path.stat().st_mtime_ns == old_mtime_ns:
        os.utime(stored_path, ns=(old_mtime_ns + 1_000_000, old_mtime_ns + 1_000_000))

    try:
        save_settings({"dataset_root": str(workspace), "auto_detect": False})

        listed = client.get("/api/annotation/list").json()

        item = listed["images"][0]
        assert item["status"] == "in_progress"
        assert item["keypoint_status"] == "in_progress"
        assert item["shenton_status"] == "pending"
        assert item["annotation_mtime_ns"] == stored_path.stat().st_mtime_ns
    finally:
        save_settings(old_settings)


def test_open_folder_imports_existing_sidecar_yolo_label_without_auto_overwrite(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_image(workspace / "case.png")
    annotation = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    annotation.keypoints["left_acetabular_outer"] = make_keypoint(
        "left", "acetabular_outer", 40, 30, source="manual", confidence=1
    )
    (workspace / "case.txt").write_text(annotation_to_yolo_text(annotation), encoding="utf-8")
    try:
        save_settings({"auto_detect": True})

        res = client.post("/api/annotation/open-folder", json={"folder_path": str(workspace)})
        assert res.status_code == 200
        payload = res.json()
        assert payload["queued"] == 0
        assert payload["existing"] == 1

        imported = load_annotation("case.png", workspace)
        assert imported is not None
        point = imported.keypoints["left_acetabular_outer"]
        assert point.x == 40
        assert point.y == 30
        assert point.source == "imported_label"
    finally:
        save_settings(old_settings)


def test_open_folder_imports_legacy_json_only_folder_by_copying_sibling_images(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "legacy-json"
    source_images = tmp_path / "source-images"
    workspace.mkdir()
    source_images.mkdir()
    _write_image(source_images / "case.png")
    annotation = create_blank_annotation("case.png", 80, 60, annotator="legacy")
    annotation.keypoints["left_acetabular_outer"] = make_keypoint(
        "left", "acetabular_outer", 40, 30, source="manual", confidence=1
    )
    (workspace / "case.json").write_text(annotation.model_dump_json() if hasattr(annotation, "model_dump_json") else annotation.json(), encoding="utf-8")
    try:
        save_settings({"auto_detect": True})

        res = client.post("/api/annotation/open-folder", json={"folder_path": str(workspace)})

        assert res.status_code == 200
        payload = res.json()
        assert payload["total"] == 1
        assert payload["queued"] == 0
        assert payload["existing"] == 1
        assert payload["import_report"]["legacy_annotations"] == 1
        assert payload["import_report"]["copied_external_images"] == 1
        assert (workspace / "images" / "train" / "case.png").exists()
        imported = load_annotation("case.png", workspace)
        assert imported is not None
        assert imported.keypoints["left_acetabular_outer"].x == 40
    finally:
        save_settings(old_settings)


def test_open_folder_legacy_json_without_matching_image_does_not_switch_workspace(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    current = tmp_path / "current"
    workspace = tmp_path / "legacy-json"
    current.mkdir()
    workspace.mkdir()
    annotation = create_blank_annotation("missing.png", 80, 60, annotator="legacy")
    (workspace / "missing.json").write_text(annotation.model_dump_json() if hasattr(annotation, "model_dump_json") else annotation.json(), encoding="utf-8")
    try:
        save_settings({"dataset_root": str(current), "auto_detect": False})

        res = client.post("/api/annotation/open-folder", json={"folder_path": str(workspace)})

        assert res.status_code == 400
        assert "找到旧标注 JSON" in res.json()["detail"]
        assert load_settings()["dataset_root"] == str(current.resolve())
    finally:
        save_settings(old_settings)


def test_open_folder_prefers_annotations_dir_json_over_root_legacy_json(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    annotations_dir = workspace / "annotations"
    annotations_dir.mkdir(parents=True)
    _write_image(workspace / "case.png")
    root_annotation = create_blank_annotation("case.png", 80, 60, annotator="root")
    root_annotation.keypoints["left_acetabular_outer"] = make_keypoint(
        "left", "acetabular_outer", 10, 10, source="manual", confidence=1
    )
    preferred = create_blank_annotation("case.png", 80, 60, annotator="preferred")
    preferred.keypoints["left_acetabular_outer"] = make_keypoint(
        "left", "acetabular_outer", 55, 30, source="manual", confidence=1
    )
    (workspace / "case.json").write_text(root_annotation.model_dump_json() if hasattr(root_annotation, "model_dump_json") else root_annotation.json(), encoding="utf-8")
    (annotations_dir / "case.json").write_text(preferred.model_dump_json() if hasattr(preferred, "model_dump_json") else preferred.json(), encoding="utf-8")
    try:
        save_settings({"auto_detect": False})

        res = client.post("/api/annotation/open-folder", json={"folder_path": str(workspace)})

        assert res.status_code == 200
        imported = load_annotation("case.png", workspace)
        assert imported is not None
        assert imported.keypoints["left_acetabular_outer"].x == 55
    finally:
        save_settings(old_settings)


def test_open_folder_writes_manifest_once_for_batch_import(monkeypatch, tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for index in range(5):
        _write_image(workspace / f"case-{index}.png")
    import annotation_tool.routes as routes_module

    original_save_manifest = routes_module.save_manifest
    calls = []

    def counting_save_manifest(*args, **kwargs):
        calls.append(1)
        return original_save_manifest(*args, **kwargs)

    monkeypatch.setattr(routes_module, "save_manifest", counting_save_manifest)
    try:
        save_settings({"auto_detect": False})

        res = client.post("/api/annotation/open-folder", json={"folder_path": str(workspace)})

        assert res.status_code == 200
        assert res.json()["total"] == 5
        assert len(calls) == 1
    finally:
        save_settings(old_settings)


def test_load_by_name_existing_annotation_does_not_rewrite_manifest(monkeypatch, tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image_path = workspace / "case.png"
    _write_image(image_path)
    annotation = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    save_annotation(annotation, workspace)
    import annotation_tool.routes as routes_module

    calls = []

    def fail_upsert(*args, **kwargs):
        calls.append(1)
        raise AssertionError("load-by-name should not rewrite manifest for an existing annotation")

    monkeypatch.setattr(routes_module, "upsert_manifest_image", fail_upsert)
    try:
        save_settings({"dataset_root": str(workspace), "auto_detect": False})

        res = client.get("/api/annotation/load-by-name", params={"filename": "case.png"})

        assert res.status_code == 200
        assert calls == []
    finally:
        save_settings(old_settings)


def test_save_can_skip_manifest_rewrite(monkeypatch, tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_image(workspace / "case.png")
    annotation = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    import annotation_tool.routes as routes_module

    calls = []

    def fail_upsert(*args, **kwargs):
        calls.append(1)
        raise AssertionError("skip_manifest save should not rewrite manifest")

    monkeypatch.setattr(routes_module, "upsert_manifest_image", fail_upsert)
    try:
        save_settings({"dataset_root": str(workspace), "auto_detect": False})

        res = client.post("/api/annotation/save?skip_manifest=true", json=annotation.model_dump() if hasattr(annotation, "model_dump") else annotation.dict())

        assert res.status_code == 200
        payload = res.json()
        assert payload["manifest_skipped"] is True
        assert payload["annotation_status"] == "pending"
        assert payload["annotation_progress"]["keypoint_status"] == "pending"
        assert payload["annotation_progress"]["shenton_status"] == "pending"
        assert calls == []
    finally:
        save_settings(old_settings)


def test_manual_keypoint_confirmation_marks_keypoints_complete_even_with_missing_points(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_image(workspace / "case.png")
    annotation = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    annotation.keypoints["left_acetabular_outer"] = make_keypoint(
        "left", "acetabular_outer", 20, 20, source="manual", confidence=1
    )
    annotation.review["manual_keypoints_complete"] = {
        "status": "confirmed",
        "updated_at": "2026-07-06T00:00:00Z",
        "annotator": "doctor-a",
    }
    try:
        save_settings({"dataset_root": str(workspace), "auto_detect": False})

        res = client.post("/api/annotation/save", json=annotation.model_dump() if hasattr(annotation, "model_dump") else annotation.dict())

        assert res.status_code == 200
        payload = res.json()
        assert payload["annotation_progress"]["keypoint_status"] == "complete"
        assert payload["annotation_status"] == "keypoint_complete"
        listed = client.get("/api/annotation/list").json()
        assert listed["images"][0]["keypoint_status"] == "complete"
    finally:
        save_settings(old_settings)


def test_image_endpoint_serves_cached_thumbnail(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write_image(workspace / "case.png")
    try:
        save_settings({"dataset_root": str(workspace), "auto_detect": False})

        res = client.get("/api/annotation/image/case.png?thumb=1")

        assert res.status_code == 200
        assert res.headers["content-type"].startswith("image/jpeg")
        assert len(res.content) < (workspace / "case.png").stat().st_size or len(res.content) < 10_000
    finally:
        save_settings(old_settings)


def test_image_endpoint_renders_tiff_original_and_enhanced_as_png(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    pixels = np.linspace(1, 65000, 80 * 60, dtype=np.uint16).reshape(60, 80)
    Image.fromarray(pixels).save(workspace / "case.tif")
    try:
        save_settings({"dataset_root": str(workspace), "auto_detect": False})

        original = client.get("/api/annotation/image/case.tif")
        enhanced = client.get("/api/annotation/image/case.tif?enhanced=true")

        assert original.status_code == 200
        assert enhanced.status_code == 200
        assert original.headers["content-type"].startswith("image/png")
        assert enhanced.headers["content-type"].startswith("image/png")
        assert original.content != enhanced.content
    finally:
        save_settings(old_settings)


def test_open_folder_returns_hospital_import_warnings(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    nested = workspace / "nested"
    nested.mkdir(parents=True)
    _write_image(workspace / "duplicate.png")
    _write_image(workspace / "case one.png")
    _write_image(nested / "duplicate.png")
    (workspace / "scan.dcm").write_bytes(b"not-supported")
    try:
        save_settings({"auto_detect": False})

        res = client.post("/api/annotation/open-folder", json={"folder_path": str(workspace)})

        assert res.status_code == 200
        report = res.json()["import_report"]
        assert report["nested_images"] == ["nested/duplicate.png"]
        assert report["duplicate_names"][0]["filename"] == "duplicate.png"
        assert report["unreadable_files"] == ["scan.dcm"]
        assert report["messy_names"][0]["filename"] == "case one.png"
        assert report["warnings"]
    finally:
        save_settings(old_settings)


def test_open_folder_imports_dicom_and_serves_png_without_phi(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    write_test_dicom(workspace / "case.dcm")
    try:
        save_settings({"auto_detect": False})

        res = client.post("/api/annotation/open-folder", json={"folder_path": str(workspace)})

        assert res.status_code == 200
        assert res.json()["total"] == 1
        loaded = client.get("/api/annotation/load-by-name", params={"filename": "case.dcm"})
        assert loaded.status_code == 200
        payload = loaded.json()
        assert payload["image"]["source_format"] == "dicom"
        assert payload["image"]["pixel_spacing_row_mm"] == 0.2
        assert "PatientName" not in payload["image"]

        image = client.get("/api/annotation/image/case.dcm")
        assert image.status_code == 200
        assert image.headers["content-type"].startswith("image/png")
    finally:
        save_settings(old_settings)


def test_upload_load_accepts_dicom(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    dicom_path = tmp_path / "upload.dcm"
    write_test_dicom(dicom_path)
    try:
        save_settings({"dataset_root": str(workspace), "auto_detect": False})

        with open(dicom_path, "rb") as handle:
            res = client.post("/api/annotation/load", files={"file": ("upload.dcm", handle, "application/dicom")})

        assert res.status_code == 200
        payload = res.json()
        assert payload["image"]["source_format"] == "dicom"
        assert (workspace / "upload.dcm").exists()
    finally:
        save_settings(old_settings)


def test_manual_auto_detect_preserves_existing_manual_points(monkeypatch, tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image_path = workspace / "case.png"
    _write_image(image_path)
    existing = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    existing.keypoints["left_acetabular_outer"] = make_keypoint(
        "left", "acetabular_outer", 40, 30, source="manual", confidence=1
    )
    save_annotation(existing, workspace)
    detected = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    detected.keypoints["left_acetabular_outer"] = make_keypoint(
        "left", "acetabular_outer", 5, 5, source="pose11_side", confidence=0.8
    )
    detected.keypoints["right_acetabular_outer"] = make_keypoint(
        "right", "acetabular_outer", 70, 30, source="pose11_side", confidence=0.8
    )

    def fake_estimate(image, *, min_visible_keypoints=None, include_partial=False):
        return AutoAnnotationResult(
            keypoints=detected.keypoints,
            warnings=[],
            model_available=True,
            source="yolo11n-best-side11",
            attempts=[{"strategy": "test", "visible_count": 2, "success": True}],
            strategy="test",
        )

    monkeypatch.setattr(auto_detection_module, "estimate_keypoints_from_image", fake_estimate)
    try:
        save_settings({"dataset_root": str(workspace)})

        res = client.post("/api/annotation/auto-detect-image", json={"filename": "case.png", "use_enhanced": True})

        assert res.status_code == 200
        payload = res.json()
        assert payload["auto_detect"]["applied"] is True
        assert payload["auto_detect"]["image_preprocess"] == "hip_demo_enhanced"
        assert payload["auto_detect"]["visible_count"] == 2
        saved = load_annotation("case.png", workspace)
        assert saved is not None
        assert saved.keypoints["left_acetabular_outer"].x == 40
        assert saved.keypoints["left_acetabular_outer"].source == "manual"
        assert saved.keypoints["right_acetabular_outer"].x == 70
        assert saved.keypoints["right_acetabular_outer"].source == "pose11_side"
    finally:
        save_settings(old_settings)


def test_manual_auto_detect_no_result_records_warning_without_overwriting(monkeypatch, tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image_path = workspace / "case.png"
    _write_image(image_path)
    existing = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    existing.keypoints["left_acetabular_outer"] = make_keypoint(
        "left", "acetabular_outer", 40, 30, source="manual", confidence=1
    )
    save_annotation(existing, workspace)
    blank = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")

    def fake_estimate(image, *, min_visible_keypoints=None, include_partial=False):
        return AutoAnnotationResult(
            keypoints=blank.keypoints,
            warnings=["no visible keypoints"],
            model_available=True,
            source="model-no-result",
            attempts=[{"strategy": "test", "visible_count": 0, "success": False}],
            strategy="no-result",
        )

    monkeypatch.setattr(auto_detection_module, "estimate_keypoints_from_image", fake_estimate)
    try:
        save_settings({"dataset_root": str(workspace)})

        res = client.post("/api/annotation/auto-detect-image", json={"filename": "case.png"})

        assert res.status_code == 200
        payload = res.json()
        assert payload["auto_detect"]["applied"] is False
        assert payload["auto_detect"]["visible_count"] == 0
        assert payload["auto_detect"]["applied_visible_count"] == 1
        assert payload["auto_detect"]["template_fallback"]["enabled"] is False
        assert payload["auto_detect"]["template_fallback"]["filled_count"] == 0
        saved = load_annotation("case.png", workspace)
        assert saved is not None
        assert saved.keypoints["left_acetabular_outer"].x == 40
        assert saved.keypoints["left_acetabular_outer"].source == "manual"
        assert saved.keypoints["right_acetabular_outer"].source == "missing"
        assert saved.keypoints["right_acetabular_outer"].visible is False
        assert saved.auto_initialization["source"] == "model-no-result"
        assert saved.auto_initialization["template_fallback"]["filled_count"] == 0
        assert saved.auto_initialization["model_visible_count"] == 0
        assert saved.auto_initialization["warnings"][0] == "no visible keypoints"
    finally:
        save_settings(old_settings)


def test_manual_auto_detect_uses_roi_and_maps_points_to_original_coordinates(monkeypatch, tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image_path = workspace / "case.png"
    _write_image(image_path)
    existing = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    existing.roi_crop = {
        "enabled": True,
        "x": 10,
        "y": 15,
        "width": 40,
        "height": 30,
        "source": "manual",
        "updated_at": "2026-07-05T00:00:00Z",
        "annotator": "doctor-a",
    }
    save_annotation(existing, workspace)
    seen_sizes = []

    def fake_estimate(image, *, min_visible_keypoints=None, include_partial=False):
        seen_sizes.append(image.size)
        detected = create_blank_annotation("case.png", image.width, image.height, annotator="doctor-a")
        detected.keypoints["right_acetabular_outer"] = make_keypoint(
            "right", "acetabular_outer", 5, 6, source="pose11_side", confidence=0.8
        )
        return AutoAnnotationResult(
            keypoints=detected.keypoints,
            warnings=[],
            model_available=True,
            source="yolo11n-best-side11",
            attempts=[{"strategy": "test", "visible_count": 1, "success": True}],
            strategy="test",
        )

    monkeypatch.setattr(auto_detection_module, "estimate_keypoints_from_image", fake_estimate)
    try:
        save_settings({"dataset_root": str(workspace)})

        res = client.post("/api/annotation/auto-detect-image", json={"filename": "case.png", "use_enhanced": False})

        assert res.status_code == 200
        payload = res.json()
        assert seen_sizes == [(40, 30)]
        assert payload["auto_detect"]["roi_crop_used"]["x"] == 10.0
        assert payload["auto_detect"]["scan_transform_used"] is None
        assert payload["auto_detect"]["image_preprocess"] == "roi_crop+original"
        assert payload["auto_detect"]["visible_count"] == 1
        saved = load_annotation("case.png", workspace)
        assert saved is not None
        point = saved.keypoints["right_acetabular_outer"]
        assert point.x == 15
        assert point.y == 21
        assert saved.auto_initialization["template_fallback"]["enabled"] is False
        assert saved.auto_initialization["template_fallback"]["filled_count"] == 0
    finally:
        save_settings(old_settings)


def test_manual_auto_detect_prefers_roi_over_scan_transform(monkeypatch, tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image_path = workspace / "case.png"
    _write_image(image_path)
    existing = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    existing.roi_crop = {
        "enabled": True,
        "x": 10,
        "y": 15,
        "width": 40,
        "height": 30,
        "source": "manual",
        "updated_at": "2026-07-05T00:00:00Z",
        "annotator": "doctor-a",
    }
    existing.scan_transform = {
        "enabled": True,
        "corners": [
            {"x": 10, "y": 5},
            {"x": 70, "y": 5},
            {"x": 70, "y": 55},
            {"x": 10, "y": 55},
        ],
        "source": "manual",
        "updated_at": "2026-07-05T00:00:00Z",
        "annotator": "doctor-a",
    }
    save_annotation(existing, workspace)
    seen_sizes = []

    def fake_estimate(image, *, min_visible_keypoints=None, include_partial=False):
        seen_sizes.append(image.size)
        detected = create_blank_annotation("case.png", image.width, image.height, annotator="doctor-a")
        detected.keypoints["right_acetabular_outer"] = make_keypoint(
            "right", "acetabular_outer", 5, 6, source="pose11_side", confidence=0.8
        )
        return AutoAnnotationResult(
            keypoints=detected.keypoints,
            warnings=[],
            model_available=True,
            source="yolo11n-best-side11",
            attempts=[{"strategy": "test", "visible_count": 1, "success": True}],
            strategy="test",
        )

    monkeypatch.setattr(auto_detection_module, "estimate_keypoints_from_image", fake_estimate)
    try:
        save_settings({"dataset_root": str(workspace)})

        res = client.post(
            "/api/annotation/auto-detect-image",
            json={"filename": "case.png", "use_enhanced": False, "use_roi": True, "use_scan": True},
        )

        assert res.status_code == 200
        payload = res.json()
        assert seen_sizes == [(40, 30)]
        assert payload["auto_detect"]["roi_crop_used"]["x"] == 10.0
        assert payload["auto_detect"]["scan_transform_used"] is None
        assert payload["auto_detect"]["image_preprocess"] == "roi_crop+original"
        saved = load_annotation("case.png", workspace)
        assert saved is not None
        point = saved.keypoints["right_acetabular_outer"]
        assert point.x == 15
        assert point.y == 21
    finally:
        save_settings(old_settings)


def test_manual_auto_detect_enhanced_roi_preprocess_label(monkeypatch, tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image_path = workspace / "case.png"
    _write_image(image_path)
    existing = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    existing.roi_crop = {
        "enabled": True,
        "x": 10,
        "y": 15,
        "width": 40,
        "height": 30,
        "source": "manual",
        "updated_at": "2026-07-05T00:00:00Z",
        "annotator": "doctor-a",
    }
    save_annotation(existing, workspace)
    seen_sizes = []

    def fake_estimate(image, *, min_visible_keypoints=None, include_partial=False):
        seen_sizes.append(image.size)
        detected = create_blank_annotation("case.png", image.width, image.height, annotator="doctor-a")
        detected.keypoints["right_acetabular_outer"] = make_keypoint(
            "right", "acetabular_outer", 5, 6, source="pose11_side", confidence=0.8
        )
        return AutoAnnotationResult(
            keypoints=detected.keypoints,
            warnings=[],
            model_available=True,
            source="yolo11n-best-side11",
            attempts=[{"strategy": "test", "visible_count": 1, "success": True}],
            strategy="test",
        )

    monkeypatch.setattr(auto_detection_module, "estimate_keypoints_from_image", fake_estimate)
    try:
        save_settings({"dataset_root": str(workspace)})

        res = client.post("/api/annotation/auto-detect-image", json={"filename": "case.png", "use_enhanced": True})

        assert res.status_code == 200
        payload = res.json()
        assert seen_sizes == [(40, 30)]
        assert payload["auto_detect"]["roi_crop_used"]["x"] == 10.0
        assert payload["auto_detect"]["image_preprocess"] == "roi_crop+hip_demo_enhanced"
    finally:
        save_settings(old_settings)


def test_manual_auto_detect_uses_scan_transform_and_maps_points_to_original_coordinates(monkeypatch, tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image_path = workspace / "case.png"
    _write_image(image_path)
    existing = create_blank_annotation("case.png", 80, 60, annotator="doctor-a")
    existing.scan_transform = {
        "enabled": True,
        "corners": [
            {"x": 10, "y": 5},
            {"x": 70, "y": 5},
            {"x": 70, "y": 55},
            {"x": 10, "y": 55},
        ],
        "source": "manual",
        "updated_at": "2026-07-05T00:00:00Z",
        "annotator": "doctor-a",
    }
    save_annotation(existing, workspace)
    seen_sizes = []

    def fake_estimate(image, *, min_visible_keypoints=None, include_partial=False):
        seen_sizes.append(image.size)
        detected = create_blank_annotation("case.png", image.width, image.height, annotator="doctor-a")
        detected.keypoints["left_acetabular_outer"] = make_keypoint(
            "left", "acetabular_outer", 0, 0, source="pose11_side", confidence=0.8
        )
        return AutoAnnotationResult(
            keypoints=detected.keypoints,
            warnings=[],
            model_available=True,
            source="yolo11n-best-side11",
            attempts=[{"strategy": "test", "visible_count": 1, "success": True}],
            strategy="test",
        )

    monkeypatch.setattr(auto_detection_module, "estimate_keypoints_from_image", fake_estimate)
    try:
        save_settings({"dataset_root": str(workspace)})

        res = client.post("/api/annotation/auto-detect-image", json={"filename": "case.png", "use_enhanced": False})

        assert res.status_code == 200
        payload = res.json()
        assert seen_sizes == [(60, 50)]
        assert payload["auto_detect"]["scan_transform_used"]["output_width"] == 60
        assert payload["auto_detect"]["roi_crop_used"] is None
        saved = load_annotation("case.png", workspace)
        assert saved is not None
        point = saved.keypoints["left_acetabular_outer"]
        assert point.x == 10
        assert point.y == 5
        assert saved.auto_initialization["image_preprocess"] == "scan_like+original"
    finally:
        save_settings(old_settings)


def test_list_recovers_from_corrupt_manifest_without_internal_server_error(tmp_path):
    old_settings = load_settings()
    client = TestClient(app)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "manifest.json").write_text('{"images": [', encoding="utf-8")
    try:
        save_settings({"dataset_root": str(workspace)})

        res = client.get("/api/annotation/list")

        assert res.status_code == 200
        assert res.json()["images"] == []
        assert not (workspace / "manifest.json").exists()
        assert list(workspace.glob("manifest.corrupt-*.json"))
    finally:
        save_settings(old_settings)
