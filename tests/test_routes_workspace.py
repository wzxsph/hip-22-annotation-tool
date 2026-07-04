from pathlib import Path

from fastapi.testclient import TestClient
from PIL import Image

from annotation_tool import routes as routes_module
from annotation_tool.heuristics import AutoAnnotationResult
from annotation_tool.schema import create_blank_annotation, make_keypoint
from annotation_tool.server import app
from annotation_tool.storage import load_annotation, load_settings, save_annotation, save_settings
from annotation_tool.yolo_export import annotation_to_yolo_text


def _write_image(path: Path) -> None:
    Image.new("RGB", (80, 60), color="black").save(path)


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
        assert report["unsupported_files"] == ["scan.dcm"]
        assert report["messy_names"][0]["filename"] == "case one.png"
        assert report["warnings"]
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

    monkeypatch.setattr(routes_module, "estimate_keypoints_from_image", fake_estimate)
    try:
        save_settings({"dataset_root": str(workspace)})

        res = client.post("/api/annotation/auto-detect-image", json={"filename": "case.png"})

        assert res.status_code == 200
        payload = res.json()
        assert payload["auto_detect"]["applied"] is True
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

    monkeypatch.setattr(routes_module, "estimate_keypoints_from_image", fake_estimate)
    try:
        save_settings({"dataset_root": str(workspace)})

        res = client.post("/api/annotation/auto-detect-image", json={"filename": "case.png"})

        assert res.status_code == 200
        payload = res.json()
        assert payload["auto_detect"]["applied"] is False
        saved = load_annotation("case.png", workspace)
        assert saved is not None
        assert saved.keypoints["left_acetabular_outer"].x == 40
        assert saved.auto_initialization["source"] == "model-no-result"
        assert saved.auto_initialization["warnings"] == ["no visible keypoints"]
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
