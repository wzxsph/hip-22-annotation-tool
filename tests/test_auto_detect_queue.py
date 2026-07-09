import inspect
from pathlib import Path

from PIL import Image

from annotation_tool import auto_detection as auto_detection_module
from annotation_tool.auto_detect_queue import AutoDetectItem, _process_item, replace_auto_detect_queue
from annotation_tool.heuristics import AutoAnnotationResult
from annotation_tool.schema import create_blank_annotation, make_keypoint
from annotation_tool.storage import load_annotation


def _write_image(path: Path) -> None:
    Image.new("RGB", (20, 20), color="black").save(path)


def test_queue_status_defaults_to_original_first_for_imported_images(tmp_path):
    signature = inspect.signature(replace_auto_detect_queue)

    assert signature.parameters["use_enhanced"].default is False


def test_background_auto_detect_tries_enhanced_after_original_no_result(monkeypatch, tmp_path):
    image_path = tmp_path / "case.png"
    _write_image(image_path)
    seen_pixels = []

    def fake_enhance(image):
        return Image.new("RGB", image.size, color="white")

    def fake_estimate(image, *, min_visible_keypoints=None, include_partial=False):
        seen_pixels.append(image.getpixel((0, 0)))
        blank = create_blank_annotation("case.png", image.width, image.height)
        return AutoAnnotationResult(
            keypoints=blank.keypoints,
            warnings=[],
            model_available=True,
            source="test-model",
            attempts=[{"strategy": "test", "visible_count": 0, "success": False}],
            strategy="test",
        )

    monkeypatch.setattr(auto_detection_module, "enhance_xray_image", fake_enhance)
    monkeypatch.setattr(auto_detection_module, "estimate_keypoints_from_image", fake_estimate)

    outcome = _process_item(AutoDetectItem(root=tmp_path, image_path=image_path, use_enhanced=True))

    assert outcome == "done"
    assert seen_pixels == [(0, 0, 0), (255, 255, 255)]
    saved = load_annotation("case.png", tmp_path)
    assert saved is not None
    assert saved.auto_initialization["image_preprocess"] == "original+hip_demo_enhanced"
    assert len(saved.auto_initialization["preprocess_attempts"]) == 2
    assert saved.auto_initialization["template_fallback"]["enabled"] is False
    assert saved.auto_initialization["template_fallback"]["filled_count"] == 0
    assert all(not point.visible for point in saved.keypoints.values())


def test_background_auto_detect_uses_original_when_both_sides_detected(monkeypatch, tmp_path):
    image_path = tmp_path / "case.png"
    _write_image(image_path)
    seen_pixels = []

    def fake_enhance(image):
        raise AssertionError("enhanced preprocessing should not run when original detects both sides")

    def fake_estimate(image, *, min_visible_keypoints=None, include_partial=False):
        seen_pixels.append(image.getpixel((0, 0)))
        annotation = create_blank_annotation("case.png", image.width, image.height)
        annotation.keypoints["left_acetabular_outer"] = make_keypoint(
            "left", "acetabular_outer", 4, 5, source="pose11_side", confidence=0.8
        )
        annotation.keypoints["right_acetabular_outer"] = make_keypoint(
            "right", "acetabular_outer", 14, 5, source="pose11_side", confidence=0.8
        )
        return AutoAnnotationResult(
            keypoints=annotation.keypoints,
            warnings=[],
            model_available=True,
            source="test-model",
            attempts=[{"strategy": "test", "visible_count": 2, "success": True}],
            strategy="test",
        )

    monkeypatch.setattr(auto_detection_module, "enhance_xray_image", fake_enhance)
    monkeypatch.setattr(auto_detection_module, "estimate_keypoints_from_image", fake_estimate)

    outcome = _process_item(AutoDetectItem(root=tmp_path, image_path=image_path))

    assert outcome == "done"
    assert seen_pixels == [(0, 0, 0)]
    saved = load_annotation("case.png", tmp_path)
    assert saved is not None
    assert saved.auto_initialization["image_preprocess"] == "original"
    assert len(saved.auto_initialization["preprocess_attempts"]) == 1


def test_background_auto_detect_accepts_partial_enhanced_result(monkeypatch, tmp_path):
    image_path = tmp_path / "case.png"
    _write_image(image_path)
    calls = []

    def fake_enhance(image):
        return Image.new("RGB", image.size, color="white")

    def fake_estimate(image, *, min_visible_keypoints=None, include_partial=False):
        calls.append(
            {
                "pixel": image.getpixel((0, 0)),
                "min_visible_keypoints": min_visible_keypoints,
                "include_partial": include_partial,
            }
        )
        annotation = create_blank_annotation("case.png", image.width, image.height)
        annotation.keypoints["left_acetabular_outer"] = make_keypoint(
            "left", "acetabular_outer", 4, 5, source="pose11_side", confidence=0.8
        )
        if len(calls) > 1:
            annotation.keypoints["right_acetabular_outer"] = make_keypoint(
                "right", "acetabular_outer", 14, 5, source="pose11_side", confidence=0.8
            )
        return AutoAnnotationResult(
            keypoints=annotation.keypoints,
            warnings=[],
            model_available=True,
            source="yolo11n-best-partial",
            attempts=[{"strategy": "test", "visible_count": 2, "success": True}],
            strategy="partial",
        )

    monkeypatch.setattr(auto_detection_module, "enhance_xray_image", fake_enhance)
    monkeypatch.setattr(auto_detection_module, "estimate_keypoints_from_image", fake_estimate)

    outcome = _process_item(AutoDetectItem(root=tmp_path, image_path=image_path))

    assert outcome == "done"
    assert calls == [
        {"pixel": (0, 0, 0), "min_visible_keypoints": 1, "include_partial": True},
        {"pixel": (255, 255, 255), "min_visible_keypoints": 1, "include_partial": True},
    ]
    saved = load_annotation("case.png", tmp_path)
    assert saved is not None
    assert saved.auto_initialization["image_preprocess"] == "hip_demo_enhanced"
    assert saved.keypoints["left_acetabular_outer"].visible is True
    assert saved.keypoints["right_acetabular_outer"].visible is True


def test_background_auto_detect_keeps_single_side_enhanced_result(monkeypatch, tmp_path):
    image_path = tmp_path / "case.png"
    _write_image(image_path)
    calls = []

    def fake_enhance(image):
        return Image.new("RGB", image.size, color="white")

    def fake_estimate(image, *, min_visible_keypoints=None, include_partial=False):
        calls.append(image.getpixel((0, 0)))
        annotation = create_blank_annotation("case.png", image.width, image.height)
        annotation.keypoints["right_acetabular_outer"] = make_keypoint(
            "right", "acetabular_outer", 14, 5, source="pose11_side", confidence=0.8
        )
        return AutoAnnotationResult(
            keypoints=annotation.keypoints,
            warnings=[],
            model_available=True,
            source="yolo11n-best-side11",
            attempts=[{"strategy": "test", "visible_count": 1, "success": True}],
            strategy="test",
        )

    monkeypatch.setattr(auto_detection_module, "enhance_xray_image", fake_enhance)
    monkeypatch.setattr(auto_detection_module, "estimate_keypoints_from_image", fake_estimate)

    outcome = _process_item(AutoDetectItem(root=tmp_path, image_path=image_path))

    assert outcome == "done"
    assert calls == [(0, 0, 0), (255, 255, 255)]
    saved = load_annotation("case.png", tmp_path)
    assert saved is not None
    assert saved.auto_initialization["image_preprocess"] == "hip_demo_enhanced"
    assert saved.auto_initialization["model_visible_count"] == 1
    assert saved.keypoints["left_acetabular_outer"].visible is False
    assert saved.keypoints["right_acetabular_outer"].visible is True


def test_background_auto_detect_infers_12_after_detection(monkeypatch, tmp_path):
    image_path = tmp_path / "case.png"
    _write_image(image_path)

    def fake_estimate(image, *, min_visible_keypoints=None, include_partial=False):
        annotation = create_blank_annotation("case.png", image.width, image.height)
        annotation.keypoints["left_femoral_head_center"] = make_keypoint(
            "left", "femoral_head_center", 10, 12, source="pose11_side", confidence=0.8
        )
        annotation.keypoints["left_femoral_neck_axis_center"] = make_keypoint(
            "left", "femoral_neck_axis_center", 18, 20, source="pose11_side", confidence=0.7
        )
        annotation.keypoints["right_acetabular_outer"] = make_keypoint(
            "right", "acetabular_outer", 14, 10, source="pose11_side", confidence=0.8
        )
        return AutoAnnotationResult(
            keypoints=annotation.keypoints,
            warnings=[],
            model_available=True,
            source="test-model",
            attempts=[{"strategy": "test", "visible_count": 2, "success": True}],
            strategy="test",
        )

    monkeypatch.setattr(auto_detection_module, "estimate_keypoints_from_image", fake_estimate)

    outcome = _process_item(AutoDetectItem(root=tmp_path, image_path=image_path, use_enhanced=False))

    assert outcome == "done"
    saved = load_annotation("case.png", tmp_path)
    assert saved is not None
    point = saved.keypoints["left_femoral_neck_axis_proximal"]
    assert point.visible is True
    assert point.x == 14
    assert point.y == 16
    assert point.source == "estimated"
