import inspect
from pathlib import Path

from PIL import Image

from annotation_tool import auto_detection as auto_detection_module
from annotation_tool.auto_detect_queue import AutoDetectItem, _process_item, replace_auto_detect_queue
from annotation_tool.heuristics import AutoAnnotationResult
from annotation_tool.schema import create_blank_annotation
from annotation_tool.storage import load_annotation


def _write_image(path: Path) -> None:
    Image.new("RGB", (20, 20), color="black").save(path)


def test_queue_status_defaults_to_enhanced_for_imported_images(tmp_path):
    signature = inspect.signature(replace_auto_detect_queue)

    assert signature.parameters["use_enhanced"].default is True


def test_background_auto_detect_uses_enhanced_preprocess(monkeypatch, tmp_path):
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
    assert seen_pixels == [(255, 255, 255)]
    saved = load_annotation("case.png", tmp_path)
    assert saved is not None
    assert saved.auto_initialization["image_preprocess"] == "hip_demo_enhanced"
    assert saved.auto_initialization["template_fallback"]["enabled"] is False
    assert saved.auto_initialization["template_fallback"]["filled_count"] == 0
    assert all(not point.visible for point in saved.keypoints.values())
