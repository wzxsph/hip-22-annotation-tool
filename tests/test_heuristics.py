from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image

from annotation_tool import heuristics as heuristics_module
from annotation_tool.heuristics import _preferred_device, estimate_keypoints_from_image, model_path_from_env
from annotation_tool.schema import LANDMARK_DEFS, SIDES, key_for, make_keypoint


def test_model_path_can_be_overridden(monkeypatch, tmp_path):
    weights = tmp_path / "custom.pt"
    monkeypatch.setenv("HIP22_MODEL_PATH", str(weights))

    assert model_path_from_env() == weights.resolve()


def test_device_prefers_cuda_when_available(monkeypatch):
    monkeypatch.delenv("HIP22_DEVICE", raising=False)
    monkeypatch.delenv("HIP22_MODEL_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))

    assert _preferred_device() == "cuda:0"


def test_device_can_be_overridden(monkeypatch):
    monkeypatch.setenv("HIP22_DEVICE", "cpu")

    assert _preferred_device() == "cpu"


def test_missing_model_returns_blank_template_without_crashing(monkeypatch, tmp_path):
    monkeypatch.setenv("HIP22_MODEL_PATH", str(tmp_path / "missing.pt"))
    image = Image.new("RGB", (80, 60), color="black")

    result = estimate_keypoints_from_image(image)

    assert not result.model_available
    assert len(result.keypoints) == 24
    assert any("Model unavailable" in item for item in result.warnings)
    for key, point in result.keypoints.items():
        assert not point.visible, key
        assert point.x is None
        assert point.y is None


def _decoded_with_visible_points(count: int):
    decoded = heuristics_module._empty_template()
    used = 0
    for side in SIDES:
        for landmark in LANDMARK_DEFS:
            if used >= count:
                return decoded
            decoded[key_for(side, landmark.name)] = make_keypoint(
                side,
                landmark.name,
                10 + used,
                20 + used,
                source="pose11_side",
                confidence=0.8,
            )
            used += 1
    return decoded


def test_auto_detect_retries_with_lower_threshold_after_empty_result(monkeypatch, tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fake")
    calls = []

    monkeypatch.setenv("HIP22_MODEL_PATH", str(weights))
    monkeypatch.setenv("HIP22_MIN_VISIBLE", "2")
    monkeypatch.delenv("HIP22_AUTO_FALLBACK", raising=False)
    monkeypatch.setattr(heuristics_module, "_load_model", lambda _: object())

    def fake_run_model(model, image, *, imgsz, conf):
        calls.append((imgsz, conf))
        return [SimpleNamespace(index=len(calls))]

    def fake_decode(result):
        if result.index == 1:
            return _decoded_with_visible_points(0)
        return _decoded_with_visible_points(2)

    monkeypatch.setattr(heuristics_module, "_run_model", fake_run_model)
    monkeypatch.setattr(heuristics_module, "decode_side11_result", fake_decode)

    result = estimate_keypoints_from_image(Image.new("RGB", (80, 60), color="black"))

    assert result.source == "yolo11n-best-side11"
    assert result.strategy == "large_1024_low_conf"
    assert result.visible_count == 2
    assert calls == [(800, 0.25), (1024, 0.15)]
    assert result.attempts[0]["success"] is False
    assert result.attempts[1]["success"] is True


def test_auto_detect_no_result_keeps_blank_template(monkeypatch, tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fake")

    monkeypatch.setenv("HIP22_MODEL_PATH", str(weights))
    monkeypatch.setenv("HIP22_MIN_VISIBLE", "2")
    monkeypatch.setattr(heuristics_module, "_load_model", lambda _: object())
    monkeypatch.setattr(heuristics_module, "_run_model", lambda *args, **kwargs: [object()])
    monkeypatch.setattr(heuristics_module, "decode_side11_result", lambda _: _decoded_with_visible_points(0))

    result = estimate_keypoints_from_image(Image.new("RGB", (80, 60), color="black"))

    assert result.model_available
    assert result.source == "model-no-result"
    assert result.visible_count == 0
    assert len(result.attempts) == 5
    assert any("no visible" in warning for warning in result.warnings)


def test_auto_detect_fallback_can_be_disabled_and_tuned(monkeypatch, tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"fake")
    calls = []

    monkeypatch.setenv("HIP22_MODEL_PATH", str(weights))
    monkeypatch.setenv("HIP22_AUTO_FALLBACK", "0")
    monkeypatch.setenv("HIP22_IMGSZ", "640")
    monkeypatch.setenv("HIP22_CONF", "0.33")
    monkeypatch.setenv("HIP22_MIN_VISIBLE", "2")
    monkeypatch.setattr(heuristics_module, "_load_model", lambda _: object())

    def fake_run_model(model, image, *, imgsz, conf):
        calls.append((imgsz, conf))
        return [object()]

    monkeypatch.setattr(heuristics_module, "_run_model", fake_run_model)
    monkeypatch.setattr(heuristics_module, "decode_side11_result", lambda _: _decoded_with_visible_points(0))

    result = estimate_keypoints_from_image(Image.new("RGB", (80, 60), color="black"))

    assert result.source == "model-no-result"
    assert calls == [(640, 0.33)]
    assert len(result.attempts) == 1
