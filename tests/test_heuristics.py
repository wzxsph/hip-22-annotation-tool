from pathlib import Path
import sys
from types import SimpleNamespace

from PIL import Image

from annotation_tool.heuristics import _preferred_device, estimate_keypoints_from_image, model_path_from_env


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
    assert len(result.keypoints) == 22
    assert any("Model unavailable" in item for item in result.warnings)
    for key, point in result.keypoints.items():
        assert not point.visible, key
        assert point.x is None
        assert point.y is None
