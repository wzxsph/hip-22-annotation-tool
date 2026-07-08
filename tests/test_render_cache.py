import numpy as np
from PIL import Image

from annotation_tool.image_io import read_supported_image
from annotation_tool.image_processing import enhance_xray_image
from annotation_tool import render_cache


def test_cached_rendered_png_reuses_cache_until_source_changes(monkeypatch, tmp_path):
    source = tmp_path / "case.png"
    Image.new("RGB", (20, 10), color=(10, 20, 30)).save(source)
    monkeypatch.setattr(render_cache, "user_data_dir", lambda: tmp_path / "user-data")

    first = render_cache.cached_rendered_png(source, enhanced=True)
    second = render_cache.cached_rendered_png(source, enhanced=True)

    assert first == second
    assert first.exists()

    Image.new("RGB", (21, 10), color=(200, 210, 220)).save(source)
    third = render_cache.cached_rendered_png(source, enhanced=True)

    assert third != first
    assert third.exists()


def test_read_supported_image_normalizes_16bit_tiff(tmp_path):
    source = tmp_path / "case.tif"
    pixels = np.linspace(1000, 4000, 100 * 80, dtype=np.uint16).reshape(80, 100)
    Image.fromarray(pixels).save(source)

    image, metadata = read_supported_image(source)
    data = np.asarray(image)

    assert metadata["source_format"] == "tif"
    assert image.mode == "RGB"
    assert data.min() == 0
    assert data.max() == 255
    assert len(np.unique(data.reshape(-1, 3), axis=0)) > 16


def test_cached_rendered_png_enhances_16bit_tiff(monkeypatch, tmp_path):
    source = tmp_path / "case.tif"
    pixels = np.linspace(1000, 4000, 100 * 80, dtype=np.uint16).reshape(80, 100)
    Image.fromarray(pixels).save(source)
    monkeypatch.setattr(render_cache, "user_data_dir", lambda: tmp_path / "user-data")

    enhanced = render_cache.cached_rendered_png(source, enhanced=True)
    data = np.asarray(Image.open(enhanced))

    assert data.min() == 0
    assert data.max() == 255
    assert len(np.unique(data.reshape(-1, 3), axis=0)) > 16


def test_enhance_xray_image_does_not_lift_already_bright_image():
    pixels = np.linspace(180, 245, 80 * 100, dtype=np.uint8).reshape(80, 100)
    source = Image.fromarray(pixels, mode="L").convert("RGB")

    enhanced = enhance_xray_image(source)
    source_mean = float(np.asarray(source.convert("L"), dtype=np.float32).mean())
    enhanced_mean = float(np.asarray(enhanced.convert("L"), dtype=np.float32).mean())

    assert enhanced.mode == "RGB"
    assert enhanced.size == source.size
    assert enhanced_mean <= source_mean + 2.0
