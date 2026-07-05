from PIL import Image

from annotation_tool import render_cache


def test_cached_rendered_png_reuses_cache_until_source_changes(monkeypatch, tmp_path):
    source = tmp_path / "case.png"
    Image.new("RGB", (20, 10), color=(10, 20, 30)).save(source)
    monkeypatch.setattr(render_cache, "user_data_dir", lambda: tmp_path / "user-data")

    first = render_cache.cached_rendered_png(source, enhanced=True)
    second = render_cache.cached_rendered_png(source, enhanced=True)

    assert first == second
    assert first.exists()

    Image.new("RGB", (20, 10), color=(200, 210, 220)).save(source)
    third = render_cache.cached_rendered_png(source, enhanced=True)

    assert third != first
    assert third.exists()
