from __future__ import annotations

import hashlib
from pathlib import Path

from .image_io import read_supported_image
from .image_processing import enhance_xray_image
from .paths import user_data_dir


RENDER_CACHE_VERSION = "v1"


def rendered_cache_key(source_path: Path, *, enhanced: bool) -> str:
    stat = source_path.stat()
    payload = "|".join(
        [
            RENDER_CACHE_VERSION,
            str(source_path.expanduser().resolve()),
            str(stat.st_mtime_ns),
            str(stat.st_size),
            "enhanced" if enhanced else "display",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def rendered_png_cache_path(source_path: Path, *, enhanced: bool) -> Path:
    return user_data_dir() / "cache" / "rendered-images" / f"{rendered_cache_key(source_path, enhanced=enhanced)}.png"


def cached_rendered_png(source_path: Path, *, enhanced: bool) -> Path:
    cache_path = rendered_png_cache_path(source_path, enhanced=enhanced)
    if cache_path.exists():
        return cache_path

    image, _metadata = read_supported_image(source_path)
    if enhanced:
        image = enhance_xray_image(image)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp.png")
    image.save(tmp_path, format="PNG")
    tmp_path.replace(cache_path)
    return cache_path
