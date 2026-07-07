from __future__ import annotations

import hashlib
from pathlib import Path

from .image_io import read_supported_image
from .image_processing import enhance_xray_image
from .paths import user_data_dir


RENDER_CACHE_VERSION = "v3"
THUMB_CACHE_VERSION = "v2"
THUMB_MAX_SIZE = 220


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

    image, _metadata = read_supported_image(source_path, normalize=not enhanced)
    if enhanced:
        image = enhance_xray_image(image)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp.png")
    image.save(tmp_path, format="PNG")
    tmp_path.replace(cache_path)
    return cache_path


def thumbnail_cache_key(source_path: Path, *, max_size: int = THUMB_MAX_SIZE) -> str:
    stat = source_path.stat()
    payload = "|".join(
        [
            THUMB_CACHE_VERSION,
            str(source_path.expanduser().resolve()),
            str(stat.st_mtime_ns),
            str(stat.st_size),
            f"thumb-{max_size}",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def thumbnail_cache_path(source_path: Path, *, max_size: int = THUMB_MAX_SIZE) -> Path:
    return user_data_dir() / "cache" / "thumbnails" / f"{thumbnail_cache_key(source_path, max_size=max_size)}.jpg"


def cached_thumbnail_jpeg(source_path: Path, *, max_size: int = THUMB_MAX_SIZE) -> Path:
    cache_path = thumbnail_cache_path(source_path, max_size=max_size)
    if cache_path.exists():
        return cache_path

    image, _metadata = read_supported_image(source_path)
    image.thumbnail((max_size, max_size))
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = cache_path.with_suffix(".tmp.jpg")
    image.convert("RGB").save(tmp_path, format="JPEG", quality=78, optimize=True)
    tmp_path.replace(cache_path)
    return cache_path
