from __future__ import annotations

import os
import sys
from pathlib import Path


APP_NAME = "Hip22AnnotationTool"
SOURCE_ROOT = Path(__file__).resolve().parents[1]


def is_packaged() -> bool:
    return bool(getattr(sys, "frozen", False))


def resource_root() -> Path:
    if hasattr(sys, "_MEIPASS"):
        return Path(sys._MEIPASS)  # type: ignore[attr-defined]
    return SOURCE_ROOT


def resource_path(*parts: str) -> Path:
    return resource_root().joinpath(*parts)


def user_data_dir() -> Path:
    configured = os.environ.get("HIP22_USER_DATA_DIR", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()

    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA")
        if base:
            return Path(base) / APP_NAME
        return Path.home() / "AppData" / "Local" / APP_NAME

    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / APP_NAME

    base = os.environ.get("XDG_DATA_HOME")
    if base:
        return Path(base).expanduser() / "hip-22-annotation-tool"
    return Path.home() / ".local" / "share" / "hip-22-annotation-tool"


def runtime_root() -> Path:
    configured = os.environ.get("HIP22_RUNTIME_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return user_data_dir() if is_packaged() else SOURCE_ROOT


def default_dataset_root() -> Path:
    configured = os.environ.get("HIP22_DATASET_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return runtime_root() / "workspace" if is_packaged() else SOURCE_ROOT
