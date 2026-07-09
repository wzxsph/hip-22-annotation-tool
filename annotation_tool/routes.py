from __future__ import annotations

import filecmp
import io
import shutil
import time
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel

from .auto_detect_queue import auto_detect_status, replace_auto_detect_queue
from .auto_detection import (
    AUTO_DETECT_POLICY_LABEL,
    clear_non_manual_keypoints,
    estimate_keypoints_original_then_enhanced,
    should_retry_auto_detection,
)
from .completion import PROGRESS_STATUS_VERSION, annotation_progress
from .dicom_utils import is_dicom_path
from .image_io import is_supported_image_path, read_supported_image
from .measurements import compute_measurements
from .progress_report import build_progress_payload_from_manifest
from .render_cache import RENDER_CACHE_VERSION, cached_rendered_png, cached_thumbnail_jpeg, rendered_png_cache_path, thumbnail_cache_path
from .schema import LANDMARK_DEFS, Annotation, Manifest, create_blank_annotation, ensure_keypoint_template, model_to_dict, normalize_split
from .storage import (
    annotation_path,
    annotations_dir,
    current_root,
    data_yaml_path,
    ensure_dataset_layout,
    find_annotation_path,
    find_image_path,
    annotation_path_candidates,
    image_path_for,
    load_annotation,
    load_annotation_from_path,
    load_annotation_from_yolo_label,
    load_manifest,
    load_settings,
    manifest_image_for_path,
    save_annotation,
    save_manifest,
    save_settings,
    upsert_manifest_image,
    label_path,
)
from .template_points import visible_keypoint_count
from .yolo_export import annotation_to_yolo_text, data_yaml_text

router = APIRouter(prefix="/api/annotation", tags=["annotation"])


class OpenFolderRequest(BaseModel):
    folder_path: str = ""
    split: str = "train"


class SettingsRequest(BaseModel):
    dataset_root: Optional[str] = None
    auto_detect: Optional[bool] = None
    autosave: Optional[bool] = None
    annotator: Optional[str] = None
    display_brightness: Optional[int] = None
    display_contrast: Optional[int] = None


class SelectFolderRequest(BaseModel):
    purpose: str = "import"


class AutoDetectImageRequest(BaseModel):
    filename: str
    preserve_manual: bool = True
    include_partial: bool = True
    use_enhanced: bool = False
    use_roi: bool = True
    use_scan: bool = True


class BatchDeleteImagesRequest(BaseModel):
    filenames: list[str]


class RestoreTrashRequest(BaseModel):
    trash_paths: list[str]


@dataclass(frozen=True)
class LegacyAnnotationRecord:
    path: Path
    annotation: Annotation
    image_filename: str
    split: str


def _is_image_file(path: Path) -> bool:
    return is_supported_image_path(path)


def _safe_filename(filename: str | None) -> str:
    name = Path(filename or "upload.png").name
    if not name or name in {".", ".."}:
        return f"upload_{int(time.time())}.png"
    return name


def _read_image_record(path: Path) -> tuple[Image.Image, dict]:
    try:
        return read_supported_image(path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read image: {path.name}") from exc


def _apply_image_metadata(annotation: Annotation, metadata: dict) -> Annotation:
    for key in (
        "source_format",
        "pixel_spacing_mm",
        "pixel_spacing_row_mm",
        "pixel_spacing_col_mm",
        "pixel_spacing_source",
        "dicom_warnings",
    ):
        if key in metadata:
            setattr(annotation.image, key, metadata[key])
    return annotation


def _image_url_for_path(image_path: Path) -> str:
    stat = image_path.stat()
    filename = quote(image_path.name)
    return f"/api/annotation/image/{filename}?source_v={stat.st_mtime_ns}-{stat.st_size}&render_v={RENDER_CACHE_VERSION}"


def _path_for_response(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def _trash_destination(path: Path, trash_dir: Path) -> Path:
    destination = trash_dir / path.name
    if not destination.exists():
        return destination

    stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    base = trash_dir / f"{path.stem}_deleted_{stamp}{path.suffix}"
    destination = base
    index = 2
    while destination.exists():
        destination = trash_dir / f"{path.stem}_deleted_{stamp}_{index}{path.suffix}"
        index += 1
    return destination


def _move_to_trash(path: Path, root: Path, trash_dir: Path) -> dict[str, str]:
    destination = _trash_destination(path, trash_dir)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(path), str(destination))
    return {
        "source": _path_for_response(path, root),
        "trash_path": _path_for_response(destination, root),
    }


def _delete_image_files(filename: str, root: Path) -> dict:
    image_path = find_image_path(filename, root)
    if image_path is None or not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")

    trash_dir = image_path.parent / "trash"
    deleted: list[str] = []
    trashed: list[dict[str, str]] = []
    for cache_path in (
        rendered_png_cache_path(image_path, enhanced=False),
        rendered_png_cache_path(image_path, enhanced=True),
        thumbnail_cache_path(image_path),
    ):
        if cache_path.exists():
            cache_path.unlink()
            deleted.append(str(cache_path))

    candidates = {
        image_path,
        annotation_path(filename, root),
        image_path.with_suffix(".json"),
        label_path(filename, root, _split_from_image_path(image_path)),
        image_path.with_suffix(".txt"),
        root / f"{image_path.stem}.txt",
    }
    candidates.update(annotation_path_candidates(filename, root, image_path))

    for candidate in sorted(candidates, key=lambda item: str(item)):
        if candidate.exists() and candidate.is_file():
            trashed.append(_move_to_trash(candidate, root, trash_dir))

    manifest = load_manifest(root)
    image_id = image_path.stem
    manifest.images = [
        item
        for item in manifest.images
        if item.id != image_id and Path(item.image_path).name != filename
    ]
    save_manifest(manifest, root)

    return {
        "filename": filename,
        "deleted": deleted,
        "trashed": trashed,
        "trash_dir": _path_for_response(trash_dir, root),
    }


def _safe_workspace_path(path_text: str, root: Path) -> Path:
    root_resolved = root.resolve()
    path = Path(path_text)
    resolved = path.expanduser().resolve() if path.is_absolute() else (root / path).resolve()
    if not resolved.is_relative_to(root_resolved):
        raise HTTPException(status_code=400, detail="Path is outside the current workspace.")
    return resolved


def _trash_sidecars_for_image(trash_image_path: Path) -> list[Path]:
    return [
        candidate
        for candidate in (
            trash_image_path.with_suffix(".json"),
            trash_image_path.with_suffix(".txt"),
        )
        if candidate.exists() and candidate.is_file()
    ]


def _restore_trash_image(trash_path_text: str, root: Path) -> dict:
    trash_image_path = _safe_workspace_path(trash_path_text, root)
    if not trash_image_path.exists() or not trash_image_path.is_file():
        raise HTTPException(status_code=404, detail="Trash image not found.")
    if trash_image_path.parent.name != "trash" or not _is_image_file(trash_image_path):
        raise HTTPException(status_code=400, detail="Path is not a restorable trash image.")

    restore_dir = trash_image_path.parent.parent
    files_to_restore = [trash_image_path, *_trash_sidecars_for_image(trash_image_path)]
    conflicts = [path for path in files_to_restore if (restore_dir / path.name).exists()]
    if conflicts:
        return {
            "trash_path": _path_for_response(trash_image_path, root),
            "status": "conflict",
            "detail": "Restore target already exists.",
            "conflicts": [_path_for_response(restore_dir / path.name, root) for path in conflicts],
        }

    restored: list[dict[str, str]] = []
    restore_dir.mkdir(parents=True, exist_ok=True)
    for source in files_to_restore:
        destination = restore_dir / source.name
        shutil.move(str(source), str(destination))
        restored.append(
            {
                "trash_path": _path_for_response(source, root),
                "restored_path": _path_for_response(destination, root),
            }
        )

    image_path = restore_dir / trash_image_path.name
    annotation = _existing_annotation_for_image(
        image_path,
        root=root,
        split=_split_from_image_path(image_path),
        persist_imported=True,
    )
    upsert_manifest_image(image_path, annotation=annotation, root=root, split=_split_from_image_path(image_path))
    return {
        "trash_path": _path_for_response(trash_image_path, root),
        "status": "restored",
        "filename": image_path.name,
        "restored": restored,
    }


def _split_from_image_path(path: Path) -> str:
    if path.parent.name in {"train", "val"}:
        return path.parent.name
    return "train"


def _needs_browser_rendering(path: Path) -> bool:
    return is_dicom_path(path) or path.suffix.lower() in {".tif", ".tiff"}


def _scan_workspace_images(root: Path) -> list[Path]:
    seen: set[Path] = set()
    images: list[Path] = []
    search_dirs = [root, root / "images" / "train", root / "images" / "val"]
    for directory in search_dirs:
        if not directory.exists() or not directory.is_dir():
            continue
        for path in sorted(directory.iterdir()):
            if not path.is_file() or not _is_image_file(path):
                continue
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            images.append(resolved)
    return images


def _readability_check(paths: list[Path], *, full_check_limit: int = 100) -> tuple[list[Path], bool]:
    if len(paths) > full_check_limit:
        return [], False
    unreadable: list[Path] = []
    for path in paths:
        try:
            read_supported_image(path)
        except Exception:
            unreadable.append(path)
    return unreadable, True


def _annotation_json_candidates(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for directory in (root / "annotations", root):
        if not directory.exists() or not directory.is_dir():
            continue
        for path in sorted(directory.glob("*.json")):
            if path.name in {"manifest.json", "tool-settings.json"}:
                continue
            candidates.append(path)
    return candidates


def _scan_legacy_annotation_records(root: Path) -> list[LegacyAnnotationRecord]:
    records_by_filename: dict[str, LegacyAnnotationRecord] = {}
    for path in _annotation_json_candidates(root):
        try:
            annotation = ensure_keypoint_template(load_annotation_from_path(path))
        except Exception:
            continue
        image_filename = Path(annotation.image.filename).name
        if not image_filename:
            continue
        annotation.image.filename = image_filename
        split = normalize_split(getattr(annotation.image, "split", "train"))
        annotation.image.split = split
        key = image_filename.lower()
        if key in records_by_filename:
            continue
        records_by_filename[key] = LegacyAnnotationRecord(
            path=path,
            annotation=annotation,
            image_filename=image_filename,
            split=split,
        )
    return list(records_by_filename.values())


def _candidate_image_dirs_for_legacy(root: Path) -> list[Path]:
    seen: set[Path] = set()
    directories: list[Path] = []

    def add(path: Path) -> None:
        try:
            resolved = path.expanduser().resolve()
        except OSError:
            return
        if resolved in seen or not resolved.exists() or not resolved.is_dir():
            return
        seen.add(resolved)
        directories.append(resolved)

    for base in (root, root / "images", root / "images" / "train", root / "images" / "val"):
        add(base)

    parent = root.parent
    if parent.exists() and parent.is_dir():
        for sibling in sorted(parent.iterdir()):
            if not sibling.is_dir() or sibling.resolve() == root.resolve():
                continue
            for base in (sibling, sibling / "images", sibling / "images" / "train", sibling / "images" / "val"):
                add(base)

    project_root = root.parent.parent
    for base in (
        project_root / "annotation-tool" / "images",
        project_root / "annotation-tool" / "images" / "train",
        project_root / "annotation-tool" / "images" / "val",
    ):
        add(base)
    return directories


def _index_images_by_name(directories: list[Path]) -> tuple[dict[str, list[Path]], dict[str, list[Path]]]:
    by_name: dict[str, list[Path]] = {}
    by_stem: dict[str, list[Path]] = {}
    for directory in directories:
        for path in sorted(directory.iterdir()):
            if not path.is_file() or not _is_image_file(path):
                continue
            by_name.setdefault(path.name.lower(), []).append(path)
            by_stem.setdefault(path.stem.lower(), []).append(path)
    return by_name, by_stem


def _match_legacy_image(record: LegacyAnnotationRecord, by_name: dict[str, list[Path]], by_stem: dict[str, list[Path]]) -> Path | None:
    filename = Path(record.image_filename).name
    exact = by_name.get(filename.lower())
    if exact:
        return exact[0]
    stem = Path(filename).stem.lower()
    stem_matches = by_stem.get(stem)
    if stem_matches:
        return stem_matches[0]
    return None


def _refresh_stale_manifest_progress(root: Path, manifest: Manifest) -> Manifest:
    changed = False
    for item in manifest.images:
        filename = Path(item.image_path).name
        image_path = find_image_path(filename, root) or (root / item.image_path)
        annotation_file = find_annotation_path(filename, root, image_path)
        if annotation_file is None or not annotation_file.exists():
            continue
        annotation_mtime_ns = annotation_file.stat().st_mtime_ns
        if (
            item.status_detail
            and item.annotation_mtime_ns == annotation_mtime_ns
            and item.progress_status_version == PROGRESS_STATUS_VERSION
        ):
            continue
        try:
            annotation = load_annotation_from_path(annotation_file)
        except Exception:
            continue
        split = _split_from_image_path(image_path) if image_path.exists() else normalize_split(annotation.image.split)
        refreshed = manifest_image_for_path(image_path, annotation=annotation, root=root, split=split)
        item.status = refreshed.status
        item.keypoint_status = refreshed.keypoint_status
        item.shenton_status = refreshed.shenton_status
        item.status_detail = refreshed.status_detail
        item.keypoint_visible_count = refreshed.keypoint_visible_count
        item.keypoint_manual_count = refreshed.keypoint_manual_count
        item.shenton_complete_sides = refreshed.shenton_complete_sides
        item.shenton_started_sides = refreshed.shenton_started_sides
        item.annotation_mtime_ns = refreshed.annotation_mtime_ns
        item.progress_status_version = refreshed.progress_status_version
        item.annotator = refreshed.annotator
        item.completed_at = refreshed.completed_at
        changed = True
    if changed:
        save_manifest(manifest, root)
    return manifest


def _copy_legacy_image(source: Path, root: Path, split: str) -> tuple[Path | None, bool, str | None]:
    source_resolved = source.resolve()
    try:
        source_resolved.relative_to(root.resolve())
        return source_resolved, False, None
    except ValueError:
        pass

    target_dir = root / "images" / normalize_split(split)
    target = target_dir / source.name
    if target.exists():
        try:
            if target.resolve() == source_resolved or filecmp.cmp(source_resolved, target, shallow=False):
                return target.resolve(), False, None
        except OSError:
            pass
        return None, False, f"{source.name} 已存在但内容不同，已跳过。"

    target_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_resolved, target)
    return target.resolve(), True, None


def _materialize_legacy_only_workspace(root: Path, records: list[LegacyAnnotationRecord]) -> tuple[list[Path], dict]:
    directories = _candidate_image_dirs_for_legacy(root)
    by_name, by_stem = _index_images_by_name(directories)
    image_paths: list[Path] = []
    source_dirs: set[str] = set()
    copied = 0
    missing: list[str] = []
    conflicts: list[str] = []

    for record in records:
        source = _match_legacy_image(record, by_name, by_stem)
        if source is None:
            missing.append(record.image_filename)
            continue
        image_path, was_copied, conflict = _copy_legacy_image(source, root, record.split)
        if conflict:
            conflicts.append(conflict)
            continue
        if image_path is None:
            continue
        if was_copied:
            copied += 1
        source_dirs.add(_relative_path_text(source.parent, root))
        image_paths.append(image_path)

    warnings: list[str] = []
    if copied:
        warnings.append(f"已从旧标注引用的图片目录复制 {copied} 张图片。")
    if missing:
        warnings.append(f"有 {len(missing)} 个旧标注找不到对应图片。")
    if conflicts:
        warnings.append(f"有 {len(conflicts)} 张图片目标文件名冲突，已跳过。")
    return image_paths, {
        "legacy_annotations": len(records),
        "legacy_images_resolved": len(image_paths),
        "copied_external_images": copied,
        "missing_legacy_images": missing[:20],
        "conflicting_legacy_images": conflicts[:20],
        "external_image_dirs": sorted(source_dirs)[:20],
        "warnings": warnings,
    }


def _relative_path_text(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return str(path)


def _is_internal_artifact(path: Path, root: Path) -> bool:
    try:
        relative = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    parts = relative.parts
    if parts and parts[0] in {"annotations", "labels", "splits", "images"}:
        return True
    name = path.name
    return (
        name in {"manifest.json", "data.yaml", "tool-settings.json", "HIP22_status_report.html", "HIP22_status_report.csv"}
        or name == "HIP22_SUBMISSION_README.txt"
        or name.startswith("HIP22_STATUS_DONE_")
    )


def _limited_paths(paths: list[Path], root: Path, limit: int = 20) -> list[str]:
    return [_relative_path_text(path, root) for path in paths[:limit]]


def _import_folder_report(
    root: Path,
    selected_images: list[Path],
    *,
    unreadable_files: list[Path] | None = None,
    readability_checked_all: bool = True,
    legacy_report: dict | None = None,
) -> dict:
    selected = {path.resolve() for path in selected_images}
    supported_images: list[Path] = []
    unsupported_files: list[Path] = []
    unreadable_files = unreadable_files or []
    warnings: list[str] = []
    try:
        all_files = [path for path in root.rglob("*") if path.is_file()]
    except Exception as exc:
        return {
            "warnings": [f"无法完整扫描文件夹：{exc}"],
            "nested_images": [],
            "duplicate_names": [],
            "unsupported_files": [],
            "unreadable_files": [],
            "messy_names": [],
            "legacy_annotations": 0,
            "legacy_images_resolved": 0,
            "copied_external_images": 0,
            "missing_legacy_images": [],
            "conflicting_legacy_images": [],
            "external_image_dirs": [],
        }

    for path in all_files:
        if _is_internal_artifact(path, root):
            continue
        if _is_image_file(path):
            supported_images.append(path)
            continue
        if path.suffix.lower() not in {".txt", ".json", ".yaml", ".yml", ".csv", ".html"}:
            unsupported_files.append(path)

    nested_images = [path for path in supported_images if path.resolve() not in selected]
    by_name: dict[str, list[Path]] = {}
    for path in supported_images:
        by_name.setdefault(path.name.lower(), []).append(path)
    duplicate_names = [
        {"filename": paths[0].name, "paths": _limited_paths(paths, root)}
        for paths in by_name.values()
        if len(paths) > 1
    ][:20]
    messy_names = [
        {"filename": path.name, "reason": "文件名包含空格或过长，建议由项目团队统一整理"}
        for path in selected_images
        if any(char.isspace() for char in path.stem) or len(path.stem) > 80
    ][:20]

    if nested_images:
        warnings.append(f"发现 {len(nested_images)} 张图片在嵌套目录中，当前不会导入；请先整理到同一个文件夹。")
    if duplicate_names:
        warnings.append(f"发现 {len(duplicate_names)} 组重复文件名，建议整理后再标注，避免标签对应混乱。")
    if unsupported_files:
        warnings.append(f"发现 {len(unsupported_files)} 个非图片/不支持文件。")
    if unreadable_files:
        warnings.append(f"发现 {len(unreadable_files)} 张图片无法读取，请联系项目团队。")
    if not readability_checked_all:
        warnings.append("图片较多，导入时已跳过逐张读图检查；损坏图片会在打开或后台识别时提示。")
    if messy_names:
        warnings.append(f"发现 {len(messy_names)} 个文件名可能需要整理。")
    if legacy_report:
        warnings.extend(legacy_report.get("warnings", []))

    return {
        "warnings": warnings,
        "nested_images": _limited_paths(nested_images, root),
        "duplicate_names": duplicate_names,
        "unsupported_files": _limited_paths(unsupported_files, root),
        "unreadable_files": _limited_paths(unreadable_files, root),
        "messy_names": messy_names,
        "legacy_annotations": int((legacy_report or {}).get("legacy_annotations", 0) or 0),
        "legacy_images_resolved": int((legacy_report or {}).get("legacy_images_resolved", 0) or 0),
        "copied_external_images": int((legacy_report or {}).get("copied_external_images", 0) or 0),
        "missing_legacy_images": list((legacy_report or {}).get("missing_legacy_images", [])),
        "conflicting_legacy_images": list((legacy_report or {}).get("conflicting_legacy_images", [])),
        "external_image_dirs": list((legacy_report or {}).get("external_image_dirs", [])),
        "readability_checked_all": readability_checked_all,
    }


def _existing_annotation_for_image(
    image_path: Path,
    *,
    root: Path,
    split: str = "train",
    persist_imported: bool = False,
    annotator: str | None = None,
) -> Annotation | None:
    existing_path = find_annotation_path(image_path.name, root, image_path)
    if existing_path is not None:
        existing = ensure_keypoint_template(load_annotation_from_path(existing_path))
        existing.image.filename = image_path.name
        existing.image.split = normalize_split(split)
        if persist_imported and existing_path.resolve() != annotation_path(image_path.name, root).resolve():
            save_annotation(existing, root, sync_yolo=False)
        return existing

    imported = load_annotation_from_yolo_label(image_path, root, annotator=annotator or load_settings().get("annotator", "default"))
    if imported is None:
        return None
    imported.image.split = normalize_split(split)
    imported = ensure_keypoint_template(imported)
    if persist_imported:
        save_annotation(imported, root, sync_yolo=False)
    return imported


def _auto_annotation_for_image(
    filename: str,
    image: Image.Image,
    *,
    root: Path,
    split: str = "train",
    metadata: dict | None = None,
) -> Annotation:
    existing = load_annotation(filename, root)
    if existing is not None:
        existing = ensure_keypoint_template(existing)
        return _apply_image_metadata(existing, metadata or {})

    settings = load_settings()
    width, height = image.size
    annotation = create_blank_annotation(filename, width, height, annotator=settings.get("annotator", "default"))
    annotation.image.split = normalize_split(split)
    _apply_image_metadata(annotation, metadata or {})
    if settings.get("auto_detect", True):
        run = estimate_keypoints_original_then_enhanced(image)
        if run.usable:
            annotation.keypoints = run.result.keypoints
        else:
            annotation = clear_non_manual_keypoints(annotation)
        annotation.auto_initialization = _auto_initialization_from_run(run, created_at=annotation.annotator.created_at)
    else:
        annotation.auto_initialization = {
            "source": "disabled",
            "warnings": [],
            "created_at": annotation.annotator.created_at,
        }
    return ensure_keypoint_template(annotation)


def _blank_annotation_for_image(filename: str, image: Image.Image, *, split: str = "train", metadata: dict | None = None) -> Annotation:
    settings = load_settings()
    annotation = create_blank_annotation(filename, image.width, image.height, annotator=settings.get("annotator", "default"))
    annotation.image.split = normalize_split(split)
    _apply_image_metadata(annotation, metadata or {})
    annotation.auto_initialization = {
        "source": "queued" if settings.get("auto_detect", True) else "disabled",
        "warnings": [],
        "created_at": annotation.annotator.created_at,
    }
    return ensure_keypoint_template(annotation)


def _count_visible_keypoints(annotation: Annotation) -> int:
    return sum(1 for point in annotation.keypoints.values() if point.visible and point.x is not None and point.y is not None)


def _disabled_template_fallback(*, reason: str, model_visible_count: int) -> dict[str, object]:
    return {
        "enabled": False,
        "filled_count": 0,
        "reason": reason,
        "model_visible_count": int(model_visible_count),
        "note": "Template fallback is disabled; missing model points remain missing for manual annotation.",
    }


def _model_visible_count_for_run(run) -> int:
    return visible_keypoint_count(run.result.keypoints) if run.usable else 0


def _auto_initialization_from_run(
    run,
    *,
    created_at: str,
    manual_retry: bool = False,
    include_partial: bool | None = None,
    preserve_manual: bool | None = None,
) -> dict:
    model_visible_count = _model_visible_count_for_run(run)
    result = run.result
    payload = {
        "source": result.source,
        "strategy": result.strategy,
        "attempts": result.attempts,
        "warnings": list(result.warnings),
        "image_preprocess": run.image_preprocess,
        "preprocess_policy": AUTO_DETECT_POLICY_LABEL,
        "preprocess_attempts": run.preprocess_attempts,
        "roi_crop_used": run.roi_used,
        "scan_transform_used": run.scan_used,
        "model_visible_count": model_visible_count,
        "template_fallback": _disabled_template_fallback(
            reason=result.source if model_visible_count == 0 else "partial-missing-keypoints",
            model_visible_count=model_visible_count,
        ),
        "created_at": created_at,
    }
    if manual_retry:
        payload["manual_retry"] = True
    if include_partial is not None:
        payload["include_partial"] = include_partial
    if preserve_manual is not None:
        payload["preserve_manual"] = preserve_manual
    return payload


def _count_manual_keypoints(annotation: Annotation) -> int:
    return sum(
        1
        for point in annotation.keypoints.values()
        if point.visible and point.x is not None and point.y is not None and point.source == "manual"
    )


def _merge_auto_keypoints(existing: Annotation, auto_annotation: Annotation, *, preserve_manual: bool) -> Annotation:
    existing = ensure_keypoint_template(existing)
    auto_annotation = ensure_keypoint_template(auto_annotation)
    for key, auto_point in auto_annotation.keypoints.items():
        current = existing.keypoints.get(key)
        if (
            preserve_manual
            and current is not None
            and current.visible
            and current.x is not None
            and current.y is not None
            and current.source == "manual"
        ):
            continue
        existing.keypoints[key] = auto_point
    return ensure_keypoint_template(existing)


def _choose_folder_with_dialog(title: str) -> Path:
    try:
        import tkinter as tk
        from tkinter import filedialog

        window = tk.Tk()
        window.withdraw()
        window.attributes("-topmost", True)
        selected = filedialog.askdirectory(title=title)
        window.destroy()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Folder picker unavailable: {exc}") from exc
    if not selected:
        raise HTTPException(status_code=400, detail="No folder selected.")
    return Path(selected).expanduser().resolve()


@router.get("/settings")
async def get_settings():
    settings = load_settings()
    ensure_dataset_layout(Path(settings["dataset_root"]))
    return settings


@router.post("/settings")
async def update_settings(request: SettingsRequest):
    payload = {key: value for key, value in model_to_dict(request).items() if value is not None}
    settings = save_settings(payload)
    return settings


@router.post("/select-folder")
async def select_folder(request: SelectFolderRequest):
    title = "选择数据集保存目录" if request.purpose == "dataset" else "选择图像导入目录"
    path = _choose_folder_with_dialog(title)
    if request.purpose == "dataset":
        settings = save_settings({"dataset_root": str(path)})
        return {"path": str(path), "settings": settings}
    return {"path": str(path)}


@router.post("/load")
async def load_image_annotation(file: UploadFile = File(...)):
    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    filename = _safe_filename(file.filename)
    root = current_root()
    ensure_dataset_layout(root)
    split = "train"
    image_path = image_path_for(filename, root, split)
    with open(image_path, "wb") as handle:
        handle.write(content)
    try:
        image, metadata = _read_image_record(image_path)
    except HTTPException:
        try:
            image_path.unlink()
        except OSError:
            pass
        raise

    annotation = _auto_annotation_for_image(filename, image, root=root, split=split, metadata=metadata)
    upsert_manifest_image(image_path, annotation=annotation, root=root, split=split)
    payload = model_to_dict(annotation)
    payload["image_url"] = _image_url_for_path(image_path)
    return payload


@router.post("/open-folder")
async def open_folder(request: OpenFolderRequest):
    folder = Path(request.folder_path).expanduser().resolve()
    if not folder.exists():
        raise HTTPException(status_code=400, detail=f"Folder not found: {folder}")
    if not folder.is_dir():
        raise HTTPException(status_code=400, detail=f"Not a directory: {folder}")

    root = folder
    settings_before = load_settings()
    split = normalize_split(request.split)
    image_files = _scan_workspace_images(root)
    legacy_report: dict | None = None
    if not image_files:
        legacy_records = _scan_legacy_annotation_records(root)
        if not legacy_records:
            raise HTTPException(status_code=400, detail="No supported image files found in the folder.")
        image_files, legacy_report = _materialize_legacy_only_workspace(root, legacy_records)
        if not image_files:
            raise HTTPException(
                status_code=400,
                detail="找到旧标注 JSON，但未找到对应图片；请确认图片目录或把图片放入当前文件夹。",
            )

    selected_for_report = list(image_files)
    unreadable_files, readability_checked_all = _readability_check(image_files)
    if readability_checked_all and unreadable_files:
        unreadable = {path.resolve() for path in unreadable_files}
        image_files = [path for path in image_files if path.resolve() not in unreadable]
    if not image_files:
        raise HTTPException(status_code=400, detail="No readable supported image files found in the folder.")
    import_report = _import_folder_report(
        root,
        selected_for_report,
        unreadable_files=unreadable_files,
        readability_checked_all=readability_checked_all,
        legacy_report=legacy_report,
    )

    settings = save_settings({"dataset_root": str(root)})
    queued: list[Path] = []
    imported_or_saved = 0
    annotator = settings_before.get("annotator", "default")
    manifest_by_id = {}
    for image_path in image_files:
        image_split = _split_from_image_path(image_path) if image_path.parent.name in {"train", "val"} else split
        annotation = _existing_annotation_for_image(
            image_path,
            root=root,
            split=image_split,
            persist_imported=True,
            annotator=annotator,
        )
        if annotation is None:
            manifest_item = manifest_image_for_path(image_path, annotation=None, root=root, split=image_split)
            if settings.get("auto_detect", True):
                queued.append(image_path)
        else:
            imported_or_saved += 1
            manifest_item = manifest_image_for_path(image_path, annotation=annotation, root=root, split=image_split)
            if settings.get("auto_detect", True) and should_retry_auto_detection(annotation):
                queued.append(image_path)
        manifest_by_id[manifest_item.id] = manifest_item
    manifest = Manifest(images=list(manifest_by_id.values()))
    save_manifest(manifest, root)
    queue_status = replace_auto_detect_queue(
        root,
        queued if settings.get("auto_detect", True) else [],
        split=split,
        annotator=annotator,
    )
    return {
        "status": "success",
        "added": len(image_files),
        "existing": imported_or_saved,
        "queued": len(queued),
        "total": len(manifest.images),
        "settings": settings,
        "auto_detect": queue_status,
        "import_report": import_report,
    }


@router.get("/list")
async def list_annotations():
    root = current_root()
    ensure_dataset_layout(root)
    manifest = load_manifest(root)
    manifest = _refresh_stale_manifest_progress(root, manifest)
    payload = model_to_dict(manifest)
    payload["settings"] = load_settings()
    payload["auto_detect"] = auto_detect_status(root)
    payload["progress"] = build_progress_payload_from_manifest(manifest)
    return payload


@router.get("/load-by-name")
async def load_annotation_by_name(filename: str):
    filename = _safe_filename(filename)
    root = current_root()
    image_path = find_image_path(filename, root)
    if image_path is None or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    split = _split_from_image_path(image_path)
    image, metadata = _read_image_record(image_path)
    annotation = _existing_annotation_for_image(image_path, root=root, split=split, persist_imported=True)
    settings = load_settings()
    if annotation is None:
        if settings.get("auto_detect", True):
            annotation = _auto_annotation_for_image(filename, image, root=root, split=split, metadata=metadata)
            save_annotation(annotation, root)
            upsert_manifest_image(image_path, annotation=annotation, root=root, split=split)
        else:
            annotation = _blank_annotation_for_image(filename, image, split=split, metadata=metadata)
            upsert_manifest_image(image_path, annotation=None, root=root, split=split)
    else:
        annotation.image.width = image.width
        annotation.image.height = image.height
        _apply_image_metadata(annotation, metadata)
        if settings.get("auto_detect", True) and should_retry_auto_detection(annotation):
            run = estimate_keypoints_original_then_enhanced(image)
            if run.usable:
                annotation.keypoints = run.result.keypoints
            else:
                annotation = clear_non_manual_keypoints(annotation)
            annotation.auto_initialization = _auto_initialization_from_run(run, created_at=annotation.annotator.created_at)
            annotation.measurements_snapshot = compute_measurements(annotation)
            save_annotation(annotation, root)
            upsert_manifest_image(image_path, annotation=annotation, root=root, split=split)
    payload = model_to_dict(annotation)
    payload["image_url"] = _image_url_for_path(image_path)
    return payload


@router.post("/save")
async def save_annotation_data(annotation: Annotation, skip_manifest: bool = False):
    annotation = ensure_keypoint_template(annotation)
    annotation.measurements_snapshot = compute_measurements(annotation)
    progress = annotation_progress(annotation)
    root = current_root()
    ensure_dataset_layout(root)
    save_annotation(annotation, root)
    image_path = find_image_path(annotation.image.filename, root)
    if not skip_manifest and image_path is not None and image_path.exists():
        upsert_manifest_image(image_path, annotation=annotation, root=root, split=annotation.image.split)
    saved_label = label_path(annotation.image.filename, root, annotation.image.split)
    payload = {
        "status": "success",
        "annotation_path": str(annotation_path(annotation.image.filename, root).relative_to(root)),
        "label_path": str(saved_label.relative_to(root)) if saved_label.is_relative_to(root) else str(saved_label),
        "measurements_snapshot": annotation.measurements_snapshot,
        "annotation_status": str(progress["status"]),
        "annotation_progress": progress,
        "manifest_skipped": skip_manifest,
    }
    if not skip_manifest:
        payload["progress"] = build_progress_payload_from_manifest(load_manifest(root))
    return payload


@router.delete("/image/{image_filename:path}")
async def delete_image(image_filename: str):
    filename = _safe_filename(image_filename)
    root = current_root()
    payload = _delete_image_files(filename, root)
    manifest = load_manifest(root)
    return {
        "status": "success",
        "filename": filename,
        "deleted": payload["deleted"],
        "trashed": payload["trashed"],
        "trash_dir": payload["trash_dir"],
        "progress": build_progress_payload_from_manifest(manifest),
    }


@router.post("/images/delete")
async def delete_images(request: BatchDeleteImagesRequest):
    root = current_root()
    results: list[dict] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    for raw_filename in request.filenames:
        filename = _safe_filename(raw_filename)
        if filename in seen:
            continue
        seen.add(filename)
        try:
            result = _delete_image_files(filename, root)
            result["status"] = "success"
            results.append(result)
        except HTTPException as exc:
            errors.append({"filename": filename, "detail": str(exc.detail)})
    manifest = load_manifest(root)
    return {
        "status": "success" if not errors else "partial",
        "deleted_count": len(results),
        "failed_count": len(errors),
        "results": results,
        "errors": errors,
        "progress": build_progress_payload_from_manifest(manifest),
    }


@router.get("/trash")
async def list_trash_images():
    root = current_root()
    ensure_dataset_layout(root)
    items: list[dict] = []
    for trash_dir in sorted((path for path in root.rglob("trash") if path.is_dir()), key=lambda item: str(item)):
        if trash_dir.name != "trash":
            continue
        for path in sorted(trash_dir.iterdir()):
            if not path.is_file() or not _is_image_file(path):
                continue
            sidecars = _trash_sidecars_for_image(path)
            items.append(
                {
                    "filename": path.name,
                    "trash_path": _path_for_response(path, root),
                    "restore_dir": _path_for_response(trash_dir.parent, root),
                    "sidecars": [_path_for_response(sidecar, root) for sidecar in sidecars],
                    "sidecar_count": len(sidecars),
                }
            )
    return {"items": items, "total": len(items)}


@router.post("/trash/restore")
async def restore_trash_images(request: RestoreTrashRequest):
    root = current_root()
    results: list[dict] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()
    for trash_path in request.trash_paths:
        if trash_path in seen:
            continue
        seen.add(trash_path)
        try:
            result = _restore_trash_image(trash_path, root)
            if result.get("status") == "restored":
                results.append(result)
            else:
                errors.append({"trash_path": trash_path, "detail": result.get("detail", "Restore failed.")})
        except HTTPException as exc:
            errors.append({"trash_path": trash_path, "detail": str(exc.detail)})
    manifest = load_manifest(root)
    return {
        "status": "success" if not errors else "partial",
        "restored_count": len(results),
        "failed_count": len(errors),
        "results": results,
        "errors": errors,
        "progress": build_progress_payload_from_manifest(manifest),
    }


@router.post("/auto-detect-image")
async def auto_detect_image(request: AutoDetectImageRequest):
    filename = _safe_filename(request.filename)
    root = current_root()
    image_path = find_image_path(filename, root)
    if image_path is None or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")

    image, metadata = _read_image_record(image_path)
    split = _split_from_image_path(image_path)
    settings = load_settings()
    existing = load_annotation(filename, root)
    if existing is None:
        existing = create_blank_annotation(filename, image.width, image.height, annotator=settings.get("annotator", "default"))
        existing.image.split = split
    else:
        existing = ensure_keypoint_template(existing)
        existing.image.width = image.width
        existing.image.height = image.height
        existing.image.split = normalize_split(split)
    _apply_image_metadata(existing, metadata)

    run = estimate_keypoints_original_then_enhanced(
        image,
        use_scan=request.use_scan,
        min_visible_keypoints=1,
        include_partial=request.include_partial,
        roi_crop=existing.roi_crop if request.use_roi else None,
        scan_transform=existing.scan_transform,
    )
    result = run.result
    model_visible_count = _model_visible_count_for_run(run)
    auto_annotation = create_blank_annotation(filename, image.width, image.height, annotator=existing.annotator.user_id)
    auto_annotation.image.split = normalize_split(split)
    auto_annotation.keypoints = result.keypoints
    auto_annotation = ensure_keypoint_template(auto_annotation)
    preserved_manual = _count_manual_keypoints(existing) if request.preserve_manual else 0

    if run.usable:
        annotation = _merge_auto_keypoints(existing, auto_annotation, preserve_manual=request.preserve_manual)
    else:
        annotation = clear_non_manual_keypoints(existing) if request.preserve_manual else auto_annotation
    applied_visible_count = _count_visible_keypoints(annotation)
    applied = run.usable

    warnings = list(result.warnings)
    annotation.auto_initialization = _auto_initialization_from_run(
        run,
        created_at=annotation.annotator.created_at,
        manual_retry=True,
        include_partial=request.include_partial,
        preserve_manual=request.preserve_manual,
    )
    _apply_image_metadata(annotation, metadata)
    annotation.measurements_snapshot = compute_measurements(annotation)
    save_annotation(annotation, root)
    upsert_manifest_image(image_path, annotation=annotation, root=root, split=annotation.image.split)
    saved_label = label_path(annotation.image.filename, root, annotation.image.split)
    payload = model_to_dict(annotation)
    payload["image_url"] = _image_url_for_path(image_path)
    payload["auto_detect"] = {
        "source": result.source,
        "strategy": result.strategy,
        "attempts": result.attempts,
        "warnings": warnings,
        "visible_count": model_visible_count,
        "applied_visible_count": applied_visible_count,
        "applied": applied,
        "preserved_manual_count": preserved_manual,
        "image_preprocess": run.image_preprocess,
        "preprocess_policy": AUTO_DETECT_POLICY_LABEL,
        "preprocess_attempts": run.preprocess_attempts,
        "roi_crop_used": run.roi_used,
        "scan_transform_used": run.scan_used,
        "template_fallback": annotation.auto_initialization["template_fallback"],
        "annotation_path": str(annotation_path(annotation.image.filename, root).relative_to(root)),
        "label_path": str(saved_label.relative_to(root)) if saved_label.is_relative_to(root) else str(saved_label),
    }
    return payload


@router.get("/auto-detect/status")
async def get_auto_detect_status():
    return auto_detect_status(current_root())


@router.post("/measurements/compute")
async def compute_annotation_measurements(annotation: Annotation):
    annotation = ensure_keypoint_template(annotation)
    return compute_measurements(annotation)


@router.get("/image/{image_filename:path}")
async def serve_image(image_filename: str, enhanced: bool = False, thumb: bool = False):
    root = current_root()
    image_path = find_image_path(_safe_filename(image_filename), root)
    if image_path is None or not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    if thumb:
        return FileResponse(cached_thumbnail_jpeg(image_path), media_type="image/jpeg")
    if _needs_browser_rendering(image_path) or enhanced:
        return FileResponse(cached_rendered_png(image_path, enhanced=enhanced), media_type="image/png")
    return FileResponse(image_path)


@router.get("/export-json")
async def export_json_annotations():
    root = current_root()
    path = annotations_dir(root)
    if not path.exists():
        raise HTTPException(status_code=404, detail="No annotations directory found.")
    annotation_files = sorted(path.glob("*.json"))
    if not annotation_files:
        raise HTTPException(status_code=404, detail="No annotation files found.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for item in annotation_files:
            zf.write(item, arcname=f"annotations/{item.name}")
        manifest = root / "manifest.json"
        if manifest.exists():
            zf.write(manifest, arcname="manifest.json")
        data_yaml = data_yaml_path(root)
        if data_yaml.exists():
            zf.write(data_yaml, arcname="data.yaml")
    buf.seek(0)
    filename = f"hip22_annotations_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/export-yolo")
async def export_yolo_annotations():
    root = current_root()
    path = annotations_dir(root)
    if not path.exists():
        raise HTTPException(status_code=404, detail="No annotations directory found.")
    annotations: list[Annotation] = []
    for item in sorted(path.glob("*.json")):
        annotation = load_annotation(item.name, root)
        if annotation is None:
            continue
        annotations.append(annotation)
    if not annotations:
        raise HTTPException(status_code=404, detail="No annotation files found.")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("data.yaml", data_yaml_text(".", train="images/train", val="images/val"))
        for annotation in annotations:
            stem = Path(annotation.image.filename).stem
            split = normalize_split(getattr(annotation.image, "split", "train"))
            zf.writestr(f"labels/{split}/{stem}.txt", annotation_to_yolo_text(annotation))
    buf.seek(0)
    filename = f"hip22_yolo_pose_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


@router.get("/schema")
async def annotation_schema():
    return {
        "sides": ["left", "right"],
        "side_convention": "left/right keys follow image-left/image-right; no anatomical swap",
        "landmarks": [
            {
                "number": item.number,
                "name": item.name,
                "label_zh": item.label_zh,
                "description": item.description,
            }
            for item in LANDMARK_DEFS
        ],
    }
