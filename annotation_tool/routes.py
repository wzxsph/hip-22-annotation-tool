from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel

from .auto_detect_queue import auto_detect_status, replace_auto_detect_queue
from .auto_detection import estimate_keypoints_with_preprocessing
from .dicom_utils import is_dicom_path
from .image_io import is_supported_image_path, read_supported_image
from .measurements import compute_measurements
from .progress_report import build_progress_payload
from .render_cache import cached_rendered_png
from .schema import LANDMARK_DEFS, Annotation, Manifest, create_blank_annotation, ensure_keypoint_template, model_to_dict, normalize_split
from .storage import (
    annotation_path,
    annotations_dir,
    current_root,
    data_yaml_path,
    ensure_dataset_layout,
    find_image_path,
    image_path_for,
    load_annotation,
    load_annotation_from_yolo_label,
    load_manifest,
    load_settings,
    save_annotation,
    save_manifest,
    save_settings,
    upsert_manifest_image,
    label_path,
)
from .template_points import apply_template_fallback, visible_keypoint_count
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


class SelectFolderRequest(BaseModel):
    purpose: str = "import"


class AutoDetectImageRequest(BaseModel):
    filename: str
    preserve_manual: bool = True
    include_partial: bool = True
    use_enhanced: bool = False
    use_roi: bool = True
    use_scan: bool = True


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
    return f"/api/annotation/image/{filename}?source_v={stat.st_mtime_ns}-{stat.st_size}"


def _split_from_image_path(path: Path) -> str:
    if path.parent.name in {"train", "val"}:
        return path.parent.name
    return "train"


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


def _readable_image_files(paths: list[Path]) -> list[Path]:
    readable: list[Path] = []
    for path in paths:
        try:
            read_supported_image(path)
        except Exception:
            continue
        readable.append(path)
    return readable


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


def _import_folder_report(root: Path, selected_images: list[Path]) -> dict:
    selected = {path.resolve() for path in selected_images}
    supported_images: list[Path] = []
    unsupported_files: list[Path] = []
    unreadable_files: list[Path] = []
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
        }

    for path in all_files:
        if _is_internal_artifact(path, root):
            continue
        if _is_image_file(path):
            supported_images.append(path)
            continue
        if path.suffix.lower() not in {".txt", ".json", ".yaml", ".yml", ".csv", ".html"}:
            unsupported_files.append(path)

    for path in selected_images:
        try:
            read_supported_image(path)
        except Exception:
            unreadable_files.append(path)

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
    if messy_names:
        warnings.append(f"发现 {len(messy_names)} 个文件名可能需要整理。")

    return {
        "warnings": warnings,
        "nested_images": _limited_paths(nested_images, root),
        "duplicate_names": duplicate_names,
        "unsupported_files": _limited_paths(unsupported_files, root),
        "unreadable_files": _limited_paths(unreadable_files, root),
        "messy_names": messy_names,
    }


def _existing_annotation_for_image(
    image_path: Path,
    *,
    root: Path,
    split: str = "train",
    persist_imported: bool = False,
    annotator: str | None = None,
) -> Annotation | None:
    existing = load_annotation(image_path.name, root)
    if existing is not None:
        existing.image.split = normalize_split(split)
        return ensure_keypoint_template(existing)

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
        result, image_preprocess, roi_used, scan_used = estimate_keypoints_with_preprocessing(image, use_enhanced=False)
        model_visible_count = visible_keypoint_count(result.keypoints)
        annotation.keypoints = result.keypoints
        template_fallback = apply_template_fallback(
            annotation,
            reason=result.source if model_visible_count == 0 else "partial-missing-keypoints",
            model_visible_count=model_visible_count,
        )
        warnings = list(result.warnings)
        if template_fallback["enabled"]:
            warnings.append("Auto-detect was incomplete; draggable template points were added for doctor review.")
        annotation.auto_initialization = {
            "source": result.source,
            "strategy": result.strategy,
            "attempts": result.attempts,
            "warnings": warnings,
            "image_preprocess": image_preprocess,
            "roi_crop_used": roi_used,
            "scan_transform_used": scan_used,
            "model_visible_count": model_visible_count,
            "template_fallback": template_fallback,
            "created_at": annotation.annotator.created_at,
        }
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

    settings = save_settings({"dataset_root": str(folder)})
    root = Path(settings["dataset_root"])
    ensure_dataset_layout(root)

    image_files = _scan_workspace_images(root)
    if not image_files:
        raise HTTPException(status_code=400, detail="No supported image files found in the folder.")
    import_report = _import_folder_report(root, image_files)
    image_files = _readable_image_files(image_files)
    if not image_files:
        raise HTTPException(status_code=400, detail="No readable supported image files found in the folder.")

    split = normalize_split(request.split)
    save_manifest(Manifest(), root)
    queued: list[Path] = []
    imported_or_saved = 0
    annotator = settings.get("annotator", "default")
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
            upsert_manifest_image(image_path, annotation=None, root=root, split=image_split)
            if settings.get("auto_detect", True):
                queued.append(image_path)
        else:
            imported_or_saved += 1
            upsert_manifest_image(image_path, annotation=annotation, root=root, split=image_split)
    queue_status = replace_auto_detect_queue(
        root,
        queued if settings.get("auto_detect", True) else [],
        split=split,
        annotator=annotator,
    )
    manifest = load_manifest(root)
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
    payload = model_to_dict(manifest)
    payload["settings"] = load_settings()
    payload["auto_detect"] = auto_detect_status(root)
    payload["progress"] = build_progress_payload(root, manifest)
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
    if annotation is None:
        annotation = _blank_annotation_for_image(filename, image, split=split, metadata=metadata)
        upsert_manifest_image(image_path, annotation=None, root=root, split=split)
    else:
        annotation.image.width = image.width
        annotation.image.height = image.height
        _apply_image_metadata(annotation, metadata)
        upsert_manifest_image(image_path, annotation=annotation, root=root, split=split)
    payload = model_to_dict(annotation)
    payload["image_url"] = _image_url_for_path(image_path)
    return payload


@router.post("/save")
async def save_annotation_data(annotation: Annotation):
    annotation = ensure_keypoint_template(annotation)
    annotation.measurements_snapshot = compute_measurements(annotation)
    root = current_root()
    ensure_dataset_layout(root)
    save_annotation(annotation, root)
    image_path = find_image_path(annotation.image.filename, root)
    if image_path is not None and image_path.exists():
        upsert_manifest_image(image_path, annotation=annotation, root=root, split=annotation.image.split)
    saved_label = label_path(annotation.image.filename, root, annotation.image.split)
    return {
        "status": "success",
        "annotation_path": str(annotation_path(annotation.image.filename, root).relative_to(root)),
        "label_path": str(saved_label.relative_to(root)) if saved_label.is_relative_to(root) else str(saved_label),
        "measurements_snapshot": annotation.measurements_snapshot,
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

    result, image_preprocess, roi_used, scan_used = estimate_keypoints_with_preprocessing(
        image,
        use_enhanced=request.use_enhanced,
        use_scan=request.use_scan,
        min_visible_keypoints=1,
        include_partial=request.include_partial,
        roi_crop=existing.roi_crop if request.use_roi else None,
        scan_transform=existing.scan_transform,
    )
    model_visible_count = visible_keypoint_count(result.keypoints)
    auto_annotation = create_blank_annotation(filename, image.width, image.height, annotator=existing.annotator.user_id)
    auto_annotation.image.split = normalize_split(split)
    auto_annotation.keypoints = result.keypoints
    auto_annotation = ensure_keypoint_template(auto_annotation)
    preserved_manual = _count_manual_keypoints(existing) if request.preserve_manual else 0

    if model_visible_count > 0:
        annotation = _merge_auto_keypoints(existing, auto_annotation, preserve_manual=request.preserve_manual)
    else:
        annotation = existing
    template_fallback = apply_template_fallback(
        annotation,
        reason=result.source if model_visible_count == 0 else "partial-missing-keypoints",
        model_visible_count=model_visible_count,
    )
    applied_visible_count = _count_visible_keypoints(annotation)
    applied = model_visible_count > 0 or template_fallback["enabled"]

    warnings = list(result.warnings)
    if template_fallback["enabled"]:
        warnings.append("Auto-detect was incomplete; draggable template points were added for doctor review.")
    annotation.auto_initialization = {
        "source": result.source,
        "strategy": result.strategy,
        "attempts": result.attempts,
        "warnings": warnings,
        "manual_retry": True,
        "include_partial": request.include_partial,
        "preserve_manual": request.preserve_manual,
        "image_preprocess": image_preprocess,
        "roi_crop_used": roi_used,
        "scan_transform_used": scan_used,
        "model_visible_count": model_visible_count,
        "template_fallback": template_fallback,
        "created_at": annotation.annotator.created_at,
    }
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
        "image_preprocess": image_preprocess,
        "roi_crop_used": roi_used,
        "scan_transform_used": scan_used,
        "template_fallback": template_fallback,
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
async def serve_image(image_filename: str, enhanced: bool = False):
    root = current_root()
    image_path = find_image_path(_safe_filename(image_filename), root)
    if image_path is None or not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
    if is_dicom_path(image_path) or enhanced:
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
