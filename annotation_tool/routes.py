from __future__ import annotations

import io
import time
import zipfile
from pathlib import Path
from typing import Optional, Set

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from PIL import Image, ImageOps
from pydantic import BaseModel

from .auto_detect_queue import auto_detect_status, replace_auto_detect_queue
from .heuristics import estimate_keypoints_from_image
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
from .yolo_export import annotation_to_yolo_text, data_yaml_text


IMAGE_EXTENSIONS: Set[str] = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}

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


def _is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def _safe_filename(filename: str | None) -> str:
    name = Path(filename or "upload.png").name
    if not name or name in {".", ".."}:
        return f"upload_{int(time.time())}.png"
    return name


def _read_image_bytes(content: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(content))
        image = ImageOps.exif_transpose(image)
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Unable to read image file.") from exc
    return image.convert("RGB")


def _read_image_path(path: Path) -> Image.Image:
    try:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image)
        image.load()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to read image: {path.name}") from exc
    return image.convert("RGB")


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


def _auto_annotation_for_image(filename: str, image: Image.Image, *, root: Path, split: str = "train") -> Annotation:
    existing = load_annotation(filename, root)
    if existing is not None:
        return ensure_keypoint_template(existing)

    settings = load_settings()
    width, height = image.size
    annotation = create_blank_annotation(filename, width, height, annotator=settings.get("annotator", "default"))
    annotation.image.split = normalize_split(split)
    if settings.get("auto_detect", True):
        result = estimate_keypoints_from_image(image)
        annotation.keypoints = result.keypoints
        annotation.auto_initialization = {
            "source": "yolo11n-best-side11" if result.model_available else "model-unavailable",
            "warnings": result.warnings,
            "created_at": annotation.annotator.created_at,
        }
    else:
        annotation.auto_initialization = {
            "source": "disabled",
            "warnings": [],
            "created_at": annotation.annotator.created_at,
        }
    return ensure_keypoint_template(annotation)


def _blank_annotation_for_image(filename: str, image: Image.Image, *, split: str = "train") -> Annotation:
    settings = load_settings()
    annotation = create_blank_annotation(filename, image.width, image.height, annotator=settings.get("annotator", "default"))
    annotation.image.split = normalize_split(split)
    annotation.auto_initialization = {
        "source": "queued" if settings.get("auto_detect", True) else "disabled",
        "warnings": [],
        "created_at": annotation.annotator.created_at,
    }
    return ensure_keypoint_template(annotation)


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
    image = _read_image_bytes(content)
    root = current_root()
    ensure_dataset_layout(root)
    split = "train"
    image_path = image_path_for(filename, root, split)
    with open(image_path, "wb") as handle:
        handle.write(content)

    annotation = _auto_annotation_for_image(filename, image, root=root, split=split)
    upsert_manifest_image(image_path, annotation=annotation, root=root, split=split)
    payload = model_to_dict(annotation)
    payload["image_url"] = f"/api/annotation/image/{filename}"
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
    }


@router.get("/list")
async def list_annotations():
    root = current_root()
    ensure_dataset_layout(root)
    payload = model_to_dict(load_manifest(root))
    payload["settings"] = load_settings()
    payload["auto_detect"] = auto_detect_status(root)
    return payload


@router.get("/load-by-name")
async def load_annotation_by_name(filename: str):
    filename = _safe_filename(filename)
    root = current_root()
    image_path = find_image_path(filename, root)
    if image_path is None or not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found.")
    split = _split_from_image_path(image_path)
    annotation = _existing_annotation_for_image(image_path, root=root, split=split, persist_imported=True)
    if annotation is None:
        annotation = _blank_annotation_for_image(filename, _read_image_path(image_path), split=split)
        upsert_manifest_image(image_path, annotation=None, root=root, split=split)
    else:
        upsert_manifest_image(image_path, annotation=annotation, root=root, split=split)
    payload = model_to_dict(annotation)
    payload["image_url"] = f"/api/annotation/image/{filename}"
    return payload


@router.post("/save")
async def save_annotation_data(annotation: Annotation):
    annotation = ensure_keypoint_template(annotation)
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
    }


@router.get("/auto-detect/status")
async def get_auto_detect_status():
    return auto_detect_status(current_root())


@router.get("/image/{image_filename:path}")
async def serve_image(image_filename: str):
    root = current_root()
    image_path = find_image_path(_safe_filename(image_filename), root)
    if image_path is None or not image_path.exists() or not image_path.is_file():
        raise HTTPException(status_code=404, detail="Image not found.")
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
