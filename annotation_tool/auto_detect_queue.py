from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .auto_detection import estimate_keypoints_with_preprocessing, preprocess_label
from .image_io import read_supported_image
from .measurements import compute_measurements
from .schema import create_blank_annotation, normalize_split
from .storage import (
    find_sidecar_label_path,
    load_annotation,
    load_annotation_from_yolo_label,
    save_annotation,
    upsert_manifest_image,
)
from .template_points import apply_template_fallback, visible_keypoint_count


@dataclass(frozen=True)
class AutoDetectItem:
    root: Path
    image_path: Path
    split: str = "train"
    annotator: str = "default"
    use_enhanced: bool = True


_lock = threading.Lock()
_queue: deque[AutoDetectItem] = deque()
_worker: threading.Thread | None = None
_status_by_root: dict[str, dict[str, Any]] = {}


def _root_key(root: Path) -> str:
    return str(root.expanduser().resolve())


def _empty_status(root: Path) -> dict[str, Any]:
    return {
        "root": _root_key(root),
        "total": 0,
        "pending": 0,
        "processing": None,
        "done": 0,
        "failed": 0,
        "skipped": 0,
        "running": False,
        "failures": [],
        "image_preprocess": None,
    }


def replace_auto_detect_queue(
    root: Path,
    image_paths: list[Path],
    *,
    split: str = "train",
    annotator: str = "default",
    use_enhanced: bool = True,
) -> dict[str, Any]:
    root = root.expanduser().resolve()
    items = [
        AutoDetectItem(
            root=root,
            image_path=path.expanduser().resolve(),
            split=normalize_split(split),
            annotator=annotator,
            use_enhanced=use_enhanced,
        )
        for path in image_paths
    ]
    with _lock:
        _queue.clear()
        _queue.extend(items)
        status = _empty_status(root)
        status["total"] = len(items)
        status["pending"] = len(items)
        status["running"] = bool(items)
        status["image_preprocess"] = preprocess_label(use_enhanced=use_enhanced) if items else None
        _status_by_root[_root_key(root)] = status
        _ensure_worker_locked()
        return dict(status)


def auto_detect_status(root: Path) -> dict[str, Any]:
    key = _root_key(root)
    with _lock:
        status = _status_by_root.get(key, _empty_status(root))
        return {
            **status,
            "failures": list(status.get("failures", [])),
        }


def _apply_image_metadata(annotation, metadata: dict[str, Any]) -> None:
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


def _ensure_worker_locked() -> None:
    global _worker
    if _worker is not None and _worker.is_alive():
        return
    if not _queue:
        return
    _worker = threading.Thread(target=_worker_loop, name="hip22-auto-detect", daemon=True)
    _worker.start()


def _worker_loop() -> None:
    global _worker
    while True:
        with _lock:
            if not _queue:
                _worker = None
                return
            item = _queue.popleft()
            status = _status_by_root.setdefault(_root_key(item.root), _empty_status(item.root))
            status["pending"] = max(0, int(status.get("pending", 0)) - 1)
            status["processing"] = item.image_path.name
            status["running"] = True
        try:
            outcome = _process_item(item)
            with _lock:
                status = _status_by_root.setdefault(_root_key(item.root), _empty_status(item.root))
                status[outcome] = int(status.get(outcome, 0)) + 1
                status["processing"] = None
                status["running"] = bool(status.get("pending", 0))
        except Exception as exc:  # pragma: no cover - defensive worker guard
            with _lock:
                status = _status_by_root.setdefault(_root_key(item.root), _empty_status(item.root))
                status["failed"] = int(status.get("failed", 0)) + 1
                status["processing"] = None
                status["running"] = bool(status.get("pending", 0))
                failures = list(status.get("failures", []))
                failures.append({"image": item.image_path.name, "error": str(exc)})
                status["failures"] = failures[-10:]


def _process_item(item: AutoDetectItem) -> str:
    root = item.root
    filename = item.image_path.name
    existing = load_annotation(filename, root)
    if existing is not None:
        existing.image.split = normalize_split(item.split)
        try:
            image, metadata = read_supported_image(item.image_path)
            existing.image.width = image.width
            existing.image.height = image.height
            _apply_image_metadata(existing, metadata)
        except Exception:
            pass
        upsert_manifest_image(item.image_path, annotation=existing, root=root, split=item.split)
        return "skipped"

    if find_sidecar_label_path(item.image_path, root) is not None:
        imported = load_annotation_from_yolo_label(item.image_path, root, annotator=item.annotator)
        if imported is not None:
            imported.image.split = normalize_split(item.split)
            try:
                image, metadata = read_supported_image(item.image_path)
                imported.image.width = image.width
                imported.image.height = image.height
                _apply_image_metadata(imported, metadata)
            except Exception:
                pass
            imported.measurements_snapshot = compute_measurements(imported)
            save_annotation(imported, root, sync_yolo=False)
            upsert_manifest_image(item.image_path, annotation=imported, root=root, split=item.split)
        return "skipped"

    image, metadata = read_supported_image(item.image_path)
    annotation = create_blank_annotation(filename, image.width, image.height, annotator=item.annotator)
    annotation.image.split = normalize_split(item.split)
    _apply_image_metadata(annotation, metadata)
    result, image_preprocess, roi_used, scan_used = estimate_keypoints_with_preprocessing(image, use_enhanced=item.use_enhanced)
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
    annotation.measurements_snapshot = compute_measurements(annotation)
    save_annotation(annotation, root)
    upsert_manifest_image(item.image_path, annotation=annotation, root=root, split=item.split)
    return "done"
