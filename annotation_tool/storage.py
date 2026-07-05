from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Optional

from .paths import default_dataset_root, runtime_root
from .schema import Annotation, Manifest, ManifestImage, annotation_from_dict, model_to_dict, normalize_split, utc_now


ROOT = runtime_root()
IMAGES_DIRNAME = "images"
LABELS_DIRNAME = "labels"
ANNOTATIONS_DIRNAME = "annotations"
SPLITS_DIRNAME = "splits"
MANIFEST_NAME = "manifest.json"
SETTINGS_NAME = "tool-settings.json"
DATA_YAML_NAME = "data.yaml"
_manifest_lock = threading.RLock()
_settings_lock = threading.RLock()


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
    tmp_path.replace(path)


def _atomic_write_json(path: Path, payload: dict) -> None:
    _atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False))


def settings_path() -> Path:
    return ROOT / SETTINGS_NAME


def default_settings() -> dict:
    return {
        "dataset_root": str(default_dataset_root()),
        "auto_detect": True,
        "autosave": True,
        "annotator": "default",
    }


def load_settings() -> dict:
    path = settings_path()
    settings = default_settings()
    with _settings_lock:
        if path.exists():
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            if isinstance(payload, dict):
                settings.update(payload)
    settings["dataset_root"] = str(Path(settings.get("dataset_root") or ROOT).expanduser().resolve())
    settings["auto_detect"] = bool(settings.get("auto_detect", True))
    settings["autosave"] = bool(settings.get("autosave", False))
    settings["annotator"] = str(settings.get("annotator") or "default")
    return settings


def save_settings(payload: dict) -> dict:
    with _settings_lock:
        current = load_settings()
        for key in ("dataset_root", "auto_detect", "autosave", "annotator"):
            if key in payload:
                current[key] = payload[key]
        current["dataset_root"] = str(Path(current.get("dataset_root") or ROOT).expanduser().resolve())
        current["auto_detect"] = bool(current.get("auto_detect", True))
        current["autosave"] = bool(current.get("autosave", False))
        current["annotator"] = str(current.get("annotator") or "default")
        _atomic_write_json(settings_path(), current)
    ensure_dataset_layout(Path(current["dataset_root"]))
    return current


def current_root() -> Path:
    return Path(load_settings()["dataset_root"]).expanduser().resolve()


def images_dir(root: Path = ROOT, split: str | None = None) -> Path:
    base = root / IMAGES_DIRNAME
    return base / normalize_split(split) if split else base


def labels_dir(root: Path = ROOT, split: str | None = None) -> Path:
    base = root / LABELS_DIRNAME
    return base / normalize_split(split) if split else base


def annotations_dir(root: Path = ROOT) -> Path:
    return root / ANNOTATIONS_DIRNAME


def splits_dir(root: Path = ROOT) -> Path:
    return root / SPLITS_DIRNAME


def manifest_path(root: Path = ROOT) -> Path:
    return root / MANIFEST_NAME


def data_yaml_path(root: Path = ROOT) -> Path:
    return root / DATA_YAML_NAME


def annotation_path(image_filename: str, root: Path = ROOT) -> Path:
    return annotations_dir(root) / f"{Path(image_filename).stem}.json"


def image_path_for(image_filename: str, root: Path = ROOT, split: str = "train") -> Path:
    return root / Path(image_filename).name


def find_image_path(image_filename: str, root: Path = ROOT) -> Path | None:
    filename = Path(image_filename).name
    candidates = [
        root / filename,
        images_dir(root, "train") / filename,
        images_dir(root, "val") / filename,
        images_dir(root) / filename,
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def label_path_for_image_path(image_path: Path) -> Path:
    return image_path.with_suffix(".txt")


def label_path(image_filename: str, root: Path = ROOT, split: str = "train") -> Path:
    image_path = find_image_path(image_filename, root)
    if image_path is not None:
        return label_path_for_image_path(image_path)
    return root / f"{Path(image_filename).stem}.txt"


def find_sidecar_label_path(image_path: Path, root: Path = ROOT) -> Path | None:
    candidates = [
        label_path_for_image_path(image_path),
        root / f"{image_path.stem}.txt",
        labels_dir(root, "train") / f"{image_path.stem}.txt",
        labels_dir(root, "val") / f"{image_path.stem}.txt",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def ensure_dataset_layout(root: Path = ROOT) -> None:
    root.mkdir(parents=True, exist_ok=True)
    annotations_dir(root).mkdir(parents=True, exist_ok=True)
    splits_dir(root).mkdir(parents=True, exist_ok=True)
    split_path = splits_dir(root) / "train_val_split.json"
    if not split_path.exists():
        with open(split_path, "w", encoding="utf-8") as handle:
            json.dump({"train": [], "val": []}, handle, indent=2, ensure_ascii=False)
    from .yolo_export import data_yaml_text

    _atomic_write_text(data_yaml_path(root), data_yaml_text(str(root.resolve())))


def checksum_sha256(file_path: Path) -> str:
    h = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_annotation(image_filename: str, root: Path = ROOT) -> Optional[Annotation]:
    path = annotation_path(image_filename, root)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return annotation_from_dict(json.load(handle))


def load_annotation_from_yolo_label(
    image_path: Path,
    root: Path = ROOT,
    *,
    annotator: str = "default",
) -> Optional[Annotation]:
    label_file = find_sidecar_label_path(image_path, root)
    if label_file is None:
        return None
    from .yolo_export import annotation_from_yolo_text
    from .image_io import read_supported_image

    try:
        image, metadata = read_supported_image(image_path)
    except Exception:
        return None
    text = label_file.read_text(encoding="utf-8")
    annotation = annotation_from_yolo_text(
        text,
        image_path.name,
        int(image.width),
        int(image.height),
        annotator=annotator,
    )
    for key, value in metadata.items():
        setattr(annotation.image, key, value)
    return annotation


def save_annotation(annotation: Annotation, root: Path = ROOT, *, sync_yolo: bool = True) -> Path:
    ensure_dataset_layout(root)
    annotation.annotator.updated_at = utc_now()
    annotation.image.split = normalize_split(getattr(annotation.image, "split", "train"))
    path = annotation_path(annotation.image.filename, root)
    _atomic_write_json(path, model_to_dict(annotation))
    if sync_yolo:
        write_yolo_label(annotation, root)
    return path


def write_yolo_label(annotation: Annotation, root: Path = ROOT) -> Path:
    from .yolo_export import annotation_to_yolo_text

    path = label_path(annotation.image.filename, root, normalize_split(getattr(annotation.image, "split", "train")))
    _atomic_write_text(path, annotation_to_yolo_text(annotation))
    return path


def load_manifest(root: Path = ROOT) -> Manifest:
    path = manifest_path(root)
    with _manifest_lock:
        if not path.exists():
            return Manifest()
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return Manifest(**json.load(handle))
        except json.JSONDecodeError:
            corrupt_path = path.with_suffix(f".corrupt-{utc_now().replace(':', '').replace('-', '')}.json")
            path.replace(corrupt_path)
            return Manifest()


def save_manifest(manifest: Manifest, root: Path = ROOT) -> Path:
    with _manifest_lock:
        root.mkdir(parents=True, exist_ok=True)
        path = manifest_path(root)
        _atomic_write_json(path, model_to_dict(manifest))
        from .progress_report import write_progress_reports

        write_progress_reports(root, manifest)
        return path


def annotation_status(annotation: Annotation) -> str:
    visible_count = sum(1 for kp in annotation.keypoints.values() if kp.visible and kp.x is not None and kp.y is not None)
    manual_count = sum(1 for kp in annotation.keypoints.values() if kp.visible and kp.source == "manual")
    if visible_count == 0:
        return "pending"
    if manual_count == 0:
        return "auto"
    if visible_count == 22:
        return "done"
    if manual_count > 0:
        return "in_progress"
    return "pending"


def upsert_manifest_image(
    image_path: Path,
    *,
    annotation: Annotation | None = None,
    root: Path = ROOT,
    split: str = "train",
    compute_checksum: bool = False,
) -> Manifest:
    with _manifest_lock:
        ensure_dataset_layout(root)
        manifest = load_manifest(root)
        image_id = image_path.stem
        split = normalize_split(getattr(annotation.image, "split", split) if annotation else split)
        try:
            rel_image_path = image_path.resolve().relative_to(root.resolve()).as_posix()
        except ValueError:
            rel_image_path = f"{IMAGES_DIRNAME}/{split}/{image_path.name}"
        rel_annotation_path = f"{ANNOTATIONS_DIRNAME}/{image_path.stem}.json"
        existing = next((item for item in manifest.images if item.id == image_id), None)
        status = annotation_status(annotation) if annotation else "pending"
        annotator = annotation.annotator.user_id if annotation else ""
        completed_at = utc_now() if status == "done" else None
        checksum = checksum_sha256(image_path) if compute_checksum and image_path.exists() else None
        if existing is None:
            manifest.images.append(
                ManifestImage(
                    id=image_id,
                    image_path=rel_image_path,
                    annotation_path=rel_annotation_path,
                    status=status,
                    annotator=annotator,
                    completed_at=completed_at,
                    checksum_sha256=checksum,
                )
            )
        else:
            existing.image_path = rel_image_path
            existing.annotation_path = rel_annotation_path
            existing.status = status
            existing.annotator = annotator
            existing.completed_at = completed_at
            existing.checksum_sha256 = checksum
        save_manifest(manifest, root)
        return manifest


def image_path_from_manifest_item(item: ManifestImage, root: Path = ROOT) -> Path:
    path = root / item.image_path
    if path.exists():
        return path
    return find_image_path(Path(item.image_path).name, root) or path
