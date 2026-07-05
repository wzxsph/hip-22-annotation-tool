from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from annotation_tool.image_io import read_supported_image
from annotation_tool.measurements import compute_measurements
from annotation_tool.schema import Annotation, annotation_from_dict, model_to_dict
from annotation_tool.storage import find_image_path


CLASS_NAMES = {
    0: "obturator_shenton_arc",
    1: "femoral_neck_shenton_arc",
}
SEGMENT_CLASSES = {
    "obturator_upper_curve": 0,
    "femoral_neck_inner_lower_curve": 1,
}
CONTROL_POINT_COUNT = 4
MIN_BOX_SIZE = 0.03
BOX_PADDING = 0.025


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export internal Shenton curve annotations for research training.")
    parser.add_argument("--workspace", required=True, type=Path, help="Hip22 dataset folder that contains annotations/.")
    parser.add_argument("--output", required=True, type=Path, help="Output folder for shenton_curves.jsonl and YOLO pose data.")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="YOLO split name to write.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output folder first if it already exists.")
    return parser.parse_args(argv)


def export_training_set(workspace: Path, output: Path, *, split: str = "train", overwrite: bool = False) -> dict[str, int]:
    workspace = workspace.expanduser().resolve()
    output = output.expanduser().resolve()
    annotations_dir = workspace / "annotations"
    if not annotations_dir.exists():
        raise ValueError(f"Annotation directory not found: {annotations_dir}")
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    jsonl_path = output / "shenton_curves.jsonl"
    yolo_root = output / "yolo_pose"
    images_dir = yolo_root / "images" / split
    labels_dir = yolo_root / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    records = 0
    yolo_images = 0
    yolo_objects = 0
    skipped_missing_image = 0

    with open(jsonl_path, "w", encoding="utf-8") as jsonl:
        for index, annotation_path in enumerate(sorted(annotations_dir.glob("*.json")), start=1):
            annotation = _load_annotation(annotation_path)
            if annotation is None or not _has_shenton_points(annotation):
                continue

            measurements = compute_measurements(annotation)
            annotation.measurements_snapshot = {
                **(annotation.measurements_snapshot or {}),
                **measurements,
            }
            image_path = find_image_path(annotation.image.filename, workspace)
            record = _jsonl_record(annotation, workspace, image_path)
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            records += 1

            if image_path is None:
                skipped_missing_image += 1
                continue

            try:
                image, _metadata = read_supported_image(image_path)
            except Exception:
                skipped_missing_image += 1
                continue

            lines = yolo_pose_lines(annotation)
            if not lines:
                continue

            output_name = f"shenton_{index:05d}.png"
            image.save(images_dir / output_name, format="PNG")
            (labels_dir / f"shenton_{index:05d}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            yolo_images += 1
            yolo_objects += len(lines)

    _write_yolo_data_yaml(yolo_root, split=split)
    return {
        "jsonl_records": records,
        "yolo_images": yolo_images,
        "yolo_objects": yolo_objects,
        "skipped_missing_image": skipped_missing_image,
    }


def yolo_pose_lines(annotation: Annotation) -> list[str]:
    width = max(1, int(annotation.image.width))
    height = max(1, int(annotation.image.height))
    lines: list[str] = []
    curves = annotation.shenton_curves or {}
    for side in ("left", "right"):
        side_curves = curves.get(side, {}) if isinstance(curves.get(side, {}), dict) else {}
        for segment_key, class_id in SEGMENT_CLASSES.items():
            segment = side_curves.get(segment_key, {}) if isinstance(side_curves.get(segment_key, {}), dict) else {}
            points = _clean_points(segment.get("points", []))
            if len(points) < 3:
                continue
            control_points = _resample_polyline(points, CONTROL_POINT_COUNT)
            cx, cy, box_w, box_h = _bbox(control_points, width, height)
            values = [str(class_id), _fmt(cx), _fmt(cy), _fmt(box_w), _fmt(box_h)]
            for point in control_points:
                values.extend([_fmt(_clamp(point["x"] / width)), _fmt(_clamp(point["y"] / height)), "2"])
            lines.append(" ".join(values))
    return lines


def _load_annotation(path: Path) -> Annotation | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return annotation_from_dict(json.load(handle))
    except Exception:
        return None


def _jsonl_record(annotation: Annotation, workspace: Path, image_path: Path | None) -> dict[str, Any]:
    image_rel = ""
    if image_path is not None:
        try:
            image_rel = image_path.resolve().relative_to(workspace.resolve()).as_posix()
        except ValueError:
            image_rel = str(image_path)
    return {
        "filename": annotation.image.filename,
        "image_path": image_rel,
        "image": {
            "width": annotation.image.width,
            "height": annotation.image.height,
            "source_format": annotation.image.source_format,
            "pixel_spacing_row_mm": annotation.image.pixel_spacing_row_mm,
            "pixel_spacing_col_mm": annotation.image.pixel_spacing_col_mm,
            "pixel_spacing_source": annotation.image.pixel_spacing_source,
            "dicom_warnings": list(annotation.image.dicom_warnings or []),
        },
        "shenton_curves": annotation.shenton_curves,
        "shenton_review": annotation.shenton_review,
        "measurements_snapshot": annotation.measurements_snapshot,
    }


def _has_shenton_points(annotation: Annotation) -> bool:
    curves = annotation.shenton_curves or {}
    for side in ("left", "right"):
        side_curves = curves.get(side, {}) if isinstance(curves.get(side, {}), dict) else {}
        for segment_key in SEGMENT_CLASSES:
            segment = side_curves.get(segment_key, {}) if isinstance(side_curves.get(segment_key, {}), dict) else {}
            if _clean_points(segment.get("points", [])):
                return True
    return False


def _clean_points(points: Any) -> list[dict[str, float]]:
    cleaned: list[dict[str, float]] = []
    if not isinstance(points, list):
        return cleaned
    for point in points:
        if not isinstance(point, dict):
            continue
        try:
            cleaned.append({"x": float(point["x"]), "y": float(point["y"])})
        except Exception:
            continue
    return cleaned


def _resample_polyline(points: list[dict[str, float]], count: int) -> list[dict[str, float]]:
    if len(points) <= count:
        return points + [points[-1]] * (count - len(points))
    distances = [0.0]
    for prev, cur in zip(points, points[1:]):
        distances.append(distances[-1] + ((cur["x"] - prev["x"]) ** 2 + (cur["y"] - prev["y"]) ** 2) ** 0.5)
    total = distances[-1]
    if total <= 0:
        return [points[0]] * count
    targets = [total * i / (count - 1) for i in range(count)]
    resampled: list[dict[str, float]] = []
    segment_index = 0
    for target in targets:
        while segment_index < len(distances) - 2 and distances[segment_index + 1] < target:
            segment_index += 1
        start = points[segment_index]
        end = points[segment_index + 1]
        segment_length = distances[segment_index + 1] - distances[segment_index]
        ratio = 0.0 if segment_length <= 0 else (target - distances[segment_index]) / segment_length
        resampled.append(
            {
                "x": start["x"] + (end["x"] - start["x"]) * ratio,
                "y": start["y"] + (end["y"] - start["y"]) * ratio,
            }
        )
    return resampled


def _bbox(points: list[dict[str, float]], width: int, height: int) -> tuple[float, float, float, float]:
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    x1 = _clamp((min(xs) / width) - BOX_PADDING)
    x2 = _clamp((max(xs) / width) + BOX_PADDING)
    y1 = _clamp((min(ys) / height) - BOX_PADDING)
    y2 = _clamp((max(ys) / height) + BOX_PADDING)
    box_w = max(MIN_BOX_SIZE, x2 - x1)
    box_h = max(MIN_BOX_SIZE, y2 - y1)
    cx = _clamp((x1 + x2) / 2)
    cy = _clamp((y1 + y2) / 2)
    return cx, cy, box_w, box_h


def _write_yolo_data_yaml(root: Path, *, split: str) -> None:
    lines = [
        f"path: {root.resolve().as_posix()}",
        f"train: images/{split}",
        f"val: images/{split}",
        "",
        "kpt_shape: [4, 3]",
        "flip_idx: [0, 1, 2, 3]",
        "",
        "names:",
    ]
    for class_id, name in CLASS_NAMES.items():
        lines.append(f"  {class_id}: {name}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _fmt(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        result = export_training_set(args.workspace, args.output, split=args.split, overwrite=args.overwrite)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
