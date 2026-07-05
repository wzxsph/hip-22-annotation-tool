from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

from annotation_tool.image_io import read_supported_image
from annotation_tool.measurements import compute_measurements
from annotation_tool.schema import Annotation, annotation_from_dict
from annotation_tool.storage import find_image_path


CLASS_NAMES = {
    0: "obturator_upper_curve",
    1: "femoral_neck_inner_lower_curve",
}
SEGMENT_CLASSES = {
    "obturator_upper_curve": 0,
    "femoral_neck_inner_lower_curve": 1,
}
ROI_KEYPOINTS = (
    "teardrop_lower",
    "femoral_neck_axis_center",
    "obturator_upper",
    "femoral_neck_inner_lower",
)
DEFAULT_BAND_WIDTH = 7
DEFAULT_PADDING_FRACTION = 0.22
MIN_ROI_SIZE = 64


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Shenton curve annotations as ROI YOLO segmentation data.")
    parser.add_argument("--workspace", required=True, type=Path, help="Hip22 dataset folder that contains annotations/.")
    parser.add_argument("--output", required=True, type=Path, help="Output folder for shenton metadata and YOLO seg data.")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="YOLO split name to write.")
    parser.add_argument("--band-width", default=DEFAULT_BAND_WIDTH, type=int, help="Polyline mask width in ROI pixels.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output folder first if it already exists.")
    return parser.parse_args(argv)


def export_training_set(
    workspace: Path,
    output: Path,
    *,
    split: str = "train",
    band_width: int = DEFAULT_BAND_WIDTH,
    overwrite: bool = False,
) -> dict[str, int]:
    workspace = workspace.expanduser().resolve()
    output = output.expanduser().resolve()
    annotations_dir = workspace / "annotations"
    if not annotations_dir.exists():
        raise ValueError(f"Annotation directory not found: {annotations_dir}")
    if output.exists() and overwrite:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    metadata_path = output / "shenton_seg_records.jsonl"
    yolo_root = output / "yolo_seg"
    images_dir = yolo_root / "images" / split
    labels_dir = yolo_root / "labels" / split
    maps_dir = output / "roi_maps" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    maps_dir.mkdir(parents=True, exist_ok=True)

    records = 0
    roi_images = 0
    yolo_objects = 0
    skipped_missing_image = 0
    skipped_without_curves = 0
    skipped_without_roi = 0

    with open(metadata_path, "w", encoding="utf-8") as jsonl:
        for index, annotation_path in enumerate(sorted(annotations_dir.glob("*.json")), start=1):
            annotation = _load_annotation(annotation_path)
            if annotation is None or not _has_shenton_points(annotation):
                skipped_without_curves += 1
                continue

            image_path = find_image_path(annotation.image.filename, workspace)
            if image_path is None:
                skipped_missing_image += 1
                continue
            try:
                image, _metadata = read_supported_image(image_path)
            except Exception:
                skipped_missing_image += 1
                continue

            measurements = compute_measurements(annotation)
            annotation.measurements_snapshot = {
                **(annotation.measurements_snapshot or {}),
                **measurements,
            }

            image_records: list[dict[str, Any]] = []
            for side in ("left", "right"):
                roi = shenton_side_roi(annotation, side)
                if roi is None:
                    skipped_without_roi += 1
                    continue
                labels = yolo_seg_lines_for_side(annotation, side, roi, band_width=band_width)
                if not labels:
                    skipped_without_curves += 1
                    continue
                output_stem = f"shenton_{index:05d}_{side}"
                crop = image.crop((roi["x"], roi["y"], roi["x"] + roi["width"], roi["y"] + roi["height"]))
                crop.save(images_dir / f"{output_stem}.png", format="PNG")
                (labels_dir / f"{output_stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
                map_record = _roi_map_record(annotation, workspace, image_path, side, roi, output_stem)
                (maps_dir / f"{output_stem}.json").write_text(
                    json.dumps(map_record, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                image_records.append(map_record)
                roi_images += 1
                yolo_objects += len(labels)

            if image_records:
                jsonl.write(json.dumps({"filename": annotation.image.filename, "rois": image_records}, ensure_ascii=False) + "\n")
                records += 1

    _write_yolo_data_yaml(yolo_root, split=split)
    _write_training_readme(output)
    return {
        "jsonl_records": records,
        "roi_images": roi_images,
        "yolo_objects": yolo_objects,
        "skipped_missing_image": skipped_missing_image,
        "skipped_without_curves": skipped_without_curves,
        "skipped_without_roi": skipped_without_roi,
    }


def shenton_side_roi(annotation: Annotation, side: str, *, padding_fraction: float = DEFAULT_PADDING_FRACTION) -> dict[str, int] | None:
    points = _side_reference_points(annotation, side)
    if not points:
        return None
    width = max(1, int(annotation.image.width))
    height = max(1, int(annotation.image.height))
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)
    span = max(x2 - x1, y2 - y1, float(MIN_ROI_SIZE))
    pad = max(24.0, span * padding_fraction)
    x1 = max(0.0, x1 - pad)
    y1 = max(0.0, y1 - pad)
    x2 = min(float(width), x2 + pad)
    y2 = min(float(height), y2 + pad)
    if x2 - x1 < MIN_ROI_SIZE:
        center = (x1 + x2) / 2
        x1 = max(0.0, center - MIN_ROI_SIZE / 2)
        x2 = min(float(width), x1 + MIN_ROI_SIZE)
    if y2 - y1 < MIN_ROI_SIZE:
        center = (y1 + y2) / 2
        y1 = max(0.0, center - MIN_ROI_SIZE / 2)
        y2 = min(float(height), y1 + MIN_ROI_SIZE)
    x = int(round(x1))
    y = int(round(y1))
    roi_width = max(1, min(width - x, int(round(x2 - x1))))
    roi_height = max(1, min(height - y, int(round(y2 - y1))))
    return {"x": x, "y": y, "width": roi_width, "height": roi_height}


def yolo_seg_lines_for_side(annotation: Annotation, side: str, roi: dict[str, int], *, band_width: int = DEFAULT_BAND_WIDTH) -> list[str]:
    lines: list[str] = []
    curves = annotation.shenton_curves or {}
    side_curves = curves.get(side, {}) if isinstance(curves.get(side, {}), dict) else {}
    for segment_key, class_id in SEGMENT_CLASSES.items():
        segment = side_curves.get(segment_key, {}) if isinstance(side_curves.get(segment_key, {}), dict) else {}
        points = _clean_points(segment.get("points", []))
        if len(points) < 3:
            continue
        roi_points = [{"x": point["x"] - roi["x"], "y": point["y"] - roi["y"]} for point in points]
        polygons = polyline_to_band_polygons(roi_points, roi["width"], roi["height"], band_width=band_width)
        for polygon in polygons:
            if len(polygon) < 3:
                continue
            values = [str(class_id)]
            for point in polygon:
                values.extend([_fmt(_clamp(point["x"] / roi["width"])), _fmt(_clamp(point["y"] / roi["height"]))])
            lines.append(" ".join(values))
    return lines


def polyline_to_band_polygons(
    points: list[dict[str, float]],
    width: int,
    height: int,
    *,
    band_width: int = DEFAULT_BAND_WIDTH,
) -> list[list[dict[str, float]]]:
    width = max(1, int(width))
    height = max(1, int(height))
    clean = [
        {
            "x": _clamp(float(point["x"]), 0.0, float(width - 1)),
            "y": _clamp(float(point["y"]), 0.0, float(height - 1)),
        }
        for point in points
        if _is_finite_point(point)
    ]
    if len(clean) < 2:
        return []
    mask = Image.new("L", (width, height), 0)
    draw = ImageDraw.Draw(mask)
    xy = [(point["x"], point["y"]) for point in clean]
    draw.line(xy, fill=255, width=max(1, int(band_width)), joint="curve")
    radius = max(1, int(band_width) // 2)
    for x, y in xy:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=255)
    array = np.array(mask)
    contours, _hierarchy = cv2.findContours(array, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons: list[list[dict[str, float]]] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 4:
            continue
        epsilon = max(1.0, 0.01 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True)
        polygon = [{"x": float(point[0][0]), "y": float(point[0][1])} for point in approx]
        if len(polygon) >= 3:
            polygons.append(polygon)
    return sorted(polygons, key=len, reverse=True)


def _side_reference_points(annotation: Annotation, side: str) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for name in ROI_KEYPOINTS:
        keypoint = annotation.keypoints.get(f"{side}_{name}")
        if keypoint is not None and keypoint.visible and keypoint.x is not None and keypoint.y is not None:
            points.append({"x": float(keypoint.x), "y": float(keypoint.y)})
    curves = annotation.shenton_curves or {}
    side_curves = curves.get(side, {}) if isinstance(curves.get(side, {}), dict) else {}
    for segment_key in SEGMENT_CLASSES:
        segment = side_curves.get(segment_key, {}) if isinstance(side_curves.get(segment_key, {}), dict) else {}
        points.extend(_clean_points(segment.get("points", [])))
    return [point for point in points if _is_finite_point(point)]


def _load_annotation(path: Path) -> Annotation | None:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return annotation_from_dict(json.load(handle))
    except Exception:
        return None


def _roi_map_record(
    annotation: Annotation,
    workspace: Path,
    image_path: Path | None,
    side: str,
    roi: dict[str, int],
    output_stem: str,
) -> dict[str, Any]:
    image_rel = ""
    if image_path is not None:
        try:
            image_rel = image_path.resolve().relative_to(workspace.resolve()).as_posix()
        except ValueError:
            image_rel = str(image_path)
    review = annotation.shenton_review.get(side, {}) if isinstance(annotation.shenton_review, dict) else {}
    return {
        "id": output_stem,
        "filename": annotation.image.filename,
        "image_path": image_rel,
        "side": side,
        "roi": roi,
        "roi_to_image": {"dx": roi["x"], "dy": roi["y"]},
        "image": {
            "width": annotation.image.width,
            "height": annotation.image.height,
            "source_format": annotation.image.source_format,
            "pixel_spacing_row_mm": annotation.image.pixel_spacing_row_mm,
            "pixel_spacing_col_mm": annotation.image.pixel_spacing_col_mm,
        },
        "shenton_review": {
            "status": review.get("status", "not_reviewed"),
            "updated_at": review.get("updated_at"),
            "annotator": review.get("annotator"),
        },
        "measurements_snapshot": (annotation.measurements_snapshot or {}).get("shenton", {}).get(side),
        "classes": CLASS_NAMES,
    }


def _has_shenton_points(annotation: Annotation) -> bool:
    curves = annotation.shenton_curves or {}
    for side in ("left", "right"):
        side_curves = curves.get(side, {}) if isinstance(curves.get(side, {}), dict) else {}
        for segment_key in SEGMENT_CLASSES:
            segment = side_curves.get(segment_key, {}) if isinstance(side_curves.get(segment_key, {}), dict) else {}
            if len(_clean_points(segment.get("points", []))) >= 3:
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
            candidate = {"x": float(point["x"]), "y": float(point["y"])}
        except Exception:
            continue
        if _is_finite_point(candidate):
            cleaned.append(candidate)
    return cleaned


def _is_finite_point(point: Any) -> bool:
    if not isinstance(point, dict):
        return False
    try:
        return bool(np.isfinite(float(point["x"])) and np.isfinite(float(point["y"])))
    except Exception:
        return False


def _write_yolo_data_yaml(root: Path, *, split: str) -> None:
    lines = [
        f"path: {root.resolve().as_posix()}",
        f"train: images/{split}",
        f"val: images/{split}",
        "",
        "names:",
    ]
    for class_id, name in CLASS_NAMES.items():
        lines.append(f"  {class_id}: {name}")
    (root / "data.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_training_readme(output: Path) -> None:
    text = """# Shenton YOLO Seg Dataset

This dataset exports one image per hip-side ROI. Labels use two narrow-band segmentation classes:

- `obturator_upper_curve`
- `femoral_neck_inner_lower_curve`

Doctor continuity status is stored in `shenton_seg_records.jsonl` and `roi_maps/`; it is evaluation metadata, not a segmentation class. Legacy `shenton_adjustments.extension_intersection` is intentionally ignored.

Suggested training command:

```bash
yolo segment train model=yolo26n-seg.pt data=yolo_seg/data.yaml imgsz=512 epochs=150 batch=auto
```
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def _clamp(value: float, lower: float = 0.0, upper: float = 1.0) -> float:
    return max(lower, min(upper, value))


def _fmt(value: float) -> str:
    text = f"{value:.6f}".rstrip("0").rstrip(".")
    return text if text else "0"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        result = export_training_set(
            args.workspace,
            args.output,
            split=args.split,
            band_width=args.band_width,
            overwrite=args.overwrite,
        )
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
