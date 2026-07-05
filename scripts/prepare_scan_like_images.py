from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from PIL import Image

from annotation_tool.image_io import is_supported_image_path, read_supported_image
from annotation_tool.image_processing import enhance_xray_image
from annotation_tool.scan_like import detect_scan_quad, warp_scan_like_image


def prepare_scan_like_dataset(
    input_root: Path,
    output_root: Path,
    *,
    corners_json: Path | None = None,
    overwrite: bool = False,
    output_format: str = "png",
) -> dict[str, Any]:
    input_root = input_root.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    if not input_root.exists() or not input_root.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_root}")
    if output_root.exists() and any(output_root.iterdir()) and not overwrite:
        raise FileExistsError(f"Output folder is not empty: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    corners_lookup = load_corners_lookup(corners_json) if corners_json else {}
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    image_paths = [path for path in sorted(input_root.rglob("*")) if path.is_file() and is_supported_image_path(path)]

    for index, image_path in enumerate(image_paths, start=1):
        relative = image_path.relative_to(input_root)
        try:
            image, metadata = read_supported_image(image_path)
            corners = lookup_corners(corners_lookup, relative, image_path)
            mode = "manual_corners"
            warnings: list[str] = []
            if corners is not None:
                corners = corners_to_array(corners)
            if corners is None:
                corners = detect_scan_quad(image)
                mode = "auto_detected_quad" if corners is not None else "full_image_fallback"
            if corners is not None:
                warp = warp_scan_like_image(image, corners)
                output_image = enhance_xray_image(warp.image)
                scan_used = warp.used
            else:
                output_image = enhance_xray_image(image)
                scan_used = {
                    "enabled": False,
                    "mode": "full_image_fallback",
                    "output_width": image.width,
                    "output_height": image.height,
                    "corners": [],
                }
                warnings.append("No quadrilateral was detected; wrote enhanced full image.")
            output_name = f"scanlike_{index:05d}_{safe_stem(image_path.stem)}.{output_format.lower()}"
            output_path = output_root / output_name
            save_image(output_image, output_path, output_format)
            records.append(
                {
                    "source_path": relative.as_posix(),
                    "output_path": output_name,
                    "source_format": metadata.get("source_format", "image"),
                    "mode": mode,
                    "width": output_image.width,
                    "height": output_image.height,
                    "scan_transform": scan_used,
                    "warnings": warnings,
                }
            )
        except Exception as exc:  # pragma: no cover - defensive batch guard
            failures.append({"source_path": relative.as_posix(), "error": str(exc)})

    write_mapping_csv(output_root / "scan_like_mapping.csv", records)
    report = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "total_candidates": len(image_paths),
        "written": len(records),
        "failed": len(failures),
        "failures": failures,
        "records": records,
    }
    (output_root / "scan_like_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def load_corners_lookup(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return data
    if isinstance(data, list):
        result: dict[str, Any] = {}
        for item in data:
            if not isinstance(item, dict):
                continue
            key = item.get("source_path") or item.get("filename") or item.get("image")
            corners = item.get("corners")
            if key and corners:
                result[str(key)] = corners
        return result
    return {}


def lookup_corners(lookup: dict[str, Any], relative: Path, image_path: Path):
    for key in (relative.as_posix(), str(relative), image_path.name, image_path.stem):
        corners = lookup.get(key)
        if corners is not None:
            return corners
    return None


def corners_to_array(corners: Any):
    import numpy as np

    points: list[tuple[float, float]] = []
    if not isinstance(corners, list) or len(corners) != 4:
        return None
    for item in corners:
        if isinstance(item, dict):
            x = item.get("x")
            y = item.get("y")
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            x, y = item[0], item[1]
        else:
            return None
        try:
            points.append((float(x), float(y)))
        except (TypeError, ValueError):
            return None
    return np.asarray(points, dtype=np.float32)


def safe_stem(stem: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return cleaned[:48] or "image"


def save_image(image: Image.Image, output_path: Path, output_format: str) -> None:
    output_format = output_format.lower()
    if output_format in {"jpg", "jpeg"}:
        image.convert("RGB").save(output_path, format="JPEG", quality=95)
    elif output_format == "png":
        image.save(output_path, format="PNG")
    else:
        raise ValueError("output_format must be png, jpg, or jpeg")


def write_mapping_csv(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_path", "output_path", "source_format", "mode", "width", "height", "warnings"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "source_path": record["source_path"],
                    "output_path": record["output_path"],
                    "source_format": record["source_format"],
                    "mode": record["mode"],
                    "width": record["width"],
                    "height": record["height"],
                    "warnings": "; ".join(record.get("warnings", [])),
                }
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare scan-like enhanced images from phone-shot X-ray folders.")
    parser.add_argument("--input", required=True, type=Path, help="Raw input folder.")
    parser.add_argument("--output", required=True, type=Path, help="Clean output folder.")
    parser.add_argument("--corners-json", type=Path, help="Optional JSON mapping source files to four scan corners.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output folder.")
    parser.add_argument("--format", choices=["png", "jpg", "jpeg"], default="png", help="Output image format.")
    args = parser.parse_args()
    report = prepare_scan_like_dataset(
        args.input,
        args.output,
        corners_json=args.corners_json,
        overwrite=args.overwrite,
        output_format=args.format,
    )
    print(f"Wrote {report['written']} scan-like images to {args.output}")
    if report["failed"]:
        print(f"Failed: {report['failed']}")


if __name__ == "__main__":
    main()
