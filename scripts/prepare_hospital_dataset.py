from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"}


@dataclass(frozen=True)
class PreparedImage:
    source_path: Path
    output_path: Path
    width: int
    height: int
    sha256: str
    sidecar_copied: bool


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        image.load()
        return int(image.width), int(image.height)


def iter_image_candidates(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def unique_output_name(index: int, source: Path, prefix: str) -> str:
    suffix = source.suffix.lower()
    if suffix == ".jpeg":
        suffix = ".jpg"
    return f"{prefix}_{index:04d}{suffix}"


def prepare_dataset(input_dir: Path, output_dir: Path, prefix: str) -> tuple[list[PreparedImage], list[dict[str, str]]]:
    input_dir = input_dir.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"Input directory not found: {input_dir}")
    if input_dir == output_dir or input_dir in output_dir.parents:
        raise ValueError("Output directory must not be inside the input directory.")

    output_dir.mkdir(parents=True, exist_ok=True)
    prepared: list[PreparedImage] = []
    issues: list[dict[str, str]] = []

    for candidate in iter_image_candidates(input_dir):
        try:
            width, height = read_image_size(candidate)
        except Exception as exc:
            issues.append({"source_path": str(candidate), "issue": f"unreadable image: {exc}"})
            continue

        output_name = unique_output_name(len(prepared) + 1, candidate, prefix)
        output_path = output_dir / output_name
        shutil.copy2(candidate, output_path)

        sidecar_copied = False
        source_sidecar = candidate.with_suffix(".txt")
        if source_sidecar.exists() and source_sidecar.is_file():
            shutil.copy2(source_sidecar, output_path.with_suffix(".txt"))
            sidecar_copied = True

        prepared.append(
            PreparedImage(
                source_path=candidate,
                output_path=output_path,
                width=width,
                height=height,
                sha256=sha256_file(candidate),
                sidecar_copied=sidecar_copied,
            )
        )

    return prepared, issues


def write_reports(output_dir: Path, prepared: list[PreparedImage], issues: list[dict[str, str]]) -> None:
    map_path = output_dir / "rename_map.csv"
    with open(map_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "new_filename",
                "original_path",
                "original_filename",
                "width",
                "height",
                "sha256",
                "sidecar_copied",
            ],
        )
        writer.writeheader()
        for item in prepared:
            writer.writerow(
                {
                    "new_filename": item.output_path.name,
                    "original_path": str(item.source_path),
                    "original_filename": item.source_path.name,
                    "width": item.width,
                    "height": item.height,
                    "sha256": item.sha256,
                    "sidecar_copied": "yes" if item.sidecar_copied else "no",
                }
            )

    issues_path = output_dir / "issues.csv"
    with open(issues_path, "w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source_path", "issue"])
        writer.writeheader()
        writer.writerows(issues)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Flatten and rename hospital image folders for Hip22 annotation.")
    parser.add_argument("--input", required=True, type=Path, help="Messy source directory from the hospital.")
    parser.add_argument("--output", required=True, type=Path, help="Clean output directory for annotation.")
    parser.add_argument("--prefix", default="case", help="Output filename prefix, e.g. case_0001.jpg.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    try:
        prepared, issues = prepare_dataset(args.input, args.output, args.prefix)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    write_reports(args.output.expanduser().resolve(), prepared, issues)
    print(f"Prepared {len(prepared)} images into {args.output.expanduser().resolve()}")
    print(f"Rename map: {args.output.expanduser().resolve() / 'rename_map.csv'}")
    print(f"Issues: {args.output.expanduser().resolve() / 'issues.csv'}")
    return 0 if prepared else 1


if __name__ == "__main__":
    raise SystemExit(main())
