from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_DATA = Path("/home/samsong/Desktop/hip-xray-ai/datasets/derived/shenton-seg-v1/yolo_seg/data.yaml")
DEFAULT_MODEL = "yolo26n-seg.pt"
FALLBACK_MODEL = "yolo11n-seg.pt"


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Shenton two-curve YOLO segmentation model.")
    parser.add_argument("--data", default=DEFAULT_DATA, type=Path, help="YOLO segmentation data.yaml.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Initial Ultralytics segmentation weights.")
    parser.add_argument("--fallback-model", default=FALLBACK_MODEL, help="Fallback model to print if YOLO26 is unavailable.")
    parser.add_argument("--imgsz", default=512, type=int)
    parser.add_argument("--epochs", default=150, type=int)
    parser.add_argument("--batch", default="auto")
    parser.add_argument("--project", default="/home/samsong/Desktop/hip-xray-ai/datasets/derived/model-zoo/shenton-seg", type=Path)
    parser.add_argument("--name", default="yolo26n-seg-v1")
    parser.add_argument("--dry-run", action="store_true", help="Print the command without running training.")
    return parser.parse_args(argv)


def build_command(args: argparse.Namespace) -> list[str]:
    return [
        "yolo",
        "segment",
        "train",
        f"model={args.model}",
        f"data={args.data.expanduser().resolve()}",
        f"imgsz={args.imgsz}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"project={args.project.expanduser().resolve()}",
        f"name={args.name}",
        "degrees=5",
        "scale=0.15",
        "translate=0.05",
        "shear=0",
        "perspective=0",
        "fliplr=0",
        "mosaic=0",
        "mixup=0",
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if not args.dry_run and not args.data.expanduser().exists():
        print(f"ERROR: data.yaml not found: {args.data}", file=sys.stderr)
        return 2
    command = build_command(args)
    print(" ".join(command))
    if args.dry_run:
        return 0
    try:
        return subprocess.run(command, check=False).returncode
    except FileNotFoundError:
        print(
            f"ERROR: 'yolo' command not found. Install Ultralytics or try model={args.fallback_model} after setup.",
            file=sys.stderr,
        )
        return 127


if __name__ == "__main__":
    raise SystemExit(main())
