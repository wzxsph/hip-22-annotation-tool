from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from scripts.export_shenton_training_set import DEFAULT_BAND_WIDTH, export_training_set


DEFAULT_OUTPUT = Path("/home/samsong/Desktop/hip-xray-ai/datasets/derived/shenton-seg-v1")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ROI YOLO segmentation data for Shenton curve training.")
    parser.add_argument("--workspace", required=True, type=Path, help="Hip22 annotated image folder.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, type=Path, help="Output dataset folder.")
    parser.add_argument("--split", default="train", choices=["train", "val"], help="YOLO split name to write.")
    parser.add_argument("--band-width", default=DEFAULT_BAND_WIDTH, type=int, help="Narrow-band mask width in ROI pixels.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output folder before writing.")
    return parser.parse_args(argv)


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
    print(f"YOLO data: {(args.output / 'yolo_seg' / 'data.yaml').expanduser().resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
