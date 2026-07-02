# AGENT.md — Hip 22-Point Annotation Tool

This repository is the standalone public annotation tool. Keep it clean, small, and safe to publish.

## Project Boundary

- This repo contains only the FastAPI + Canvas annotation app, tests, docs, screenshots, and the bundled `models/yolo11n-best.pt` weight.
- Do not import or copy code from `reference/hip_demo`, `reference/retuve-yolo-plugin`, or the parent training project.
- Do not add hospital private data, raw patient data, MTDDH original images, training folders, local workspaces, or generated annotation batches.

## Runtime Model

- Auto-initialization uses `annotation_tool/heuristics.py`, which is now the standalone side11 YOLO inference adapter.
- Default weight path: `models/yolo11n-best.pt`.
- Users may override the model with `HIP22_MODEL_PATH`.
- Device selection uses `HIP22_DEVICE` or `HIP22_MODEL_DEVICE`; unset means CUDA first, then CPU.
- If the model or `ultralytics` is unavailable, return a blank 22-point template with warnings. Never raise a 500 just because inference is unavailable.

## Annotation Contract

- Always keep 22 keypoint keys: `left/right × #1-#11`.
- `left_*` means image-left, `right_*` means image-right. Do not perform anatomical side swapping.
- Missing points are explicit: `x=null`, `y=null`, `visible=false`, `visibility=0`, `source="missing"`.
- Missingness is determined by visibility and coordinates, not by `source`.
- Existing `annotations/<stem>.json` or same-folder `<stem>.txt` must load before auto-detection and must not be overwritten by background inference.

## Public Repository Hygiene

Do not commit:

- `.venv/`, `.pytest_cache/`, `__pycache__/`
- `images/`, `annotations/`, `labels/`, `splits/`
- `manifest.json`, `tool-settings.json`, `data.yaml`
- `retuve-data/`, `backups/`, `prelabel-report.json`
- raw MTDDH images or hospital/private image folders

The model file is intentionally tracked with Git LFS through `.gitattributes`.

## License and Docs

- Repository license: AGPL-3.0.
- Keep `MODEL_CARD.md` and `THIRD_PARTY_NOTICES.md` aligned with model changes.
- README screenshots may use MTDDH-derived examples only as rendered screenshots, with CC BY 4.0 attribution.
- Do not claim clinical validation or diagnosis capability. State that auto points require review and correction.

## Verification

Before publishing:

```bash
uv run pytest
uv run python -m compileall annotation_tool
find . -maxdepth 3 \( -name '.venv' -o -name 'manifest.json' -o -name 'tool-settings.json' -o -name 'annotations' -o -name 'images' -o -name 'labels' \) -print
git lfs ls-files
```
