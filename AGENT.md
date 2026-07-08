# AGENT.md — Hip 24-Point Annotation Tool

This repository is the standalone public annotation tool. Keep it clean, small, and safe to publish.

## Project Boundary

- This repo contains only the FastAPI + Canvas annotation app, tests, docs, screenshots, and the bundled `models/yolo11n-best.pt` weight.
- Do not import or copy code from `reference/hip_demo`, `reference/retuve-yolo-plugin`, or the parent training project.
- The v0.2 image-enhancement path may reproduce the agreed hip_demo-style processing idea, but do not vendor the reference project or copy unrelated code/weights.
- Do not bundle `reference/retuve-yolo-plugin` segmentation weights in this repository or release package. Use it only as research context unless licensing and clinical scope are explicitly resolved.
- Do not add hospital private data, raw patient data, MTDDH original images, training folders, local workspaces, or generated annotation batches.
- Generated Windows ZIPs belong in GitHub Release assets, not in git history.

## Runtime Model

- Auto-initialization uses `annotation_tool/heuristics.py`, which is now the standalone side11 YOLO inference adapter.
- Default weight path: `models/yolo11n-best.pt`.
- Users may override the model with `HIP22_MODEL_PATH`.
- Device selection uses `HIP22_DEVICE` or `HIP22_MODEL_DEVICE`; unset means CUDA first, then CPU.
- If the model or `ultralytics` is unavailable, keep the app usable and leave model-missing points explicitly missing with warnings. Never raise a 500 just because inference is unavailable.
- The current-image `Auto Detect` UI action should preserve existing manual points and may use partial model results as review candidates.
- `Enhanced Detect` must preserve manual points and record that enhanced preprocessing was used.
- Template fallback is disabled for current annotations. If legacy `template_guess` points are encountered, keep them visually distinct from `pose11_side`, `estimated`, and `manual`.

## Git LFS And Download ZIP

- `models/yolo11n-best.pt` is tracked with Git LFS through `.gitattributes`.
- A normal `git clone` works only when Git LFS is installed and `git lfs pull` has completed.
- GitHub's source-code ZIP is not a reliable delivery method for the model file; it may omit the real model binary or include only a pointer.
- Before packaging or testing inference, verify that `models/yolo11n-best.pt` is a large binary file and does not begin with `version https://git-lfs.github.com/spec/v1`.
- For hospital users, point them to the Windows Release ZIP, which should include the model and runtime dependencies.

## Annotation Contract

- Always keep 24 keypoint keys: `left/right × #1-#12`.
- Completion uses 20 required keypoints: #1-#9 and #12 on both sides. #10 and #11 are optional by default and must not block keypoint completion.
- #12 (`femoral_neck_axis_proximal`) is inferred from the same-side #3 (`femoral_head_center`) and #7 (`femoral_neck_axis_center`) midpoint unless #12 has `source="manual"`. Never infer left #12 from right-side points or vice versa, and never allow duplicate #12 definitions per side.
- `left_*` means image-left, `right_*` means image-right. Do not perform anatomical side swapping.
- Missing points are explicit: `x=null`, `y=null`, `visible=false`, `visibility=0`, `source="missing"`.
- Missingness is determined by visibility and coordinates, not by `source`.
- `source="estimated"` is used for inferred points such as #12. `source="template_guess"` may appear in legacy data only and must remain visually distinct from `pose11_side` model output and `manual` doctor edits.
- Existing `annotations/<stem>.json` or same-folder `<stem>.txt` must load before auto-detection and must not be overwritten by background inference.
- DICOM metadata saved in annotations must be non-PHI only: source format, pixel spacing, spacing source, dimensions, and warnings. Do not store patient name, patient ID, accession number, birth date, institution, or free-text clinical identifiers.
- `roi_crop` is a non-destructive original-image-coordinate rectangle. Enhanced display, DICOM rendering, and ROI-based inference must map all saved points back to original image pixel coordinates.
- `scan_transform` is a non-destructive four-corner perspective transform for phone-shot X-rays. Recognition may run on the warped scan-like view, but saved keypoints must be inverse-mapped back to original image pixel coordinates.
- Shenton curves are research/review-aid data. Do not present `continuous_candidate` as a clinical diagnosis or validated threshold.
- Keep default guide connections available but visually restrained; doctors can toggle them when needed, and the main view should avoid occluding bone boundaries.

## Storage Format Notes

- Main annotation files live at `annotations/<image_stem>.json`; same-folder YOLO Pose labels live at `<image_stem>.txt`.
- Workspace bookkeeping lives in `manifest.json`, `data.yaml`, `splits/train_val_split.json`, and generated progress files named `HIP22_STATUS_DONE_<n>_TODO_<m>.txt`, `HIP22_status_report.html`, `HIP22_status_report.csv`, and `HIP22_SUBMISSION_README.txt`.
- Annotation JSON may contain `image`, `keypoints`, `connections`, `roi_crop`, `scan_transform`, `shenton_curves`, `shenton_review`, `measurements_snapshot`, `auto_initialization`, and `review`.
- `measurements_snapshot` may include Shenton endpoint gap, Shenton forward-extension gap, AI/Tonnis angle, Sharp angle, CE angle, neck-shaft angle, and acetabular depth. Use DICOM PixelSpacing for mm output when available, and keep all formulas labeled as review aids.
- `auto_initialization` should include model source, attempts, warnings, preprocessing label, ROI or scan transform used for retry, original model visible count, and template fallback metadata when applicable.
- Enhanced/DICOM display PNGs should be cached under local user data, not inside the submitted workspace or git history.
- `scripts/prepare_scan_like_images.py` is an internal data-organization tool for phone-shot images. Its outputs are derived local files; do not commit them unless explicit permission and de-identification are confirmed.
- Do not commit source demo images or local annotation JSON/TXT used to derive the template. Only the normalized numeric template may live in the repository.

## Public Repository Hygiene

Do not commit:

- `.venv/`, `.pytest_cache/`, `__pycache__/`
- `images/`, `annotations/`, `labels/`, `splits/`
- `manifest.json`, `tool-settings.json`, `data.yaml`
- `retuve-data/`, `backups/`, `prelabel-report.json`
- `demo_picture/`, `open-hip-dysplasia-data/`, raw MTDDH images, or hospital/private image folders
- `build/`, `dist/`, PyInstaller work folders, generated ZIPs, or smoke-test runtime output

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
node --check static/app.js
find . -maxdepth 3 \( -name '.venv' -o -name 'manifest.json' -o -name 'tool-settings.json' -o -name 'annotations' -o -name 'images' -o -name 'labels' \) -print
git lfs ls-files
git diff --cached --name-only -- '*.jpg' '*.jpeg' '*.png' '*.bmp' '*.tif' '*.tiff' '*.webp'
```

For Windows delivery, also run the generated `dist/smoke_test.ps1` outside the source tree when practical, then upload only the ZIP as a GitHub Release asset.
