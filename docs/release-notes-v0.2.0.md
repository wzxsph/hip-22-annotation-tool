# Hip22AnnotationTool v0.2.0 Prerelease

Label: non-production ready

This prerelease adds DICOM import, enhanced X-ray preview, reduced default line clutter, Shenton curve collection, ROI/scan-assisted recognition, cached rendering, and measurement/export foundations for research review.

## Highlights

- Supports `.dcm`, `.dicom`, and extensionless DICOM with valid DICOM headers.
- Dynamically renders DICOM to PNG for the browser without creating derived PNG files in the submitted folder.
- Reads PixelSpacing / ImagerPixelSpacing and stores only non-PHI image metadata in annotation JSON.
- Opens in Enhanced display by default, with Original view available for comparison. Enhanced preview uses a hip_demo-style contrast pipeline and does not change coordinates or overwrite source images.
- Caches rendered Enhanced/DICOM display PNGs under local app data so repeated image switching does not recompute the same preview.
- Uses enhanced preprocessing by default for folder-import background auto-detection; Enhanced Detect uses the same path for current-image retries.
- Leaves model-missing landmarks explicitly missing when auto-detection is unavailable or incomplete; template fallback is disabled for new auto-detection output.
- Adds a non-destructive ROI crop tool for cluttered-background images. Current-image auto-detect can retry inside the ROI and maps results back to original image coordinates.
- Adds a four-corner scan-like transform for phone-shot X-rays. Current-image auto-detect can run on the perspective-corrected view and inverse-map results back to original image coordinates.
- Hides default 22-point guide connections by default, with toggles for default lines, manual lines, Shenton curves, measurement lines, and point numbers.
- Adds Shenton curve annotation for left/right obturator upper curve and femoral-neck inner-lower curve, with at least 3 points per segment and no hard upper limit.
- Adds Shenton measurement snapshot: endpoint gap, optional mm distances, and warnings.
- Adds live research-aid measurements from the current 22 points: AI/Tonnis angle, Sharp angle, CE angle, neck-shaft angle, and acetabular depth. Acetabular depth is shown in millimeters when valid DICOM PixelSpacing is available.
- Adds `scripts/export_shenton_training_set.py` for internal JSONL and YOLO pose training export.
- Adds `scripts/prepare_scan_like_images.py` for internal local preparation of scan-like enhanced images from phone-shot folders.

## Important Limitations

- This is not production ready and not a medical device.
- Shenton output is a research/review aid only; it is not a clinical diagnosis or validated threshold.
- The live angle/depth outputs are review aids from the current keypoints. They still require doctor confirmation and formula/threshold review before clinical use.
- DICOM support targets ordinary single-frame CR/DR grayscale images. Unsupported compressed/private formats should produce clear warnings.
- Acetabular depth currently uses the available `teardrop_lower` point as a teardrop-edge proxy because the 22-point schema has no independent teardrop outer-edge landmark.
- The phone-photo scan-like workflow depends on visible film corners; difficult photos may still require project-team cleanup or future automatic perspective correction improvements.
- Do not upload hospital DICOM, private X-rays, raw MTDDH images, or generated local workspaces to git.

## Verification

Passed locally:

```text
uv run pytest
uv run python -m compileall annotation_tool
node --check static/app.js
```

Before attaching a Windows ZIP release asset, also rebuild the CPU package and run `dist\smoke_test.ps1` outside the source tree.
