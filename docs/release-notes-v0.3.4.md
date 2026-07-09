# Hip22AnnotationTool v0.3.4

Label: non-production ready

This patch release updates the model-assisted auto-detection completion workflow so reviewer progress stays aligned with images that still need keypoint or Shenton confirmation.

## Changes

- Keep model-assisted initialization focused on pending work: empty or failed auto annotations are retried, while manual or confirmed annotations remain protected.
- Improve completion-state handling after auto-detect finishes so images that still need required keypoints or Shenton review are not treated as fully done.
- Simplify the browser-side auto-detect polling and completion refresh path so progress, current-image state, and workspace status update consistently after background processing.
- Expand queue and workspace route tests for retry, preservation, and completion edge cases.
- Update README and Chinese hospital guide language for the v0.3.4 workflow.

## Verification

Expected release verification on Windows:

```text
python -m pytest -q
python -m compileall annotation_tool
node --check static/app.js
powershell -ExecutionPolicy Bypass -File .\dist\smoke_test.ps1
```
