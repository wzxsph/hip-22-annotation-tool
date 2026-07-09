# Hip22AnnotationTool v0.3.3

Label: non-production ready

This patch release improves annotation review ergonomics and cleans up the default anatomical guide lines.

## Changes

- Reduce default anatomical guide connections to the current 11-line set: per-side #1-#3, #8-#9, #12-#7, #5-#6, #1-#2, plus cross-side #2-#2.
- Remove retired legacy default spokes #3-#8, #3-#9, and #3-#7 from default annotations while preserving reviewer-created manual lines.
- Allow drawn Shenton curve control points to be selected and dragged from either the Shenton tool or the selection tool, with panel and measurement updates after adjustment.
- Improve the workspace grid, canvas toolbar, and side/review panel sizing so controls wrap cleanly and avoid clipping on narrower screens.
- Update README, Chinese hospital guide, and demo script language to match the current default original-image view, guide-line set, and Shenton editing behavior.

## Verification

Passed locally on Windows:

```text
python -m pytest -q
python -m compileall annotation_tool
node --check static/app.js
powershell -ExecutionPolicy Bypass -File .\dist\smoke_test.ps1
```
