# Hip22AnnotationTool v0.3.2

Label: non-production ready

This patch release fixes a Windows-specific path compatibility issue in trash and restore API responses.

## Changes

- Normalize API response paths to use forward slashes on Windows.
- Keeps restored/deleted image paths stable for the browser UI and packaged Windows release.

## Verification

Passed locally on Windows:

```text
python -m pytest -q
python -m compileall annotation_tool
node --check static/app.js
powershell -ExecutionPolicy Bypass -File .\dist\smoke_test.ps1
```
