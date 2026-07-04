# Windows ZIP Distribution

Distribute `dist\Hip22AnnotationTool-v<version>-win64-cpu.zip` to the hospital. The target computer does not need Python, CUDA, or administrator access.

## End-User Steps

1. Unzip the ZIP anywhere.
2. Open the extracted `Hip22AnnotationTool` folder.
3. Double-click `Hip22AnnotationTool.exe`.
4. If the browser does not open, manually visit `http://127.0.0.1:8010/`.
5. Import the image folder, review/drag keypoints, save, then send the whole annotated folder back.

## Before Sharing

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\dist\smoke_test.ps1
```

Do not share the ZIP unless the smoke test passes. In particular, the test must not report `model-unavailable` or `No module named 'matplotlib'`.

## Troubleshooting

- SmartScreen warning: click “More info” and “Run anyway”.
- Program closes immediately: run `Run-Hip22.bat` and send the generated log.
- Browser does not open: manually visit the printed local URL.
- Antivirus quarantine: re-extract the ZIP or allow the internal build after IT review.
