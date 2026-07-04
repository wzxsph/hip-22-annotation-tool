# Windows ZIP Distribution

Distribute `dist\Hip22AnnotationTool-v<version>-win64-cpu.zip` to the hospital. The target computer does not need Python, CUDA, or administrator access.

## End-User Steps

1. Unzip the ZIP anywhere.
2. Open the extracted `Hip22AnnotationTool` folder.
3. Double-click `Hip22AnnotationTool.exe`.
4. If the browser does not open, manually visit `http://127.0.0.1:8010/`.
5. Import the image folder, review/drag keypoints, save, then send the whole annotated folder back.
6. Use the right-side progress check and generated `HIP22_status_report.html` to confirm which images are still pending.

## Before Sharing

Run:

```powershell
powershell -ExecutionPolicy Bypass -File .\dist\smoke_test.ps1
```

For formal demos, prefer an MTDDH/public/de-identified image:

```powershell
powershell -ExecutionPolicy Bypass -File .\dist\smoke_test.ps1 -DemoImagePath D:\MTDDH\sample.jpg
```

Do not share the ZIP unless the smoke test passes. In particular, the test must not report `model-unavailable` or `No module named 'matplotlib'`.

## Troubleshooting

- SmartScreen warning: click “More info” and “Run anyway”.
- Program closes immediately: run `Run-Hip22.bat` and send the generated log.
- Browser does not open: manually visit the printed local URL.
- Antivirus quarantine: re-extract the ZIP or allow the internal build after IT review.
- Messy hospital folders: ask the hospital to send the photos first, then use `scripts/prepare_hospital_dataset.py` to flatten and rename them before returning the clean folder for annotation.
