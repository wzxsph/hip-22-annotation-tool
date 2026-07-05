# Windows CPU Package

This project ships to hospitals as a CPU-only Windows ZIP. The app starts a local FastAPI server and opens the user's browser at `127.0.0.1`.

## Build Requirements

- 64-bit Windows build machine.
- Python 3.10, 3.11, or 3.12. Python 3.13/3.14 is intentionally rejected for release builds.
- Git LFS model file must be present: `models/yolo11n-best.pt` should be about 5.6 MB, not a text pointer file.

## Build

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\build_windows_cpu.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\package_zip.ps1 -SkipBuild
```

Output:

```text
dist\Hip22AnnotationTool\Hip22AnnotationTool.exe
dist\Hip22AnnotationTool-v<version>-win64-cpu.zip
dist\smoke_test.ps1
```

## Verify

```powershell
powershell -ExecutionPolicy Bypass -File .\dist\smoke_test.ps1
```

The smoke test extracts the ZIP, starts the exe with browser auto-open disabled, checks the HTTP endpoints, imports one demo image, runs model-assisted initialization, saves, and verifies JSON plus YOLO sidecar outputs.

For v0.2.0, also verify one synthetic or de-identified DICOM if available: import the folder, open the DICOM through the browser UI, confirm the image renders, and confirm annotation JSON stores pixel spacing without patient identifiers.

## Runtime Data

Packaged runtime settings and logs are stored under:

```text
%LOCALAPPDATA%\Hip22AnnotationTool
```

After a user imports a folder, annotations are saved into that selected folder so the whole folder can be returned to the team.
