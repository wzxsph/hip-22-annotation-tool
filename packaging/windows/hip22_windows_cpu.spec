# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules


project_root = Path(SPECPATH).parents[1].resolve()
entrypoint = project_root / "annotation_tool" / "desktop.py"

datas = [
    (str(project_root / "static"), "static"),
    (str(project_root / "models"), "models"),
]
datas += collect_data_files("ultralytics", include_py_files=False)
datas += collect_data_files("matplotlib", include_py_files=False)
datas += collect_data_files("pydicom", include_py_files=False)

binaries = []
binaries += collect_dynamic_libs("torch")
binaries += collect_dynamic_libs("torchvision")
binaries += collect_dynamic_libs("cv2")

hiddenimports = []
hiddenimports += collect_submodules("annotation_tool")
hiddenimports += collect_submodules("matplotlib")
hiddenimports += collect_submodules("multipart")
hiddenimports += collect_submodules("pydicom")
hiddenimports += collect_submodules("ultralytics")
hiddenimports += collect_submodules("uvicorn")
hiddenimports += ["matplotlib.backends.backend_agg"]

excludes = [
    "IPython",
    "jupyter",
    "notebook",
    "pytest",
]

a = Analysis(
    [str(entrypoint)],
    pathex=[str(project_root)],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="Hip22AnnotationTool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="Hip22AnnotationTool",
)
