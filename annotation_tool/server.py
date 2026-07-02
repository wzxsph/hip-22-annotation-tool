from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .routes import router as annotation_router


ROOT = Path(__file__).resolve().parents[1]
STATIC_DIR = ROOT / "static"

app = FastAPI(title="Hip 22-Point Annotation Tool", version="0.1.0")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
app.include_router(annotation_router)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/annotation")
def annotation() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
