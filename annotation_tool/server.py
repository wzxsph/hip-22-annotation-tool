from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .paths import resource_path
from .routes import router as annotation_router


STATIC_DIR = resource_path("static")

app = FastAPI(title="Hip 24-Point Annotation Tool", version="0.3.4")
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
