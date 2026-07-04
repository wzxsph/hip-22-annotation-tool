from __future__ import annotations

import os
import socket
import threading
import time
import webbrowser
from contextlib import closing
from multiprocessing import freeze_support

import uvicorn


os.environ.setdefault("HIP22_DEVICE", "cpu")
os.environ.setdefault("HIP22_MODEL_DEVICE", "cpu")


def _port_available(host: str, port: int) -> bool:
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex((host, port)) != 0


def _choose_port(host: str, preferred: int) -> int:
    if _port_available(host, preferred):
        return preferred
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


def _open_browser(url: str) -> None:
    time.sleep(1.0)
    webbrowser.open(url)


def main() -> None:
    freeze_support()

    from annotation_tool.server import app

    host = os.environ.get("HIP22_HOST", "127.0.0.1")
    preferred_port = int(os.environ.get("HIP22_PORT", "8010"))
    port = _choose_port(host, preferred_port)
    url = f"http://{host}:{port}/"

    print("Hip 22 Annotation Tool")
    print(f"Running CPU inference at {url}")
    print("Close this window to stop the tool.")

    if os.environ.get("HIP22_NO_BROWSER", "").strip().lower() not in {"1", "true", "yes"}:
        threading.Thread(target=_open_browser, args=(url,), daemon=True).start()

    uvicorn.run(app, host=host, port=port, log_level=os.environ.get("HIP22_LOG_LEVEL", "info"))


if __name__ == "__main__":
    main()
