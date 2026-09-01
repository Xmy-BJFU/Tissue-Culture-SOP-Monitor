"""朱顶红无菌实训评估台 Web 服务。."""

from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent))

from webui.engine import engine

app = FastAPI(title="朱顶红无菌实训评估台", version="1.0")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")
UPLOAD_DIR = ROOT / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class ControlIn(BaseModel):
    action: str
    source: str | None = None
    student: str | None = None
    hand_mode: str | None = None


class SettingsIn(BaseModel):
    source: str | None = None
    student: str | None = None
    hand_mode: str | None = None
    device: str | None = None
    imgsz: int | None = None
    conf: float | None = None
    overlay: dict | None = None
    thresholds: dict | None = None


@app.get("/", response_class=HTMLResponse)
def index():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.get("/api/state")
def api_state():
    return engine.state()


@app.get("/api/report")
def api_report():
    return engine.report()


@app.post("/api/control")
def api_control(body: ControlIn):
    act = body.action.lower()
    try:
        if act == "load":
            engine.load_models()
        elif act == "start":
            if body.hand_mode in ("glove", "bare"):
                engine.hand_mode = body.hand_mode
            engine.start(source=body.source, student=body.student or "")
        elif act == "stop":
            engine.stop()
            engine._push_event("系统", "实训已停止")
        elif act == "pause":
            engine.pause(True)
        elif act == "resume":
            engine.pause(False)
        elif act == "toggle_pause":
            engine.pause()
        elif act == "reset":
            engine.reset_actions()
        elif act == "snapshot":
            path = engine.snapshot()
            return {"ok": True, "path": path}
        elif act == "hand_mode":
            if body.hand_mode in ("glove", "bare"):
                engine.hand_mode = body.hand_mode
                engine._push_event("系统", f"手模式 → {engine.hand_mode}")
        else:
            return JSONResponse({"ok": False, "error": f"未知动作 {act}"}, status_code=400)
        return {"ok": True, "state": engine.state()}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=500)


@app.post("/api/settings")
def api_settings(body: SettingsIn):
    if body.source is not None:
        engine.source = body.source
    if body.student is not None:
        engine.student = body.student
    if body.hand_mode in ("glove", "bare"):
        engine.hand_mode = body.hand_mode
    if body.device is not None:
        engine.device = body.device
    if body.imgsz is not None:
        engine.imgsz = int(body.imgsz)
    if body.conf is not None:
        engine.conf = float(body.conf)
    if body.overlay:
        engine.overlay.update(body.overlay)
    if body.thresholds:
        engine.apply_thresholds(body.thresholds)
    return {"ok": True, "state": engine.state()}


@app.post("/api/upload")
async def api_upload(file: UploadFile = File(...)):
    suffix = Path(file.filename or "video.mp4").suffix or ".mp4"
    dest = UPLOAD_DIR / f"clip{suffix}"
    data = await file.read()
    dest.write_bytes(data)
    engine.source = str(dest)
    return {"ok": True, "path": str(dest)}


@app.get("/api/stream")
def api_stream():
    boundary = "frame"

    def gen():
        while True:
            jpeg = engine.latest_jpeg or b""
            yield (
                b"--" + boundary.encode() + b"\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n"
            )
            time.sleep(0.033)

    return StreamingResponse(gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")
