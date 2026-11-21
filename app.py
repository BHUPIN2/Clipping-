"""ViralClip — Simple YouTube Viral Vertical Clipper (FastAPI Server)."""

from __future__ import annotations

import os
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Literal, Optional

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import uvicorn
from dotenv import dotenv_values
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from clipper.ai import default_gameplay_box, detect_facecam_box
from clipper.pipeline import ClipPipeline, ClipSettings
from clipper.renderer import CropBox
from clipper.youtube import capture_preview_frame, inspect_youtube_url

STATIC_DIR = BASE_DIR / "static"
CLIPS_DIR = BASE_DIR / "clips"
PREVIEWS_DIR = BASE_DIR / ".work" / "previews"
ENV_FILE = BASE_DIR / ".env"

CLIPS_DIR.mkdir(parents=True, exist_ok=True)
PREVIEWS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ViralClip", version="2.0.0")
pipeline = ClipPipeline(BASE_DIR)

app.mount("/clips", StaticFiles(directory=str(CLIPS_DIR)), name="clips")
app.mount("/previews", StaticFiles(directory=str(PREVIEWS_DIR)), name="previews")


def _load_saved_api_key() -> str:
    if ENV_FILE.exists():
        vals = dotenv_values(ENV_FILE)
        key = (vals.get("GEMINI_API_KEY") or "").strip()
        if key:
            return key
    return (os.environ.get("GEMINI_API_KEY") or "").strip()


def _save_api_key(key: str) -> None:
    key = key.strip()
    ENV_FILE.write_text(f"GEMINI_API_KEY={key}\n", encoding="utf-8")
    os.environ["GEMINI_API_KEY"] = key


class ConfigRequest(BaseModel):
    api_key: str


class BoxModel(BaseModel):
    x: float = Field(ge=0.0, le=1.0)
    y: float = Field(ge=0.0, le=1.0)
    w: float = Field(gt=0.02, le=1.0)
    h: float = Field(gt=0.02, le=1.0)


class InspectRequest(BaseModel):
    url: str
    seek_ratio: float = Field(default=0.25, ge=0.0, le=0.95)
    api_key: Optional[str] = None
    reuse_cache: bool = False


class DetectFacecamRequest(BaseModel):
    preview_filename: str
    api_key: Optional[str] = None


class StartClipRequest(BaseModel):
    url: str
    api_key: Optional[str] = None
    layout: Literal["facecam_top", "facecam_bottom", "no_facecam"] = "facecam_top"
    facecam: Optional[BoxModel] = None
    gameplay: Optional[BoxModel] = None
    lookback_minutes: int = Field(default=30, ge=5, le=180)
    max_clips: int = Field(default=5, ge=1, le=15)
    clip_length: Literal["short", "medium", "long"] = "medium"
    topic_filter: str = ""
    captions: bool = True
    loudnorm: bool = True
    watermark: str = ""
    hook_banner: bool = False
    interval_minutes: int = Field(default=15, ge=5, le=60)


class RerenderClipRequest(BaseModel):
    start_sec: float = Field(ge=0.0)
    end_sec: float = Field(gt=0.0)


def _req_to_settings(req: StartClipRequest) -> ClipSettings:
    fc = req.facecam or BoxModel(x=0.02, y=0.62, w=0.28, h=0.34)
    gp = req.gameplay or BoxModel(x=0.25, y=0.0, w=0.50, h=1.0)
    return ClipSettings(
        layout=req.layout,
        facecam=CropBox(fc.x, fc.y, fc.w, fc.h),
        gameplay=CropBox(gp.x, gp.y, gp.w, gp.h),
        lookback_minutes=req.lookback_minutes,
        max_clips=req.max_clips,
        clip_length=req.clip_length,
        topic_filter=req.topic_filter.strip(),
        captions=req.captions,
        loudnorm=req.loudnorm,
        watermark=req.watermark.strip(),
        hook_banner=req.hook_banner,
    )


@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    html = (STATIC_DIR / "index.html").read_text(encoding="utf-8")
    return HTMLResponse(content=html)


@app.get("/api/config")
def get_config() -> dict:
    key = _load_saved_api_key()
    masked = f"{key[:6]}...{key[-4:]}" if len(key) > 12 else ("set" if key else "")
    return {
        "has_api_key": bool(key),
        "masked_key": masked,
        "clips_dir": str(CLIPS_DIR),
    }


@app.post("/api/config")
def save_config(req: ConfigRequest) -> dict:
    if not req.api_key.strip():
        raise HTTPException(status_code=400, detail="API key cannot be empty.")
    _save_api_key(req.api_key)
    return get_config()


@app.post("/api/inspect")
def inspect_video(req: InspectRequest) -> dict:
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Please paste a YouTube link.")

    api_key = (req.api_key or "").strip() or _load_saved_api_key()
    if req.api_key and req.api_key.strip():
        _save_api_key(req.api_key)

    try:
        video = pipeline.get_cached_video(url) if req.reuse_cache else None
        if video is None:
            video = inspect_youtube_url(url)
            pipeline.cache_video(video)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not load YouTube link: {exc}")

    preview_name = f"{video.video_id}_{int(time.time() * 1000)}.jpg"
    preview_path = PREVIEWS_DIR / preview_name
    try:
        frame_w, frame_h = capture_preview_frame(video, preview_path, seek_ratio=req.seek_ratio)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Could not capture frame from video: {exc}")

    facecam = {"x": 0.02, "y": 0.62, "w": 0.28, "h": 0.34}
    gameplay = default_gameplay_box(frame_w, frame_h)

    return {
        "video_id": video.video_id,
        "title": video.title,
        "channel": video.channel,
        "duration": video.duration,
        "is_live": video.is_live,
        "has_heatmap": len(video.heatmap) > 0,
        "preview_filename": preview_name,
        "preview_url": f"/previews/{preview_name}",
        "frame_width": frame_w,
        "frame_height": frame_h,
        "facecam": facecam,
        "gameplay": gameplay,
        "can_auto_detect": bool(api_key),
    }


@app.post("/api/detect-facecam")
def detect_facecam_endpoint(req: DetectFacecamRequest) -> dict:
    api_key = (req.api_key or "").strip() or _load_saved_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="Enter your Gemini API key first.")
    if req.api_key and req.api_key.strip():
        _save_api_key(req.api_key)

    safe_name = Path(req.preview_filename).name
    preview_path = PREVIEWS_DIR / safe_name
    if not preview_path.exists():
        raise HTTPException(status_code=404, detail="Preview frame not found. Reload the video link.")

    try:
        return detect_facecam_box(api_key, preview_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Gemini facecam detection failed: {exc}")


@app.post("/api/clip")
def start_clipping(req: StartClipRequest) -> dict:
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Please paste a YouTube link.")

    api_key = (req.api_key or "").strip() or _load_saved_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="Please enter your Gemini API key first.")
    if req.api_key and req.api_key.strip():
        _save_api_key(req.api_key)

    job_id = pipeline.start_job(url=url, api_key=api_key, settings=_req_to_settings(req))
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
def get_job(job_id: str) -> dict:
    data = pipeline.get_job(job_id)
    if not data:
        raise HTTPException(status_code=404, detail="Job not found.")
    return data


@app.get("/api/clips")
def list_clips() -> dict:
    return {"clips": pipeline.list_saved_clips(), "clips_dir": str(CLIPS_DIR)}


@app.delete("/api/clips/{clip_id}")
def delete_clip(clip_id: str) -> dict:
    ok = pipeline.delete_saved_clip(clip_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Clip not found.")
    return {"deleted": True}


@app.post("/api/clips/{clip_id}/rerender")
def rerender_clip_endpoint(clip_id: str, req: RerenderClipRequest) -> dict:
    try:
        updated = pipeline.rerender_clip(clip_id, req.start_sec, req.end_sec)
        return {"clip": updated}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Re-render failed: {exc}")


@app.post("/api/watch-live/start")
def start_watch_live(req: StartClipRequest) -> dict:
    url = req.url.strip()
    if not url:
        raise HTTPException(status_code=400, detail="Please paste a live YouTube stream link.")
    api_key = (req.api_key or "").strip() or _load_saved_api_key()
    if not api_key:
        raise HTTPException(status_code=400, detail="Please enter your Gemini API key first.")
    if req.api_key and req.api_key.strip():
        _save_api_key(req.api_key)
    return pipeline.start_live_watch(
        url=url,
        api_key=api_key,
        settings=_req_to_settings(req),
        interval_minutes=req.interval_minutes,
    )


@app.post("/api/watch-live/stop")
def stop_watch_live() -> dict:
    return pipeline.stop_live_watch()


@app.get("/api/watch-live/status")
def get_watch_live_status() -> dict:
    return pipeline.get_live_watch_status()


@app.get("/api/download/{rel_path:path}")
def download_clip(rel_path: str) -> FileResponse:
    target = (CLIPS_DIR / rel_path).resolve()
    if not str(target).startswith(str(CLIPS_DIR.resolve())) or not target.exists():
        raise HTTPException(status_code=404, detail="File not found.")
    return FileResponse(path=str(target), filename=target.name, media_type="video/mp4")


@app.get("/api/export-zip")
def export_all_clips_zip() -> FileResponse:
    import zipfile
    clips = pipeline.list_saved_clips()
    if not clips:
        raise HTTPException(status_code=404, detail="No clips available to export.")
    zip_path = BASE_DIR / ".work" / "viral_clips_all.zip"
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for c in clips:
            rel = c.get("rel_path")
            if rel:
                p = CLIPS_DIR / rel
                if p.exists():
                    zf.write(p, arcname=c.get("filename", p.name))
    return FileResponse(
        path=str(zip_path),
        filename="viral_clips_all.zip",
        media_type="application/zip",
    )


@app.post("/api/open-folder")
def open_clips_folder() -> dict:
    try:
        os.startfile(str(CLIPS_DIR))
        return {"ok": True}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


def _open_browser(port: int) -> None:
    time.sleep(1.0)
    webbrowser.open(f"http://127.0.0.1:{port}")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8765"))
    if os.environ.get("NO_BROWSER") != "1":
        threading.Thread(target=_open_browser, args=(port,), daemon=True).start()
    print(f"\n  ViralClip is running at http://127.0.0.1:{port}\n")
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")
