"""FFmpeg renderer for 9:16 vertical clips: Split-Screen (Facecam + Gameplay) & Full 9:16 AI Live Subject Tracking."""

from __future__ import annotations

import json
import re
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

from clipper.captions import build_overlay_ass, ffmpeg_ass_filter_arg, transcribe_words
from clipper.tracker import track_main_subject_local

LayoutMode = Literal["facecam_top", "facecam_bottom", "no_facecam"]

OUT_WIDTH = 1080
OUT_HEIGHT = 1920
FACECAM_HEIGHT = 680
GAMEPLAY_HEIGHT = 1240  # 680 + 1240 = 1920 (9:16)


@dataclass
class RenderOptions:
    captions: bool = False
    loudnorm: bool = True
    watermark: str = ""
    hook_text: str = ""


@dataclass
class CropBox:
    """Normalized [0..1] rectangle relative to source video frame."""
    x: float
    y: float
    w: float
    h: float

    def to_pixels(self, src_w: int, src_h: int) -> tuple[int, int, int, int]:
        x = max(0.0, min(0.95, float(self.x)))
        y = max(0.0, min(0.95, float(self.y)))
        w = max(0.05, min(1.0 - x, float(self.w)))
        h = max(0.05, min(1.0 - y, float(self.h)))

        px = int(round(x * src_w))
        py = int(round(y * src_h))
        pw = int(round(w * src_w))
        ph = int(round(h * src_h))

        # Ensure even dimensions and valid bounds for H.264 / yuv420p
        px = max(0, min(src_w - 16, px - (px % 2)))
        py = max(0, min(src_h - 16, py - (py % 2)))
        pw = max(16, min(src_w - px, pw - (pw % 2)))
        ph = max(16, min(src_h - py, ph - (ph % 2)))
        return px, py, pw, ph


def probe_video_full(path: Path) -> tuple[int, int, float, bool]:
    """Return (width, height, duration_sec, has_audio) for a video file using ffprobe."""
    cmd = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration:stream=codec_type,width,height,duration",
        "-of", "json",
        str(path),
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(res.stdout or "{}")
    width, height = 1920, 1080
    has_audio = False
    duration = float((data.get("format") or {}).get("duration") or 0.0)
    for stream in data.get("streams", []):
        if stream.get("codec_type") == "video" and stream.get("width") and stream.get("height"):
            width = int(stream["width"])
            height = int(stream["height"])
            if duration <= 0 and stream.get("duration"):
                duration = float(stream["duration"])
        elif stream.get("codec_type") == "audio":
            has_audio = True
    return width, height, duration, has_audio


def probe_video(path: Path) -> tuple[int, int, bool]:
    """Return (width, height, has_audio) for a video file using ffprobe."""
    w, h, _, has_audio = probe_video_full(path)
    return w, h, has_audio


def render_vertical_clip(
    input_path: Path,
    output_path: Path,
    facecam: CropBox,
    gameplay: CropBox,
    layout: LayoutMode = "facecam_top",
    options: Optional[RenderOptions] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    """Render a 1080x1920 vertical split-screen MP4 (Facecam + Gameplay) with optional captions/hook/watermark/loudnorm."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    src_w, src_h, duration, has_audio = probe_video_full(input_path)
    opts = options or RenderOptions()

    fc_x, fc_y, fc_w, fc_h = facecam.to_pixels(src_w, src_h)
    gp_x, gp_y, gp_w, gp_h = gameplay.to_pixels(src_w, src_h)

    fc_filter = (
        f"[0:v]crop={fc_w}:{fc_h}:{fc_x}:{fc_y},"
        f"scale={OUT_WIDTH}:{FACECAM_HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={OUT_WIDTH}:{FACECAM_HEIGHT},setsar=1[fc]"
    )
    gp_filter = (
        f"[0:v]crop={gp_w}:{gp_h}:{gp_x}:{gp_y},"
        f"scale={OUT_WIDTH}:{GAMEPLAY_HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
        f"crop={OUT_WIDTH}:{GAMEPLAY_HEIGHT},setsar=1[gp]"
    )

    if layout == "facecam_bottom":
        stack_filter = "[gp][fc]vstack=inputs=2,setpts=PTS-STARTPTS[vbase]"
    else:
        stack_filter = "[fc][gp]vstack=inputs=2,setpts=PTS-STARTPTS[vbase]"

    base_filter = f"{fc_filter};{gp_filter};{stack_filter}"
    return _finalize_render(
        input_path=input_path,
        output_path=output_path,
        base_filter=base_filter,
        duration=duration,
        has_audio=has_audio,
        layout=layout,
        options=opts,
        progress_cb=progress_cb,
    )


# ---------------------------------------------------------------------------
# Full 9:16 Vertical with AI Live Subject Tracking (No Facecam Mode)
# ---------------------------------------------------------------------------

def detect_scene_cuts(input_path: Path, threshold: float = 0.32) -> list[float]:
    """Detect hard camera cut timestamps using FFmpeg's built-in scene change detector."""
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-i", str(input_path),
        "-vf", f"select='gt(scene,{threshold})',showinfo",
        "-f", "null",
        "-",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    cuts: list[float] = []
    for m in re.finditer(r"pts_time:(\d+(?:\.\d+)?)", proc.stderr or ""):
        t = float(m.group(1))
        if t > 0.25 and (not cuts or t - cuts[-1] > 0.4):
            cuts.append(t)
    return cuts


def sample_tracking_frames(
    input_path: Path,
    duration: float,
    out_dir: Path,
) -> list[tuple[float, Path]]:
    """Extract compact 480px-wide JPEG keyframes across the clip for Gemini Vision subject tracking."""
    out_dir.mkdir(parents=True, exist_ok=True)
    dur = max(1.0, duration)
    # ~1.25 fps (every 0.8s), capped around 54 frames for 60s clips
    sample_fps = min(1.25, max(0.5, 54.0 / dur))

    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-i", str(input_path),
        "-vf", f"fps={sample_fps:.4f},scale=480:-2",
        "-q:v", "6",
        str(out_dir / "f_%04d.jpg"),
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)

    jpgs = sorted(out_dir.glob("f_*.jpg"))
    return [(round(idx / sample_fps, 3), p) for idx, p in enumerate(jpgs)]


def _smooth_shot_values(values: list[float]) -> list[float]:
    """Median denoise + zero-phase exponential smoothing + deadzone to eliminate jitter."""
    n = len(values)
    if n <= 1:
        return list(values)

    # 3-point median filter removes single-frame AI spikes
    med = list(values)
    for i in range(1, n - 1):
        med[i] = sorted((values[i - 1], values[i], values[i + 1]))[1]

    # Forward + backward EMA (zero lag, silky gimbal motion)
    alpha = 0.48
    fwd = [med[0]]
    for v in med[1:]:
        fwd.append(alpha * v + (1.0 - alpha) * fwd[-1])

    bwd = [0.0] * n
    bwd[-1] = fwd[-1]
    for i in range(n - 2, -1, -1):
        bwd[i] = alpha * fwd[i] + (1.0 - alpha) * bwd[i + 1]

    # Small deadzone (~2.2% of frame width) prevents micro-wobble when subject stands still
    deadzone = 0.022
    out = [bwd[0]]
    for v in bwd[1:]:
        if abs(v - out[-1]) < deadzone:
            out.append(out[-1])
        else:
            out.append(v)
    return out


def _build_camera_segments(
    track: list[tuple[float, float]],
    cuts: list[float],
    duration: float,
    src_w: int,
    crop_w: int,
) -> list[tuple[float, float, float, float]]:
    """
    Convert (t, cx_norm) track + scene cuts into piecewise-linear pixel segments
    (t0, t1, x0, x1) with smooth pans inside shots and instant snaps on cuts.
    """
    max_x = float(max(0, src_w - crop_w))
    if not track or max_x <= 1.0:
        return [(0.0, max(1.0, duration + 1.0), max_x / 2.0, max_x / 2.0)]

    def to_px(cx_norm: float) -> float:
        return max(0.0, min(max_x, cx_norm * src_w - crop_w / 2.0))

    # Split track into shots at detected scene cuts or huge sudden jumps
    shots: list[tuple[float, list[tuple[float, float]]]] = []
    cur_shot: list[tuple[float, float]] = [track[0]]
    cur_start_t = 0.0

    for i in range(1, len(track)):
        prev_t, prev_cx = track[i - 1]
        cur_t, cur_cx = track[i]

        between_cuts = [c for c in cuts if prev_t < c <= cur_t]
        big_jump = abs(cur_cx - prev_cx) >= 0.38

        if between_cuts or big_jump:
            cut_t = between_cuts[0] if between_cuts else (prev_t + cur_t) / 2.0
            shots.append((cur_start_t, cur_shot))
            cur_shot = [(cur_t, cur_cx)]
            cur_start_t = cut_t
        else:
            cur_shot.append((cur_t, cur_cx))

    shots.append((cur_start_t, cur_shot))

    segments: list[tuple[float, float, float, float]] = []
    end_limit = max(duration + 1.0, track[-1][0] + 1.0)

    for s_idx, (shot_start, pts) in enumerate(shots):
        shot_end = shots[s_idx + 1][0] if s_idx + 1 < len(shots) else end_limit
        smoothed_cx = _smooth_shot_values([p[1] for p in pts])
        px_pts = [(pts[k][0], to_px(smoothed_cx[k])) for k in range(len(pts))]

        # Lead-in from exact shot_start (instant snap after a cut!)
        if px_pts[0][0] > shot_start + 1e-3:
            segments.append((shot_start, px_pts[0][0], px_pts[0][1], px_pts[0][1]))

        # Smooth pans between keyframes inside the shot
        for k in range(len(px_pts) - 1):
            t0, x0 = px_pts[k]
            t1, x1 = px_pts[k + 1]
            if t1 > t0 + 1e-3:
                segments.append((t0, t1, x0, x1))

        # Hold final position until shot_end
        last_t, last_x = px_pts[-1]
        if shot_end > last_t + 1e-3:
            segments.append((last_t, shot_end, last_x, last_x))

    if not segments:
        return [(0.0, end_limit, max_x / 2.0, max_x / 2.0)]

    # Merge consecutive stationary segments to keep FFmpeg expression super compact
    merged: list[tuple[float, float, float, float]] = [segments[0]]
    for t0, t1, x0, x1 in segments[1:]:
        pt0, pt1, px0, px1 = merged[-1]
        if abs(px0 - px1) < 0.5 and abs(x0 - x1) < 0.5 and abs(px1 - x0) < 0.5 and abs(pt1 - t0) < 0.05:
            merged[-1] = (pt0, t1, px0, x1)
        else:
            merged.append((t0, t1, x0, x1))
    return merged


def _build_balanced_x_expr(segs: list[tuple[float, float, float, float]]) -> str:
    """Compile segments into an O(log N)-depth FFmpeg expression for per-frame crop x(t)."""
    if len(segs) == 1:
        t0, t1, x0, x1 = segs[0]
        dt = max(1e-4, t1 - t0)
        if abs(x1 - x0) < 0.5:
            return f"{x0:.1f}"
        return f"({x0:.1f}+({x1 - x0:.1f})*clip((t-{t0:.3f})/{dt:.3f},0,1))"
    mid = len(segs) // 2
    t_split = segs[mid][0]
    left = _build_balanced_x_expr(segs[:mid])
    right = _build_balanced_x_expr(segs[mid:])
    return f"if(lt(t,{t_split:.3f}),{left},{right})"


def render_tracked_vertical_clip(
    input_path: Path,
    output_path: Path,
    api_key: str = "",
    options: Optional[RenderOptions] = None,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    """
    Render a full-screen 1080x1920 vertical clip with local AI live subject tracking
    (keeps the main subject centered across movement and hard camera cuts, using 0 API calls).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    src_w, src_h, duration, has_audio = probe_video_full(input_path)
    opts = options or RenderOptions()

    # Already vertical source? Just scale/crop cleanly to 1080x1920
    if src_w / max(1, src_h) <= 0.58:
        base_filter = (
            f"[0:v]scale={OUT_WIDTH}:{OUT_HEIGHT}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={OUT_WIDTH}:{OUT_HEIGHT},setpts=PTS-STARTPTS,setsar=1[vbase]"
        )
        return _finalize_render(
            input_path=input_path,
            output_path=output_path,
            base_filter=base_filter,
            duration=duration,
            has_audio=has_audio,
            layout="no_facecam",
            options=opts,
            progress_cb=progress_cb,
        )

    crop_h = src_h - (src_h % 2)
    crop_w = int(round(crop_h * OUT_WIDTH / OUT_HEIGHT))
    crop_w = max(16, min(src_w, crop_w - (crop_w % 2)))

    if progress_cb:
        progress_cb("Live-tracking main subject & detecting camera cuts...")
    track, cuts = track_main_subject_local(
        input_path,
        sample_fps=3.0,
        progress_cb=progress_cb,
    )

    segments = _build_camera_segments(
        track=track,
        cuts=cuts,
        duration=duration,
        src_w=src_w,
        crop_w=crop_w,
    )
    x_expr = _build_balanced_x_expr(segments)

    base_filter = (
        f"[0:v]crop=w={crop_w}:h={crop_h}:x='{x_expr}':y=0,"
        f"scale={OUT_WIDTH}:{OUT_HEIGHT}:flags=lanczos,setpts=PTS-STARTPTS,setsar=1[vbase]"
    )
    return _finalize_render(
        input_path=input_path,
        output_path=output_path,
        base_filter=base_filter,
        duration=duration,
        has_audio=has_audio,
        layout="no_facecam",
        options=opts,
        progress_cb=progress_cb,
    )


def _finalize_render(
    input_path: Path,
    output_path: Path,
    base_filter: str,
    duration: float,
    has_audio: bool,
    layout: LayoutMode,
    options: RenderOptions,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        words = None
        if options.captions and has_audio:
            try:
                words = transcribe_words(input_path, progress_cb=progress_cb)
            except Exception:
                words = None

        ass_path = build_overlay_ass(
            out_ass=td_path / "overlay.ass",
            duration_sec=duration,
            layout=layout,
            words=words,
            hook_text=options.hook_text,
            watermark=options.watermark,
        )

        if ass_path is not None:
            filter_str = f"{base_filter};[vbase]{ffmpeg_ass_filter_arg(ass_path)}[vout]"
        else:
            filter_str = f"{base_filter};[vbase]null[vout]"

        if progress_cb:
            progress_cb("Rendering final 1080x1920 MP4...")

        script_path = td_path / "filtergraph.txt"
        script_path.write_text(filter_str, encoding="utf-8")

        cmd = [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel", "error",
            "-i", str(input_path),
            "-/filter_complex", str(script_path),
            "-map", "[vout]",
        ]
        if has_audio:
            cmd += ["-map", "0:a:0?"]
            af_chain = ["asetpts=PTS-STARTPTS", "aresample=async=1000:first_pts=0"]
            if options.loudnorm:
                af_chain.append("loudnorm=I=-14:TP=-1.5:LRA=11")
            cmd += ["-af", ",".join(af_chain)]
            cmd += ["-c:a", "aac", "-b:a", "192k", "-ar", "48000"]
        else:
            cmd += ["-an"]

        cmd += [
            "-c:v", "libx264",
            "-preset", "fast",
            "-crf", "20",
            "-pix_fmt", "yuv420p",
            "-avoid_negative_ts", "make_zero",
            "-shortest",
            "-movflags", "+faststart",
            str(output_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"FFmpeg render failed: {proc.stderr.strip()}")
    return output_path
