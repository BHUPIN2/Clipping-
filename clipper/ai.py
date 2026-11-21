"""Gemini AI integration: Facecam detection (Vision) + Viral-Only Moment Detection (Transcript & Audio)."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from google import genai
from google.genai import types
from PIL import Image


PREFERRED_MODELS = [
    "gemini-3.8-flash",
    "gemini-3.6-flash",
    "gemini-3.5-flash",
    "gemini-2.5-flash",
    "gemini-3.5-flash-lite",
    "gemini-3.1-flash-lite",
    "gemini-2.5-flash-lite",
]
_CACHED_MODEL: Optional[str] = None


def _make_client(api_key: str, timeout_ms: int = 120_000) -> genai.Client:
    """Create a Gemini client with explicit timeout and disabled SDK multi-minute auto-retry."""
    return genai.Client(
        api_key=api_key.strip(),
        http_options=types.HttpOptions(
            timeout=timeout_ms,
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )


@dataclass
class ViralMoment:
    title: str
    start_sec: float
    end_sec: float
    viral_score: float
    reason: str
    hashtags: list[str] = field(default_factory=list)


def _format_ts(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def _parse_ts_or_float(val: Any) -> float:
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).strip()
    try:
        return float(s)
    except ValueError:
        pass
    parts = s.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0.0


def _extract_json(text: str) -> Any:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    obj_match = re.search(r"(\{.*\})", text, re.DOTALL)
    arr_match = re.search(r"(\[.*\])", text, re.DOTALL)
    candidates = []
    if obj_match:
        candidates.append((obj_match.start(), obj_match.group(1)))
    if arr_match:
        candidates.append((arr_match.start(), arr_match.group(1)))
    candidates.sort(key=lambda x: x[0])
    for _, snippet in candidates:
        try:
            return json.loads(snippet)
        except json.JSONDecodeError:
            continue
    return {}


def _discover_available_models(client: genai.Client) -> list[str]:
    """Fallback: ask the Gemini API which generateContent-capable Flash/Pro models exist."""
    skip_words = ("embedding", "imagen", "veo", "image", "tts", "omni", "audio", "robotics", "computer")
    try:
        found: list[str] = []
        for m in client.models.list():
            name = (getattr(m, "name", "") or "").replace("models/", "")
            actions = getattr(m, "supported_actions", None) or []
            if "gemini" in name and (not actions or "generateContent" in actions):
                if not any(w in name.lower() for w in skip_words):
                    found.append(name)
        flash_only = sorted([n for n in found if "flash" in n], reverse=True)
        others = sorted([n for n in found if "flash" not in n], reverse=True)
        return flash_only + others
    except Exception:
        return []


def _generate_with_fallback(
    client: genai.Client,
    contents: Any,
    config: types.GenerateContentConfig,
) -> Any:
    global _CACHED_MODEL
    candidates: list[str] = []
    if _CACHED_MODEL:
        candidates.append(_CACHED_MODEL)
    for m in PREFERRED_MODELS:
        if m not in candidates:
            candidates.append(m)

    last_err: Optional[Exception] = None
    tried: set[str] = set()

    skip_now_keys = (
        "not found", "404", "not supported", "deprecated", "no longer available",
        "429", "resource_exhausted", "quota", "limit:",
    )
    busy_keys = ("503", "unavailable", "high demand", "overloaded", "500", "internal", "deadline")

    for phase in range(2):
        for model in list(candidates):
            if model in tried:
                continue
            tried.add(model)

            for attempt in range(2):
                try:
                    resp = client.models.generate_content(
                        model=model,
                        contents=contents,
                        config=config,
                    )
                    _CACHED_MODEL = model
                    return resp
                except Exception as exc:
                    err_str = str(exc).lower()
                    last_err = exc
                    if any(k in err_str for k in skip_now_keys):
                        if _CACHED_MODEL == model:
                            _CACHED_MODEL = None
                        break  # Immediately switch to next model with its own quota
                    if any(k in err_str for k in busy_keys):
                        if attempt < 1:
                            time.sleep(1.2)
                            continue
                        if _CACHED_MODEL == model:
                            _CACHED_MODEL = None
                        break
                    raise
        if phase == 0:
            discovered = [m for m in _discover_available_models(client) if m not in tried]
            candidates.extend(discovered[:3])

    if last_err:
        raise last_err
    raise RuntimeError("No Gemini model available.")


def default_gameplay_box(img_w: int, img_h: int) -> dict[str, float]:
    """Compute a clean centered portrait gameplay crop matching 1080x1240 aspect ratio."""
    aspect_slot = 1080.0 / 1240.0
    src_aspect = (img_w / img_h) if img_h else (16.0 / 9.0)
    # Normalized width that corresponds to 1080:1240 at full frame height
    norm_w = min(0.85, max(0.35, aspect_slot / src_aspect))
    return {
        "x": round((1.0 - norm_w) / 2.0, 4),
        "y": 0.0,
        "w": round(norm_w, 4),
        "h": 1.0,
    }


def detect_facecam_box(api_key: str, image_path: Path) -> dict[str, Any]:
    """Use Gemini Vision to detect the streamer's facecam bounding box on a stream frame."""
    with Image.open(image_path) as img:
        img_w, img_h = img.size

    gameplay = default_gameplay_box(img_w, img_h)
    fallback_facecam = {"x": 0.02, "y": 0.62, "w": 0.28, "h": 0.34}

    if not api_key or not api_key.strip():
        return {"found": False, "facecam": fallback_facecam, "gameplay": gameplay}

    client = _make_client(api_key)
    img_bytes = image_path.read_bytes()

    prompt = (
        "Analyze this gaming livestream screenshot. "
        "Locate the streamer's webcam / facecam overlay box (usually in one of the corners or edges, "
        "showing the real person playing the game). "
        "Return ONLY valid JSON with this exact schema:\n"
        '{"found": true/false, "facecam_box_2d": [ymin, xmin, ymax, xmax]}\n'
        "where ymin, xmin, ymax, xmax are integers from 0 to 1000 representing the tight bounding box "
        "around the entire facecam rectangle (including the streamer's head and shoulders inside the cam box). "
        "If there is no facecam/webcam person visible, set found to false."
    )

    resp = _generate_with_fallback(
        client,
        contents=[
            types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0.1,
            response_mime_type="application/json",
        ),
    )

    data = _extract_json(resp.text or "{}")
    if not isinstance(data, dict):
        if isinstance(data, list) and data and isinstance(data[0], dict):
            data = data[0]
        else:
            data = {}
    found = bool(data.get("found"))
    box = data.get("facecam_box_2d")

    if found and isinstance(box, list) and len(box) == 4:
        ymin, xmin, ymax, xmax = [max(0.0, min(1000.0, float(v))) for v in box]
        if xmax > xmin + 30 and ymax > ymin + 30:
            fc = {
                "x": round(xmin / 1000.0, 4),
                "y": round(ymin / 1000.0, 4),
                "w": round((xmax - xmin) / 1000.0, 4),
                "h": round((ymax - ymin) / 1000.0, 4),
            }
            return {"found": True, "facecam": fc, "gameplay": gameplay}

    return {"found": False, "facecam": fallback_facecam, "gameplay": gameplay}


CLIP_LENGTH_PRESETS: dict[str, tuple[float, float]] = {
    "short": (15.0, 32.0),
    "medium": (20.0, 60.0),
    "long": (45.0, 92.0),
}


def _clean_viral_moments(
    raw_clips: list[dict[str, Any]],
    total_duration: float,
    max_clips: int,
    min_score: float = 8.0,
    clip_length: str = "medium",
) -> list[ViralMoment]:
    """Validate, filter strictly for viral quality, deduplicate, and sort clips."""
    min_len, max_len = CLIP_LENGTH_PRESETS.get(clip_length, (20.0, 60.0))
    parsed: list[ViralMoment] = []
    for item in raw_clips:
        if not isinstance(item, dict):
            continue
        start = _parse_ts_or_float(item.get("start_sec", item.get("start", 0)))
        end = _parse_ts_or_float(item.get("end_sec", item.get("end", 0)))
        score = float(item.get("viral_score", item.get("score", 0)) or 0)
        title = str(item.get("title") or "Viral Moment").strip()[:80]
        reason = str(item.get("reason") or "High-energy viral stream highlight.").strip()[:220]

        if total_duration > 0:
            start = max(0.0, min(total_duration - 5.0, start))
            end = max(start + 5.0, min(total_duration, end))
        if end <= start + 4.0:
            continue

        # Respect chosen clip length preset
        dur = end - start
        if dur < min_len - 3.0 and total_duration > 0:
            pad = (min_len - dur) / 2.0
            start = max(0.0, start - pad)
            end = min(total_duration, end + pad)
        elif dur > max_len + 5.0:
            end = start + max_len

        raw_tags = item.get("hashtags") or []
        if isinstance(raw_tags, str):
            raw_tags = re.split(r"[\s,]+", raw_tags)
        hashtags: list[str] = []
        seen_tags: set[str] = set()
        for t in raw_tags:
            cleaned = re.sub(r"[^a-zA-Z0-9_]", "", str(t).lstrip("#").strip())
            if cleaned and cleaned.lower() not in seen_tags:
                seen_tags.add(cleaned.lower())
                hashtags.append(f"#{cleaned}")
            if len(hashtags) >= 8:
                break
        if not hashtags:
            hashtags = ["#shorts", "#viral", "#clips", "#fyp"]

        parsed.append(ViralMoment(
            title=title,
            start_sec=round(start, 2),
            end_sec=round(end, 2),
            viral_score=round(score, 1),
            reason=reason,
            hashtags=hashtags,
        ))

    # Strict viral-only filter first (>= 8.0); if none pass, keep best available moments
    strict = [c for c in parsed if c.viral_score >= min_score]
    pool = strict if strict else ([c for c in parsed if c.viral_score >= 7.0] or parsed)

    # Sort best score first and remove overlapping clips
    pool.sort(key=lambda c: c.viral_score, reverse=True)
    deduped: list[ViralMoment] = []
    for cand in pool:
        overlaps = any(
            not (cand.end_sec <= kept.start_sec + 3.0 or cand.start_sec >= kept.end_sec - 3.0)
            for kept in deduped
        )
        if not overlaps:
            deduped.append(cand)
        if len(deduped) >= max_clips:
            break

    return deduped


VIRAL_EDITOR_RULES = """You are an elite viral Shorts / TikTok / Reels clip editor.
Your ONLY job is to find the genuinely VIRAL moments — nothing average, nothing filler, nothing boring.

What makes a clip TRULY VIRAL (score 8.0 to 10.0):
- Gaming / livestreams: explosive genuine reaction (screaming, uncontrollable laughter, total shock, rage, hype), crazy gameplay chaos, impossible clutch, hilarious glitch/fail, or iconic one-liner.
- Stand-up comedy / podcasts / normal videos: killer joke setup + huge punchline + big crowd laugh, savage roast/comeback, hilarious crowd work, shocking confession, or unforgettable mic-drop moment.
- Instant 2-second hook: starts RIGHT as the setup/action begins (never 15 seconds of dead air or rambling) and ends cleanly right after the punchline/payoff.
- Ideal length: 18 to 55 seconds (strictly between 15 and 60 seconds).

STRICT RULES:
- ONLY return moments that truly deserve a viral_score >= 8.0 out of 10.
- NEVER pad with mediocre, slow, or generic moments just to hit a number. If the video only has 1 or 2 truly viral moments, return ONLY those 1 or 2.
- Skip intros, sponsor reads, menus, BRB screens, mundane small talk, and slow dead air.
- Timestamps MUST be accurate in seconds (start_sec, end_sec).

Return ONLY valid JSON with this exact schema:
{
  "clips": [
    {
      "title": "Short punchy viral hook title (max 60 chars)",
      "start_sec": 1240.0,
      "end_sec": 1278.0,
      "viral_score": 9.3,
      "reason": "One concise sentence explaining why this moment will go viral",
      "hashtags": ["#shorts", "#viral", "#topic1", "#topic2", "#niche", "#fyp"]
    }
  ]
}"""


def find_viral_moments_from_transcript(
    api_key: str,
    title: str,
    channel: str,
    duration: float,
    transcript_lines: list[dict[str, Any]],
    heatmap: list[dict[str, float]],
    max_clips: int = 6,
    clip_length: str = "medium",
    topic_filter: str = "",
) -> list[ViralMoment]:
    """Review an entire 3-4+ hour stream transcript (+ YouTube heatmap) to extract ONLY viral parts."""
    client = _make_client(api_key)
    min_len, max_len = CLIP_LENGTH_PRESETS.get(clip_length, (20.0, 60.0))

    # Format compact transcript
    formatted_lines = [
        f"[{int(item['start'])}s | {_format_ts(item['start'])}] {item['text']}"
        for item in transcript_lines
    ]
    transcript_text = "\n".join(formatted_lines)

    # Summarize top YouTube "Most Replayed" heatmap peaks if available
    heatmap_section = ""
    if heatmap:
        top_peaks = sorted(heatmap, key=lambda h: h["value"], reverse=True)[:12]
        top_peaks = [p for p in top_peaks if p["value"] >= 0.45]
        if top_peaks:
            peak_lines = [
                f"- {int(p['start_time'])}s to {int(p['end_time'])}s "
                f"({_format_ts(p['start_time'])}) — replay intensity {p['value']:.2f}/1.00"
                for p in sorted(top_peaks, key=lambda x: x["start_time"])
            ]
            heatmap_section = (
                "\nYOUTUBE MOST-REPLAYED VIEWER SPIKES (strong signal for viral moments!):\n"
                + "\n".join(peak_lines)
                + "\n"
            )

    focus_line = (
        f"\nUSER TOPIC FILTER (STRICTLY prioritize moments matching this): {topic_filter.strip()}\n"
        if topic_filter and topic_filter.strip()
        else ""
    )

    prompt = (
        f"{VIRAL_EDITOR_RULES}\n\n"
        f"Stream Title: {title}\n"
        f"Channel: {channel}\n"
        f"Total Duration: {int(duration)} seconds ({_format_ts(duration)})\n"
        f"Target Clip Length: {int(min_len)} to {int(max_len)} seconds per clip.\n"
        f"Maximum clips to return: {max_clips} (return fewer if fewer are truly viral!)\n"
        f"{focus_line}"
        f"{heatmap_section}\n"
        f"FULL TIMESTAMPED TRANSCRIPT (format: [seconds | HH:MM:SS] speech):\n"
        f"{transcript_text}"
    )

    resp = _generate_with_fallback(
        client,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.25,
            response_mime_type="application/json",
        ),
    )

    data = _extract_json(resp.text or "{}")
    raw_clips: list[Any] = []
    if isinstance(data, list):
        raw_clips = data
    elif isinstance(data, dict):
        raw_clips = data.get("clips") or data.get("moments") or data.get("viral_moments") or data.get("highlights") or []
        if not raw_clips:
            for v in data.values():
                if isinstance(v, list) and v and isinstance(v[0], dict):
                    raw_clips = v
                    break

    return _clean_viral_moments(
        raw_clips,
        duration,
        max_clips=max_clips,
        clip_length=clip_length,
    )


def find_viral_moments_from_audio(
    api_key: str,
    title: str,
    channel: str,
    audio_path: Path,
    window_duration: float,
    max_clips: int = 6,
    clip_length: str = "medium",
    topic_filter: str = "",
    progress_cb: Optional[Callable[[str], None]] = None,
) -> list[ViralMoment]:
    """Upload compact stream audio to Gemini and detect ONLY viral parts by listening directly."""
    client = _make_client(api_key)
    min_len, max_len = CLIP_LENGTH_PRESETS.get(clip_length, (20.0, 60.0))

    if progress_cb:
        size_mb = audio_path.stat().st_size / (1024 * 1024)
        progress_cb(f"Uploading {size_mb:.1f} MB stream audio to Gemini AI for viral detection...")

    uploaded = client.files.upload(file=str(audio_path))
    try:
        # Wait for Gemini file processing if needed
        while getattr(uploaded, "state", None) and str(uploaded.state).upper().endswith("PROCESSING"):
            time.sleep(2)
            uploaded = client.files.get(name=uploaded.name)

        if progress_cb:
            progress_cb("Gemini AI is listening to the stream to find ONLY the viral moments...")

        focus_line = (
            f"\nUSER TOPIC FILTER (STRICTLY prioritize moments matching this): {topic_filter.strip()}\n"
            if topic_filter and topic_filter.strip()
            else ""
        )

        prompt = (
            f"{VIRAL_EDITOR_RULES}\n\n"
            f"Stream Title: {title}\n"
            f"Channel: {channel}\n"
            f"Audio Duration: {int(window_duration)} seconds ({_format_ts(window_duration)})\n"
            f"Target Clip Length: {int(min_len)} to {int(max_len)} seconds per clip.\n"
            f"Maximum clips to return: {max_clips} (return fewer if fewer are truly viral!)\n"
            f"{focus_line}\n"
            "Listen carefully to the entire attached stream audio — pay attention to voice volume spikes, "
            "screams, sudden shock, contagious laughter, game explosions/chaos, and hilarious reactions. "
            "Pinpoint the exact start_sec and end_sec (in seconds from 0 to "
            f"{int(window_duration)}) of ONLY the most viral moments."
        )

        resp = _generate_with_fallback(
            client,
            contents=[uploaded, prompt],
            config=types.GenerateContentConfig(
                temperature=0.25,
                response_mime_type="application/json",
            ),
        )
        data = _extract_json(resp.text or "{}")
        raw_clips: list[Any] = []
        if isinstance(data, list):
            raw_clips = data
        elif isinstance(data, dict):
            raw_clips = data.get("clips") or data.get("moments") or data.get("viral_moments") or data.get("highlights") or []
            if not raw_clips:
                for v in data.values():
                    if isinstance(v, list) and v and isinstance(v[0], dict):
                        raw_clips = v
                        break

        return _clean_viral_moments(
            raw_clips,
            window_duration,
            max_clips=max_clips,
            clip_length=clip_length,
        )
    finally:
        try:
            client.files.delete(name=uploaded.name)
        except Exception:
            pass


def track_subject_in_frames(
    api_key: str,
    frames: list[tuple[float, Path]],
    batch_size: int = 28,
) -> list[tuple[float, float]]:
    """
    Use Gemini Vision to track the main subject's horizontal center cx (0.0 to 1.0)
    across sampled clip frames so vertical 9:16 reframing keeps the subject dead-center.
    """
    if not frames:
        return []
    if not api_key or not api_key.strip():
        return [(t, 0.5) for t, _ in frames]

    client = _make_client(api_key)
    results: dict[int, float] = {}

    for start_idx in range(0, len(frames), batch_size):
        batch = frames[start_idx : start_idx + batch_size]
        contents: list[Any] = [
            (
                "You are a professional camera operator reframing a horizontal 16:9 video "
                "into a vertical 9:16 clip (TikTok / YouTube Shorts / Reels).\n"
                "For EVERY numbered frame below, locate the MAIN SUBJECT who should stay "
                "dead-center in the vertical frame (e.g., the stand-up comedian on stage, "
                "active speaker, presenter, performer, or main focal person — ignore background "
                "audience/crowd unless the entire shot is an audience reaction).\n"
                "Return ONLY valid JSON with this exact schema:\n"
                '{"points": [{"i": 0, "cx": 500}, ...]}\n'
                "where `i` is the exact frame index shown above each image, and `cx` is an integer "
                "from 0 (far left edge of the image) to 1000 (far right edge of the image) representing "
                "the horizontal center of the main subject's head/upper body. "
                "If no single person is visible (e.g. title card or empty wide shot), return 500."
            )
        ]

        for offset, (ts, img_path) in enumerate(batch):
            global_idx = start_idx + offset
            contents.append(f"Frame {global_idx} (t={ts:.2f}s):")
            contents.append(types.Part.from_bytes(data=img_path.read_bytes(), mime_type="image/jpeg"))

        try:
            resp = _generate_with_fallback(
                client,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.05,
                    response_mime_type="application/json",
                ),
            )
            data = _extract_json(resp.text or "{}")
            pts: list[Any] = []
            if isinstance(data, list):
                pts = data
            elif isinstance(data, dict):
                pts = data.get("points") or data.get("track") or data.get("frames") or []
                if not pts:
                    for v in data.values():
                        if isinstance(v, list) and v and isinstance(v[0], dict):
                            pts = v
                            break
            for pt in pts:
                if not isinstance(pt, dict):
                    continue
                idx = int(pt.get("i", -1))
                if start_idx <= idx < start_idx + len(batch):
                    cx_val = float(pt.get("cx", 500))
                    # Support either 0..1000 scale or accidental 0..1 scale
                    if 0.0 <= cx_val <= 1.0 and cx_val not in (0.0, 1.0):
                        cx_norm = cx_val
                    else:
                        cx_norm = max(0.0, min(1.0, cx_val / 1000.0))
                    results[idx] = cx_norm
        except Exception:
            # Fallback gracefully for this batch if a transient error occurs
            pass

    track: list[tuple[float, float]] = []
    last_cx = 0.5
    for idx, (ts, _) in enumerate(frames):
        cx = results.get(idx, last_cx)
        last_cx = cx
        track.append((ts, cx))
    return track
