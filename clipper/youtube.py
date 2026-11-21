"""YouTube VOD & Live Stream (DVR) inspector, transcript fetcher, and segment downloader."""

from __future__ import annotations

import json
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

import yt_dlp
from PIL import Image


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


@dataclass
class HLSSegment:
    url: str
    duration: float
    start: float
    end: float
    seq: Optional[int] = None


@dataclass
class HLSPlaylist:
    segments: list[HLSSegment]
    target_duration: int = 6
    map_uri: Optional[str] = None
    total_duration: float = 0.0
    media_sequence: int = 0


@dataclass
class VideoInfo:
    video_id: str
    url: str
    title: str
    channel: str
    duration: float
    is_live: bool
    live_status: str
    thumbnail: str
    width: int
    height: int
    heatmap: list[dict[str, float]] = field(default_factory=list)
    raw_info: dict[str, Any] = field(default_factory=dict)


def _ydl_base_opts() -> dict[str, Any]:
    """Base yt-dlp options: do NOT override User-Agent (must match YouTube player client to avoid 403)."""
    return {
        "quiet": True,
        "no_warnings": True,
        "nocheckcertificate": True,
        "js_runtimes": {"node": {}},
        "retries": 4,
        "fragment_retries": 4,
        "extractor_retries": 3,
    }


def inspect_youtube_url(url: str) -> VideoInfo:
    """Fetch metadata for a YouTube video, VOD, or currently running live stream."""
    opts = {
        **_ydl_base_opts(),
        "skip_download": True,
        "extract_flat": False,
    }
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if not info:
            raise ValueError("Could not retrieve video information from YouTube.")
        if "entries" in info and info["entries"]:
            info = info["entries"][0]

    is_live = bool(info.get("is_live")) or info.get("live_status") == "is_live"
    duration = float(info.get("duration") or 0.0)
    heatmap_raw = info.get("heatmap") or []
    heatmap = [
        {
            "start_time": float(h.get("start_time", 0.0)),
            "end_time": float(h.get("end_time", 0.0)),
            "value": float(h.get("value", 0.0)),
        }
        for h in heatmap_raw
        if isinstance(h, dict)
    ]

    return VideoInfo(
        video_id=str(info.get("id") or "video"),
        url=url,
        title=str(info.get("title") or "Untitled YouTube Video"),
        channel=str(info.get("uploader") or info.get("channel") or "Unknown Channel"),
        duration=duration,
        is_live=is_live,
        live_status=str(info.get("live_status") or ("is_live" if is_live else "not_live")),
        thumbnail=str(info.get("thumbnail") or ""),
        width=int(info.get("width") or 1920),
        height=int(info.get("height") or 1080),
        heatmap=heatmap,
        raw_info=info,
    )


def _pick_preview_stream(info: dict[str, Any], is_live: bool) -> tuple[Optional[str], Optional[str]]:
    """Select the fastest direct 360p-480p H.264 stream (url, user_agent) for instant frame grab."""
    formats = info.get("formats") or []
    video_formats = [
        f for f in formats
        if f.get("url") and f.get("vcodec") not in (None, "none")
    ]
    if not video_formats:
        ua = (info.get("http_headers") or {}).get("User-Agent")
        return info.get("url"), ua

    if is_live:
        hls = [
            f for f in video_formats
            if "m3u8" in str(f.get("protocol", "")) or ".m3u8" in str(f.get("url", ""))
        ]
        pool = hls or video_formats
        # Prefer 360p-480p for near-instant live preview frame
        pool.sort(key=lambda f: (
            0 if 360 <= (f.get("height") or 0) <= 480 else (1 if (f.get("height") or 0) <= 720 else 2),
            abs((f.get("height") or 480) - 480),
        ))
        chosen = pool[0]
        ua = (chosen.get("http_headers") or info.get("http_headers") or {}).get("User-Agent")
        return chosen["url"], ua

    # For VODs, prefer direct HTTPS H.264 (avc1) 360p-480p — seeks & decodes in <0.5s even on 4-hr streams!
    def score(f: dict[str, Any]) -> tuple[int, int, int, int]:
        proto = str(f.get("protocol") or "")
        is_direct = 0 if proto in ("https", "http") else 1
        vcodec = str(f.get("vcodec") or "").lower()
        is_h264 = 0 if ("avc1" in vcodec or "h264" in vcodec) else 1
        h = int(f.get("height") or 0)
        in_sweet_spot = 0 if 360 <= h <= 480 else (1 if 240 <= h <= 720 else 2)
        return (is_direct, is_h264, in_sweet_spot, abs(h - 480))

    video_formats.sort(key=score)
    chosen = video_formats[0]
    ua = (chosen.get("http_headers") or info.get("http_headers") or {}).get("User-Agent")
    return chosen["url"], ua


def capture_preview_frame(
    video: VideoInfo,
    out_path: Path,
    seek_ratio: float = 0.25,
) -> tuple[int, int]:
    """Capture an actual frame quickly from the YouTube VOD or live stream."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stream_url, stream_ua = _pick_preview_stream(video.raw_info, video.is_live)

    captured = False
    if stream_url:
        cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
        if not video.is_live and video.duration > 5:
            seek_sec = max(0.0, min(video.duration - 2.0, video.duration * seek_ratio))
            cmd += ["-ss", f"{seek_sec:.2f}"]
        if stream_ua:
            cmd += ["-user_agent", stream_ua]
        cmd += [
            "-i", stream_url,
            "-frames:v", "1",
            "-q:v", "3",
            str(out_path),
        ]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=8)
            if proc.returncode == 0 and out_path.exists() and out_path.stat().st_size > 1000:
                captured = True
        except subprocess.TimeoutExpired:
            captured = False

    # Fallback: download thumbnail if direct stream frame capture failed
    if not captured and video.thumbnail:
        req = urllib.request.Request(video.thumbnail, headers={"User-Agent": USER_AGENT})
        with urllib.request.urlopen(req, timeout=15) as resp:
            out_path.write_bytes(resp.read())
        captured = True

    if not captured or not out_path.exists():
        raise RuntimeError("Could not capture preview frame from the YouTube video.")

    with Image.open(out_path) as img:
        return img.size


def _pick_caption_lang(info: dict[str, Any]) -> tuple[Optional[str], bool]:
    """Return (lang_code, is_auto) preferring English manual > English auto > any manual > any auto."""
    subs = info.get("subtitles") or {}
    auto_subs = info.get("automatic_captions") or {}

    for source, is_auto in ((subs, False), (auto_subs, True)):
        for preferred in ("en", "en-US", "en-GB", "en-orig"):
            if preferred in source:
                return preferred, is_auto
        for k in source.keys():
            if k.startswith("en"):
                return k, is_auto

    for source, is_auto in ((subs, False), (auto_subs, True)):
        for k in source.keys():
            if k != "live_chat":
                return k, is_auto

    return None, False


def _parse_json3_subtitles(data: dict[str, Any]) -> list[dict[str, Any]]:
    events = data.get("events") or []
    raw_items: list[tuple[float, float, str]] = []
    for ev in events:
        segs = ev.get("segs")
        if not segs:
            continue
        text = "".join(s.get("utf8", "") for s in segs).replace("\n", " ").strip()
        text = re.sub(r"\s+", " ", text)
        if not text:
            continue
        start_ms = float(ev.get("tStartMs") or 0.0)
        dur_ms = float(ev.get("dDurationMs") or 2000.0)
        raw_items.append((start_ms / 1000.0, (start_ms + dur_ms) / 1000.0, text))
    return _group_transcript_items(raw_items)


def _parse_vtt_subtitles(vtt_text: str) -> list[dict[str, Any]]:
    time_re = re.compile(
        r"(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})\s*-->\s*(?:(\d+):)?(\d{2}):(\d{2})\.(\d{3})"
    )

    def to_sec(h: Optional[str], m: str, s: str, ms: str) -> float:
        return (int(h) if h else 0) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

    raw_items: list[tuple[float, float, str]] = []
    lines = vtt_text.splitlines()
    i = 0
    last_text = ""
    while i < len(lines):
        m = time_re.search(lines[i])
        if m:
            start = to_sec(m.group(1), m.group(2), m.group(3), m.group(4))
            end = to_sec(m.group(5), m.group(6), m.group(7), m.group(8))
            i += 1
            buf = []
            while i < len(lines) and lines[i].strip():
                cleaned = re.sub(r"<[^>]+>", "", lines[i]).strip()
                if cleaned:
                    buf.append(cleaned)
                i += 1
            text = re.sub(r"\s+", " ", " ".join(buf)).strip()
            if text and text != last_text:
                raw_items.append((start, end, text))
                last_text = text
        else:
            i += 1
    return _group_transcript_items(raw_items)


def _group_transcript_items(
    raw_items: list[tuple[float, float, str]],
    window_sec: float = 12.0,
) -> list[dict[str, Any]]:
    """Group short caption fragments into compact ~12s lines for long-stream Gemini analysis."""
    if not raw_items:
        return []
    grouped: list[dict[str, Any]] = []
    cur_start, cur_end, cur_parts = raw_items[0][0], raw_items[0][1], [raw_items[0][2]]

    for start, end, text in raw_items[1:]:
        if start - cur_start >= window_sec or len(" ".join(cur_parts)) > 260:
            grouped.append({
                "start": round(cur_start, 2),
                "end": round(cur_end, 2),
                "text": " ".join(cur_parts),
            })
            cur_start, cur_end, cur_parts = start, end, [text]
        else:
            # Avoid duplicating rolling VTT lines
            if not cur_parts or text not in cur_parts[-1]:
                cur_parts.append(text)
            cur_end = max(cur_end, end)

    grouped.append({
        "start": round(cur_start, 2),
        "end": round(cur_end, 2),
        "text": " ".join(cur_parts),
    })
    return grouped


def fetch_vod_transcript(video: VideoInfo, temp_dir: Path) -> list[dict[str, Any]]:
    """Download and parse YouTube subtitles/auto-captions for a VOD without downloading video."""
    lang, is_auto = _pick_caption_lang(video.raw_info)
    if not lang:
        return []

    sub_dir = temp_dir / "subs"
    sub_dir.mkdir(parents=True, exist_ok=True)
    opts = {
        **_ydl_base_opts(),
        "skip_download": True,
        "writesubtitles": not is_auto,
        "writeautomaticsub": is_auto,
        "subtitleslangs": [lang],
        "subtitlesformat": "json3/vtt/best",
        "outtmpl": str(sub_dir / "%(id)s.%(ext)s"),
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download([video.url])
    except Exception:
        return []

    for p in sorted(sub_dir.glob("*.json3")):
        try:
            return _parse_json3_subtitles(json.loads(p.read_text(encoding="utf-8", errors="ignore")))
        except Exception:
            pass
    for p in sorted(sub_dir.glob("*.vtt")):
        try:
            return _parse_vtt_subtitles(p.read_text(encoding="utf-8", errors="ignore"))
        except Exception:
            pass
    return []


def download_vod_audio_compact(
    video: VideoInfo,
    out_mp3: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> Path:
    """Download low-bitrate audio for a caption-less VOD and compress to 24kbps mono MP3."""
    out_mp3.parent.mkdir(parents=True, exist_ok=True)
    raw_dir = out_mp3.parent / "raw_audio"
    raw_dir.mkdir(parents=True, exist_ok=True)

    if progress_cb:
        progress_cb("No captions found on YouTube — downloading lightweight audio track for Gemini AI...")

    client_attempts: list[dict[str, Any]] = [
        {},
        {"extractor_args": {"youtube": {"player_client": ["android_vr", "tv", "web_safari"]}}},
        {"extractor_args": {"youtube": {"player_client": ["ios", "mweb", "web"]}}},
    ]

    last_err: Optional[Exception] = None
    for extra in client_attempts:
        for old in raw_dir.iterdir():
            if old.is_file():
                old.unlink(missing_ok=True)
        opts = {
            **_ydl_base_opts(),
            **extra,
            "format": "worstaudio[abr>=24]/worstaudio/bestaudio/best",
            "outtmpl": str(raw_dir / "audio.%(ext)s"),
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([video.url])
            files = [p for p in raw_dir.iterdir() if p.is_file() and p.stat().st_size > 500]
            if files:
                last_err = None
                break
        except Exception as exc:
            last_err = exc

    files = [p for p in raw_dir.iterdir() if p.is_file() and p.stat().st_size > 500]
    if not files:
        raise RuntimeError(f"Failed to download audio stream for analysis: {last_err}")

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", str(files[0]),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "24k",
        str(out_mp3),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Audio compression failed: {proc.stderr.strip()}")
    return out_mp3


def download_vod_clip(
    url: str,
    start_sec: float,
    end_sec: float,
    out_mp4: Path,
) -> Path:
    """Download ONLY a specific [start_sec, end_sec] segment from a YouTube VOD."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    seg_dir = out_mp4.parent / f"seg_{out_mp4.stem}"
    seg_dir.mkdir(parents=True, exist_ok=True)

    # Prefer fast H.264 MP4 <=1080p first, with multi-client fallback if YouTube returns 403
    attempts: list[dict[str, Any]] = [
        {
            "format": (
                "bestvideo[height<=1080][vcodec^=avc1]+bestaudio[ext=m4a]/"
                "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/"
                "bestvideo[height<=1080]+bestaudio/"
                "best[height<=1080]/best"
            ),
        },
        {
            "format": "bestvideo[height<=1080]+bestaudio/best[height<=1080]/best",
            "extractor_args": {"youtube": {"player_client": ["android_vr", "tv", "web_safari"]}},
        },
        {
            "format": "best[height<=1080]/best",
            "extractor_args": {"youtube": {"player_client": ["ios", "mweb", "web"]}},
        },
    ]

    last_err: Optional[Exception] = None
    for extra in attempts:
        for old in seg_dir.iterdir():
            if old.is_file():
                old.unlink(missing_ok=True)

        opts = {
            **_ydl_base_opts(),
            **extra,
            "merge_output_format": "mp4",
            "download_ranges": yt_dlp.utils.download_range_func(None, [(start_sec, end_sec)]),
            "force_keyframes_at_cuts": True,
            "outtmpl": str(seg_dir / "clip.%(ext)s"),
        }
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                ydl.download([url])
            candidates = [p for p in seg_dir.iterdir() if p.is_file() and p.stat().st_size > 1000]
            if candidates:
                last_err = None
                break
        except Exception as exc:
            last_err = exc

    candidates = [p for p in seg_dir.iterdir() if p.is_file() and p.stat().st_size > 1000]
    if not candidates:
        raise RuntimeError(
            f"Failed to download clip segment {start_sec:.1f}s - {end_sec:.1f}s: {last_err}"
        )

    # Prefer merged mp4
    candidates.sort(key=lambda p: (0 if p.suffix.lower() == ".mp4" else 1, -p.stat().st_size))
    downloaded = candidates[0]
    if downloaded != out_mp4:
        if out_mp4.exists():
            out_mp4.unlink()
        downloaded.replace(out_mp4)
    return out_mp4


# ---------------------------------------------------------------------------
# Live Stream DVR (Past Time) HLS Support
# ---------------------------------------------------------------------------

def _fetch_text(url: str) -> str:
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=20) as resp:
        return resp.read().decode("utf-8", errors="replace")


def parse_hls_playlist(m3u8_url: str) -> HLSPlaylist:
    """Parse an HLS media (or master) playlist into timestamped segments."""
    text = _fetch_text(m3u8_url)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    # If master playlist, follow the best stream variant
    if any(ln.startswith("#EXT-X-STREAM-INF") for ln in lines):
        best_url = None
        best_bw = -1
        for i, ln in enumerate(lines):
            if ln.startswith("#EXT-X-STREAM-INF") and i + 1 < len(lines):
                bw_match = re.search(r"BANDWIDTH=(\d+)", ln)
                bw = int(bw_match.group(1)) if bw_match else 0
                if bw >= best_bw:
                    best_bw = bw
                    best_url = urllib.parse.urljoin(m3u8_url, lines[i + 1])
        if best_url:
            return parse_hls_playlist(best_url)

    segments: list[HLSSegment] = []
    target_duration = 6
    map_uri: Optional[str] = None
    cur_dur: Optional[float] = None
    elapsed = 0.0
    media_sequence = 0

    for ln in lines:
        if ln.startswith("#EXT-X-MEDIA-SEQUENCE:"):
            try:
                media_sequence = int(ln.split(":", 1)[1].strip())
            except ValueError:
                pass
        elif ln.startswith("#EXT-X-TARGETDURATION:"):
            try:
                target_duration = int(float(ln.split(":", 1)[1]))
            except ValueError:
                pass
        elif ln.startswith("#EXT-X-MAP:"):
            m = re.search(r'URI="([^"]+)"', ln)
            if m:
                map_uri = urllib.parse.urljoin(m3u8_url, m.group(1))
        elif ln.startswith("#EXTINF:"):
            val = ln.split(":", 1)[1].split(",", 1)[0]
            try:
                cur_dur = float(val)
            except ValueError:
                cur_dur = 5.0
        elif not ln.startswith("#"):
            seg_url = urllib.parse.urljoin(m3u8_url, ln)
            dur = cur_dur if cur_dur is not None else 5.0
            m_sq = re.search(r"/sq/(\d+)", seg_url)
            seq_num = int(m_sq.group(1)) if m_sq else (media_sequence + len(segments))
            segments.append(HLSSegment(
                url=seg_url,
                duration=dur,
                start=elapsed,
                end=elapsed + dur,
                seq=seq_num,
            ))
            elapsed += dur
            cur_dur = None

    return HLSPlaylist(
        segments=segments,
        target_duration=target_duration,
        map_uri=map_uri,
        total_duration=elapsed,
        media_sequence=media_sequence,
    )


def pick_live_hls_urls(video: VideoInfo) -> tuple[Optional[str], Optional[str]]:
    """Return (video_m3u8_url, audio_m3u8_url). audio_m3u8_url may equal video_m3u8_url if muxed."""
    formats = video.raw_info.get("formats") or []
    hls_formats = [
        f for f in formats
        if f.get("url") and (
            "m3u8" in str(f.get("protocol", "")) or ".m3u8" in str(f.get("url", ""))
        )
    ]
    if not hls_formats:
        return None, None

    # Prefer muxed HLS <= 1080p if truly muxed; otherwise pair best video-only + audio-only HLS.
    # Note: YouTube HLS audio formats (233, 234) often have vcodec="none" and acodec=None in yt-dlp.
    muxed = [
        f for f in hls_formats
        if f.get("vcodec") not in (None, "none")
        and f.get("acodec") not in (None, "none")
    ]
    video_only = [
        f for f in hls_formats
        if f.get("vcodec") not in (None, "none")
        and f.get("acodec") in (None, "none")
    ]
    audio_only = [
        f for f in hls_formats
        if f.get("vcodec") == "none"
        and f.get("acodec") != "none"
    ]

    if muxed:
        muxed.sort(key=lambda f: (0 if (f.get("height") or 0) <= 1080 else 1, -(f.get("height") or 0)))
        best_muxed = muxed[0]["url"]
        best_audio = audio_only[-1]["url"] if audio_only else best_muxed
        return best_muxed, best_audio

    v_url = None
    a_url = None
    if video_only:
        video_only.sort(key=lambda f: (0 if (f.get("height") or 0) <= 1080 else 1, -(f.get("height") or 0)))
        v_url = video_only[0]["url"]
    if audio_only:
        audio_only.sort(key=lambda f: (
            f.get("abr") or f.get("tbr") or 0,
            int(f["format_id"]) if str(f.get("format_id", "")).isdigit() else 0,
        ), reverse=True)
        a_url = audio_only[0]["url"]
    return v_url, a_url


def sync_hls_playlists(video_pl: HLSPlaylist, audio_pl: HLSPlaylist) -> None:
    """Synchronize live video and audio HLS playlists to identical common sequence numbers."""
    if not video_pl.segments or not audio_pl.segments:
        return
    v_seqs = {s.seq for s in video_pl.segments if s.seq is not None}
    a_seqs = {s.seq for s in audio_pl.segments if s.seq is not None}
    common = sorted(v_seqs & a_seqs)
    if not common:
        return

    common_set = set(common)
    video_pl.segments = [s for s in video_pl.segments if s.seq in common_set]
    audio_pl.segments = [s for s in audio_pl.segments if s.seq in common_set]

    # Re-calculate aligned elapsed timeline starting identically from 0.0
    v_elapsed = 0.0
    for s in video_pl.segments:
        s.start = v_elapsed
        s.end = v_elapsed + s.duration
        v_elapsed += s.duration
    video_pl.total_duration = v_elapsed

    a_elapsed = 0.0
    for s in audio_pl.segments:
        s.start = a_elapsed
        s.end = a_elapsed + s.duration
        a_elapsed += s.duration
    audio_pl.total_duration = a_elapsed


def write_hls_subset(playlist: HLSPlaylist, segments: list[HLSSegment], out_m3u8: Path) -> Path:
    """Write a local VOD-style HLS playlist containing only the chosen segments."""
    out_m3u8.parent.mkdir(parents=True, exist_ok=True)
    first_seq = segments[0].seq if (segments and segments[0].seq is not None) else 0
    lines = [
        "#EXTM3U",
        "#EXT-X-VERSION:7" if playlist.map_uri else "#EXT-X-VERSION:3",
        f"#EXT-X-TARGETDURATION:{max(1, playlist.target_duration)}",
        f"#EXT-X-MEDIA-SEQUENCE:{first_seq}",
        "#EXT-X-PLAYLIST-TYPE:VOD",
    ]
    if playlist.map_uri:
        lines.append(f'#EXT-X-MAP:URI="{playlist.map_uri}"')
    for seg in segments:
        lines.append(f"#EXTINF:{seg.duration:.3f},")
        lines.append(seg.url)
    lines.append("#EXT-X-ENDLIST")
    out_m3u8.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out_m3u8


def extract_live_past_audio(
    audio_playlist: HLSPlaylist,
    lookback_minutes: int,
    out_mp3: Path,
) -> tuple[Path, float, float]:
    """
    Download the past `lookback_minutes` of a live stream's DVR audio as 24kbps mono MP3.
    Returns (mp3_path, window_start_in_playlist_sec, window_duration_sec).
    """
    if not audio_playlist.segments:
        raise RuntimeError(
            "This live stream does not have DVR/rewind history available to clip from past time."
        )

    target_sec = max(60.0, float(lookback_minutes) * 60.0)
    window_start = max(0.0, audio_playlist.total_duration - target_sec)
    selected = [s for s in audio_playlist.segments if s.end > window_start]
    if not selected:
        selected = audio_playlist.segments[-min(len(audio_playlist.segments), 60):]

    actual_start = selected[0].start
    actual_dur = selected[-1].end - actual_start

    subset_path = out_mp3.parent / "live_audio_subset.m3u8"
    write_hls_subset(audio_playlist, selected, subset_path)

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-protocol_whitelist", "file,http,https,tcp,tls,crypto",
        "-allowed_extensions", "ALL",
        "-allowed_segment_extensions", "ALL",
        "-extension_picky", "0",
        "-seg_max_retry", "3",
        "-i", str(subset_path),
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "24k",
        str(out_mp3),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to download live stream DVR audio: {proc.stderr.strip()}")
    return out_mp3, actual_start, actual_dur


def download_live_dvr_clip(
    video_playlist: HLSPlaylist,
    audio_playlist: Optional[HLSPlaylist],
    abs_start_sec: float,
    abs_end_sec: float,
    out_mp4: Path,
    separate_audio: bool,
) -> Path:
    """Download ONLY the viral clip window from the live stream's HLS DVR segments."""
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    pad_start = max(0.0, abs_start_sec - 2.0)
    pad_end = abs_end_sec + 2.0

    # Pick common sequence range if separate audio
    if separate_audio and audio_playlist and audio_playlist.segments and video_playlist.segments:
        v_by_seq = {s.seq: s for s in video_playlist.segments if s.seq is not None}
        a_by_seq = {s.seq: s for s in audio_playlist.segments if s.seq is not None}
        common = sorted(set(v_by_seq.keys()) & set(a_by_seq.keys()))
        if common:
            chosen_seqs = [sq for sq in common if v_by_seq[sq].end >= pad_start and v_by_seq[sq].start <= pad_end]
            if not chosen_seqs:
                closest = min(common, key=lambda sq: abs(v_by_seq[sq].start - abs_start_sec))
                chosen_seqs = [closest]
            v_segs = [v_by_seq[sq] for sq in chosen_seqs]
            a_segs = [a_by_seq[sq] for sq in chosen_seqs]
        else:
            v_segs = [s for s in video_playlist.segments if s.end >= pad_start and s.start <= pad_end]
            a_segs = [s for s in audio_playlist.segments if s.end >= pad_start and s.start <= pad_end]
    else:
        v_segs = [s for s in video_playlist.segments if s.end >= pad_start and s.start <= pad_end]
        a_segs = []

    if not v_segs:
        raise RuntimeError("Requested live DVR clip timestamps are outside available stream buffer.")

    v_m3u8 = out_mp4.parent / f"{out_mp4.stem}_v.m3u8"
    write_hls_subset(video_playlist, v_segs, v_m3u8)

    trim_offset = max(0.0, abs_start_sec - v_segs[0].start)
    clip_dur = max(5.0, abs_end_sec - abs_start_sec)

    hls_in_opts = [
        "-protocol_whitelist", "file,http,https,tcp,tls,crypto",
        "-allowed_extensions", "ALL",
        "-allowed_segment_extensions", "ALL",
        "-extension_picky", "0",
        "-seg_max_retry", "3",
    ]

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        *hls_in_opts,
        "-i", str(v_m3u8),
    ]

    if separate_audio and a_segs:
        a_m3u8 = out_mp4.parent / f"{out_mp4.stem}_a.m3u8"
        write_hls_subset(audio_playlist, a_segs, a_m3u8)
        cmd += [
            *hls_in_opts,
            "-i", str(a_m3u8),
            "-map", "0:v:0",
            "-map", "1:a:0",
        ]

    cmd += [
        "-ss", f"{trim_offset:.3f}",
        "-t", f"{clip_dur:.3f}",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18",
        "-af", "asetpts=PTS-STARTPTS,aresample=async=1000:first_pts=0",
        "-c:a", "aac", "-b:a", "192k",
        "-avoid_negative_ts", "make_zero",
        "-shortest",
        str(out_mp4),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to download live DVR clip: {proc.stderr.strip()}")
    return out_mp4
