"""Local Whisper word-level captions + Submagic-style ASS overlay (Captions, 3s Hook Banner, Watermark)."""

from __future__ import annotations

import re
import textwrap
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Optional

LayoutMode = Literal["facecam_top", "facecam_bottom", "no_facecam"]

_MODEL_LOCK = threading.Lock()
_WHISPER_MODEL = None


@dataclass
class WordTiming:
    word: str
    start: float
    end: float


def _get_whisper_model(progress_cb: Optional[Callable[[str], None]] = None):
    global _WHISPER_MODEL
    with _MODEL_LOCK:
        if _WHISPER_MODEL is None:
            if progress_cb:
                progress_cb("Loading local Whisper caption model (one-time download ~140 MB)...")
            from faster_whisper import WhisperModel
            _WHISPER_MODEL = WhisperModel("base", device="cpu", compute_type="int8")
        return _WHISPER_MODEL


def transcribe_words(
    video_path: Path,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> list[WordTiming]:
    """Transcribe a clip locally with word-level timestamps using faster-whisper (0 API calls)."""
    model = _get_whisper_model(progress_cb=progress_cb)
    if progress_cb:
        progress_cb("Generating word-by-word captions locally...")

    segments, _ = model.transcribe(
        str(video_path),
        beam_size=1,
        vad_filter=True,
        word_timestamps=True,
    )
    words: list[WordTiming] = []
    for seg in segments:
        if seg.words:
            for w in seg.words:
                txt = (w.word or "").strip()
                if not txt:
                    continue
                words.append(WordTiming(
                    word=txt,
                    start=max(0.0, float(w.start)),
                    end=max(float(w.start) + 0.08, float(w.end)),
                ))
        else:
            # Fallback if a segment lacks word timestamps
            parts = (seg.text or "").strip().split()
            if parts:
                dur = max(0.2, float(seg.end) - float(seg.start))
                step = dur / len(parts)
                for idx, p in enumerate(parts):
                    words.append(WordTiming(
                        word=p,
                        start=float(seg.start) + idx * step,
                        end=float(seg.start) + (idx + 1) * step,
                    ))
    return words


def _fmt_ass_time(sec: float) -> str:
    sec = max(0.0, float(sec))
    h = int(sec // 3600)
    m = int((sec % 3600) // 60)
    s = int(sec % 60)
    cs = int(round((sec - int(sec)) * 100))
    if cs >= 100:
        cs = 99
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"


def _esc_ass(text: str) -> str:
    return (
        text.replace("\\", "")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\n", "\\N")
    )


def _group_words(words: list[WordTiming], max_words: int = 3) -> list[list[WordTiming]]:
    """Group words into punchy 2-3 word Submagic-style bursts."""
    chunks: list[list[WordTiming]] = []
    cur: list[WordTiming] = []
    for w in words:
        if cur:
            gap = w.start - cur[-1].end
            cur_chars = sum(len(x.word) for x in cur) + len(w.word)
            ends_sentence = cur[-1].word.endswith((".", "!", "?", ","))
            if len(cur) >= max_words or cur_chars > 20 or gap > 0.45 or ends_sentence:
                chunks.append(cur)
                cur = [w]
                continue
        cur.append(w)
    if cur:
        chunks.append(cur)
    return chunks


def build_overlay_ass(
    out_ass: Path,
    duration_sec: float,
    layout: LayoutMode = "no_facecam",
    words: Optional[list[WordTiming]] = None,
    hook_text: str = "",
    watermark: str = "",
) -> Optional[Path]:
    """
    Create a single ASS overlay script containing any combination of:
    - Animated word-by-word Submagic-style captions
    - 3-second viral hook title banner
    - Subtle channel handle watermark
    Returns None if no overlays are active.
    """
    has_captions = bool(words)
    clean_hook = re.sub(r"\s+", " ", (hook_text or "").strip())
    clean_wm = re.sub(r"\s+", " ", (watermark or "").strip())

    if not has_captions and not clean_hook and not clean_wm:
        return None

    # Smart vertical placement so overlays never cover the streamer's facecam or platform buttons
    if layout == "facecam_bottom":
        cap_margin_v = 710   # Above bottom facecam (facecam starts at y=1240 => 680px from bottom)
        hook_margin_v = 115
    elif layout == "facecam_top":
        cap_margin_v = 340   # Lower gameplay zone
        hook_margin_v = 655  # Cleanly at the split seam below top facecam
    else:
        cap_margin_v = 350   # Sweet spot above Shorts/TikTok bottom UI
        hook_margin_v = 135

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: 1080
PlayResY: 1920
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Caption,Arial,74,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,7,3,2,60,60,{cap_margin_v},1
Style: Hook,Arial,54,&H00FFFFFF,&H00FFFFFF,&H55000000,&H55000000,-1,0,0,0,100,100,0,0,3,16,0,8,70,70,{hook_margin_v},1
Style: Watermark,Arial,30,&H66FFFFFF,&H66FFFFFF,&H99000000,&H00000000,-1,0,0,0,100,100,0,0,1,2,0,9,30,36,56,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    events: list[str] = []

    # Layer 0: Word-by-word animated captions (active word pops in bright yellow)
    if words:
        chunks = _group_words(words, max_words=3)
        for chunk in chunks:
            for active_i, active_w in enumerate(chunk):
                ev_start = active_w.start
                if active_i + 1 < len(chunk):
                    ev_end = max(active_w.end, chunk[active_i + 1].start)
                else:
                    ev_end = active_w.end + 0.08
                if ev_end <= ev_start + 0.02:
                    ev_end = ev_start + 0.08

                rendered_tokens: list[str] = []
                for idx, tok in enumerate(chunk):
                    w_txt = _esc_ass(tok.word.upper())
                    if idx == active_i:
                        # Bright viral yellow (#facc15 in BGR = &H0015CCFA&) + subtle pop scale
                        rendered_tokens.append(
                            r"{\c&H0015CCFA&\fscx108\fscy108}" + w_txt + r"{\c&H00FFFFFF&\fscx100\fscy100}"
                        )
                    else:
                        rendered_tokens.append(w_txt)

                line_text = " ".join(rendered_tokens)
                events.append(
                    f"Dialogue: 0,{_fmt_ass_time(ev_start)},{_fmt_ass_time(ev_end)},Caption,,0,0,0,,{line_text}"
                )

    # Layer 1: 3-second viral hook banner at top
    if clean_hook:
        wrapped = "\\N".join(textwrap.wrap(clean_hook.upper(), width=24)[:2])
        hook_end = min(max(2.5, duration_sec), 3.3)
        events.append(
            f"Dialogue: 1,0:00:00.00,{_fmt_ass_time(hook_end)},Hook,,0,0,0,,{{\\fad(160,320)}}{_esc_ass(wrapped)}"
        )

    # Layer 2: Channel watermark
    if clean_wm:
        end_t = max(1.0, duration_sec + 0.5)
        events.append(
            f"Dialogue: 2,0:00:00.00,{_fmt_ass_time(end_t)},Watermark,,0,0,0,,{_esc_ass(clean_wm)}"
        )

    out_ass.parent.mkdir(parents=True, exist_ok=True)
    out_ass.write_text(header + "\n".join(events) + "\n", encoding="utf-8")
    return out_ass


def ffmpeg_ass_filter_arg(ass_path: Path) -> str:
    """Format a Windows/POSIX path safely for FFmpeg's ass= filter inside filter_complex."""
    p = ass_path.resolve().as_posix()
    # Escape colon in Windows drive letter (C:/... -> C\:/...) and single quotes
    p = p.replace(":", "\\:").replace("'", "\\'")
    return f"ass='{p}'"
