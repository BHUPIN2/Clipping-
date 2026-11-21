"""Background job orchestrator, clip re-trimmer, and continuous Live Stream Watcher."""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from clipper.ai import (
    ViralMoment,
    find_viral_moments_from_audio,
    find_viral_moments_from_transcript,
)
from clipper.renderer import (
    CropBox,
    LayoutMode,
    RenderOptions,
    render_tracked_vertical_clip,
    render_vertical_clip,
)
from clipper.youtube import (
    VideoInfo,
    download_live_dvr_clip,
    download_vod_audio_compact,
    download_vod_clip,
    extract_live_past_audio,
    fetch_vod_transcript,
    inspect_youtube_url,
    parse_hls_playlist,
    pick_live_hls_urls,
    sync_hls_playlists,
)


def _slugify(text: str, max_len: int = 42) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return (s[:max_len] or "clip").strip("_")


def _fmt_time(sec: float) -> str:
    sec = max(0, int(round(sec)))
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


@dataclass
class ClipSettings:
    layout: LayoutMode = "facecam_top"
    facecam: CropBox = field(default_factory=lambda: CropBox(0.02, 0.62, 0.28, 0.34))
    gameplay: CropBox = field(default_factory=lambda: CropBox(0.25, 0.0, 0.50, 1.0))
    lookback_minutes: int = 30
    max_clips: int = 5
    clip_length: str = "medium"   # short | medium | long
    topic_filter: str = ""
    captions: bool = True
    loudnorm: bool = True
    watermark: str = ""
    hook_banner: bool = False


@dataclass
class RenderedClip:
    id: str
    title: str
    viral_score: float
    reason: str
    start_sec: float
    end_sec: float
    timestamp_label: str
    duration_sec: float
    layout: str
    filename: str
    rel_path: str
    file_size_mb: float
    source_title: str
    source_url: str
    created_at: float
    hashtags: list[str] = field(default_factory=list)
    is_live_source: bool = False
    facecam: dict[str, float] = field(default_factory=lambda: {"x": 0.02, "y": 0.62, "w": 0.28, "h": 0.34})
    gameplay: dict[str, float] = field(default_factory=lambda: {"x": 0.25, "y": 0.0, "w": 0.50, "h": 1.0})
    captions: bool = False
    loudnorm: bool = True
    watermark: str = ""
    hook_banner: bool = False


@dataclass
class JobState:
    job_id: str
    status: str = "queued"  # queued | analyzing | clipping | done | error
    progress: int = 0
    message: str = "Starting..."
    logs: list[str] = field(default_factory=list)
    clips: list[RenderedClip] = field(default_factory=list)
    error: Optional[str] = None
    source_title: str = ""
    is_live: bool = False


@dataclass
class LiveWatchState:
    active: bool = False
    url: str = ""
    title: str = ""
    interval_minutes: int = 15
    scans_completed: int = 0
    clips_created: int = 0
    status_message: str = "Idle"
    next_scan_at: float = 0.0
    last_error: Optional[str] = None


class ClipPipeline:
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.clips_dir = base_dir / "clips"
        self.work_dir = base_dir / ".work"
        self.history_file = self.clips_dir / "history.json"
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self._lock = threading.Lock()
        self._jobs: dict[str, JobState] = {}
        self._video_cache: dict[str, VideoInfo] = {}

        # Continuous live stream watcher state
        self._watch = LiveWatchState()
        self._watch_stop = threading.Event()
        self._watch_clipped_ranges: list[tuple[float, float]] = []

    def cache_video(self, video: VideoInfo) -> None:
        with self._lock:
            self._video_cache[video.url] = video

    def get_cached_video(self, url: str) -> Optional[VideoInfo]:
        with self._lock:
            return self._video_cache.get(url)

    def get_job(self, job_id: str) -> Optional[dict[str, Any]]:
        with self._lock:
            job = self._jobs.get(job_id)
            if not job:
                return None
            return {
                "job_id": job.job_id,
                "status": job.status,
                "progress": job.progress,
                "message": job.message,
                "logs": list(job.logs),
                "clips": [asdict(c) for c in job.clips],
                "error": job.error,
                "source_title": job.source_title,
                "is_live": job.is_live,
            }

    def list_saved_clips(self) -> list[dict[str, Any]]:
        if not self.history_file.exists():
            return []
        try:
            items = json.loads(self.history_file.read_text(encoding="utf-8"))
        except Exception:
            return []
        valid = [c for c in items if (self.clips_dir / c.get("rel_path", "")).exists()]
        return sorted(valid, key=lambda c: c.get("created_at", 0), reverse=True)

    def delete_saved_clip(self, clip_id: str) -> bool:
        items = self.list_saved_clips()
        kept = []
        removed = False
        for c in items:
            if c.get("id") == clip_id:
                p = self.clips_dir / c.get("rel_path", "")
                if p.exists():
                    p.unlink(missing_ok=True)
                removed = True
            else:
                kept.append(c)
        if removed:
            self._write_history(kept)
        return removed

    def _append_history(self, clip: RenderedClip) -> None:
        with self._lock:
            items = self.list_saved_clips()
            items.insert(0, asdict(clip))
            self._write_history(items[:150])

    def _replace_history_item(self, clip_id: str, updated: dict[str, Any]) -> None:
        with self._lock:
            items = self.list_saved_clips()
            new_items = [updated if c.get("id") == clip_id else c for c in items]
            self._write_history(new_items)

    def _write_history(self, items: list[dict[str, Any]]) -> None:
        self.history_file.write_text(json.dumps(items, indent=2), encoding="utf-8")

    def _log(self, job: JobState, msg: str, progress: Optional[int] = None, status: Optional[str] = None) -> None:
        with self._lock:
            job.message = msg
            if progress is not None:
                job.progress = max(0, min(100, int(progress)))
            if status is not None:
                job.status = status
            job.logs.append(msg)

    def start_job(self, url: str, api_key: str, settings: ClipSettings) -> str:
        job_id = uuid.uuid4().hex[:10]
        job = JobState(job_id=job_id, status="analyzing", progress=3, message="Connecting to YouTube...")
        with self._lock:
            self._jobs[job_id] = job

        t = threading.Thread(
            target=self._run_job,
            args=(job, url, api_key, settings),
            daemon=True,
        )
        t.start()
        return job_id

    def _run_job(self, job: JobState, url: str, api_key: str, settings: ClipSettings) -> None:
        temp_dir = self.work_dir / f"job_{job.job_id}"
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            video = self.get_cached_video(url)
            if video is None or video.is_live:
                self._log(job, "Fetching stream metadata from YouTube...", progress=6)
                video = inspect_youtube_url(url)
                self.cache_video(video)

            with self._lock:
                job.source_title = video.title
                job.is_live = video.is_live

            if video.is_live:
                self._run_live_pipeline(job, video, api_key, settings, temp_dir)
            else:
                self._run_vod_pipeline(job, video, api_key, settings, temp_dir)

        except Exception as exc:
            with self._lock:
                job.status = "error"
                job.error = str(exc)
                job.message = f"Error: {exc}"
                job.logs.append(f"ERROR: {exc}")
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _run_vod_pipeline(
        self,
        job: JobState,
        video: VideoInfo,
        api_key: str,
        settings: ClipSettings,
        temp_dir: Path,
    ) -> None:
        self._log(job, f"Loaded VOD: {video.title} ({_fmt_time(video.duration)})", progress=12)
        self._log(job, "Checking YouTube captions & Most Replayed spikes...", progress=18)

        transcript = fetch_vod_transcript(video, temp_dir)
        moments: list[ViralMoment] = []

        if transcript and len(transcript) >= 3:
            self._log(
                job,
                f"Reviewing full transcript ({len(transcript)} blocks) with Gemini AI for ONLY viral moments...",
                progress=28,
            )
            moments = find_viral_moments_from_transcript(
                api_key=api_key,
                title=video.title,
                channel=video.channel,
                duration=video.duration,
                transcript_lines=transcript,
                heatmap=video.heatmap,
                max_clips=settings.max_clips,
                clip_length=settings.clip_length,
                topic_filter=settings.topic_filter,
            )
        else:
            audio_mp3 = temp_dir / "vod_audio.mp3"
            download_vod_audio_compact(
                video,
                audio_mp3,
                progress_cb=lambda m: self._log(job, m, progress=24),
            )
            moments = find_viral_moments_from_audio(
                api_key=api_key,
                title=video.title,
                channel=video.channel,
                audio_path=audio_mp3,
                window_duration=video.duration,
                max_clips=settings.max_clips,
                clip_length=settings.clip_length,
                topic_filter=settings.topic_filter,
                progress_cb=lambda m: self._log(job, m, progress=34),
            )

        if not moments:
            self._log(
                job,
                "Gemini AI reviewed the video and found no genuinely viral moments matching your criteria.",
                progress=100,
                status="done",
            )
            return

        self._log(
            job,
            f"Found {len(moments)} viral moment(s)! Downloading & rendering vertical 9:16 clips...",
            progress=45,
            status="clipping",
        )

        self._render_moments(
            job=job,
            video=video,
            api_key=api_key,
            moments=moments,
            settings=settings,
            temp_dir=temp_dir,
            download_fn=lambda idx, m, raw_mp4: download_vod_clip(video.url, m.start_sec, m.end_sec, raw_mp4),
            time_offset=0.0,
        )

    def _run_live_pipeline(
        self,
        job: JobState,
        video: VideoInfo,
        api_key: str,
        settings: ClipSettings,
        temp_dir: Path,
        skip_ranges: Optional[list[tuple[float, float]]] = None,
    ) -> list[RenderedClip]:
        self._log(
            job,
            f"Live stream detected! Fetching DVR rewind buffer (past {settings.lookback_minutes} min)...",
            progress=12,
        )
        v_url, a_url = pick_live_hls_urls(video)
        if not v_url:
            raise RuntimeError("Could not find an HLS DVR stream for this live broadcast.")

        video_pl = parse_hls_playlist(v_url)
        separate_audio = bool(a_url and a_url != v_url)
        audio_pl = parse_hls_playlist(a_url) if separate_audio else video_pl
        if separate_audio:
            sync_hls_playlists(video_pl, audio_pl)

        avail_min = audio_pl.total_duration / 60.0
        self._log(
            job,
            f"Stream DVR buffer has {avail_min:.1f} min of past video. Pulling past {min(settings.lookback_minutes, max(1, int(avail_min)))} min audio...",
            progress=20,
        )

        past_mp3 = temp_dir / "live_past_audio.mp3"
        _, window_start_sec, window_dur_sec = extract_live_past_audio(
            audio_playlist=audio_pl,
            lookback_minutes=settings.lookback_minutes,
            out_mp3=past_mp3,
        )

        moments = find_viral_moments_from_audio(
            api_key=api_key,
            title=video.title,
            channel=video.channel,
            audio_path=past_mp3,
            window_duration=window_dur_sec,
            max_clips=settings.max_clips,
            clip_length=settings.clip_length,
            topic_filter=settings.topic_filter,
            progress_cb=lambda m: self._log(job, m, progress=32),
        )

        # Deduplicate against previously clipped ranges in Watch Live mode
        if skip_ranges and moments:
            fresh: list[ViralMoment] = []
            for m in moments:
                a_start = window_start_sec + m.start_sec
                a_end = window_start_sec + m.end_sec
                if not any(not (a_end <= r0 + 5 or a_start >= r1 - 5) for r0, r1 in skip_ranges):
                    fresh.append(m)
            moments = fresh

        if not moments:
            self._log(
                job,
                "Gemini AI reviewed the past stream window and found no new viral moments yet.",
                progress=100,
                status="done",
            )
            return []

        self._log(
            job,
            f"Found {len(moments)} viral moment(s) from the running stream! Clipping & rendering...",
            progress=45,
            status="clipping",
        )

        return self._render_moments(
            job=job,
            video=video,
            api_key=api_key,
            moments=moments,
            settings=settings,
            temp_dir=temp_dir,
            download_fn=lambda idx, m, raw_mp4: download_live_dvr_clip(
                video_playlist=video_pl,
                audio_playlist=audio_pl if separate_audio else None,
                abs_start_sec=window_start_sec + m.start_sec,
                abs_end_sec=window_start_sec + m.end_sec,
                out_mp4=raw_mp4,
                separate_audio=separate_audio,
            ),
            time_offset=window_start_sec,
        )

    def _render_moments(
        self,
        job: JobState,
        video: VideoInfo,
        api_key: str,
        moments: list[ViralMoment],
        settings: ClipSettings,
        temp_dir: Path,
        download_fn: Any,
        time_offset: float = 0.0,
    ) -> list[RenderedClip]:
        folder_name = f"{_slugify(video.title, 28)}_{video.video_id[:8]}"
        out_folder = self.clips_dir / folder_name
        out_folder.mkdir(parents=True, exist_ok=True)

        created: list[RenderedClip] = []
        total = len(moments)
        for i, moment in enumerate(moments, start=1):
            base_pct = 45 + int(((i - 1) / total) * 52)
            self._log(
                job,
                f"[{i}/{total}] Downloading viral clip: \"{moment.title}\" ({moment.viral_score}/10)...",
                progress=base_pct,
            )

            raw_mp4 = temp_dir / f"raw_{i}.mp4"
            download_fn(i, moment, raw_mp4)

            clip_id = uuid.uuid4().hex[:8]
            filename = f"{i:02d}_{_slugify(moment.title, 36)}_{clip_id}.mp4"
            final_mp4 = out_folder / filename

            render_opts = RenderOptions(
                captions=settings.captions,
                loudnorm=settings.loudnorm,
                watermark=settings.watermark,
                hook_text=moment.title if settings.hook_banner else "",
            )

            if settings.layout == "no_facecam":
                self._log(
                    job,
                    f"[{i}/{total}] Live-tracking main subject & rendering 9:16...",
                    progress=base_pct + int(20 / total),
                )
                render_tracked_vertical_clip(
                    input_path=raw_mp4,
                    output_path=final_mp4,
                    api_key=api_key,
                    options=render_opts,
                    progress_cb=lambda m, b=base_pct, idx=i: self._log(
                        job, f"[{idx}/{total}] {m}", progress=b + int(32 / total)
                    ),
                )
            else:
                label = "Facecam Top" if settings.layout == "facecam_top" else "Facecam Bottom"
                self._log(
                    job,
                    f"[{i}/{total}] Rendering 9:16 vertical ({label})...",
                    progress=base_pct + int(25 / total),
                )
                render_vertical_clip(
                    input_path=raw_mp4,
                    output_path=final_mp4,
                    facecam=settings.facecam,
                    gameplay=settings.gameplay,
                    layout=settings.layout,
                    options=render_opts,
                    progress_cb=lambda m, b=base_pct, idx=i: self._log(
                        job, f"[{idx}/{total}] {m}", progress=b + int(35 / total)
                    ),
                )

            size_mb = round(final_mp4.stat().st_size / (1024 * 1024), 2)
            abs_start = time_offset + moment.start_sec
            abs_end = time_offset + moment.end_sec
            clip_obj = RenderedClip(
                id=clip_id,
                title=moment.title,
                viral_score=moment.viral_score,
                reason=moment.reason,
                start_sec=round(abs_start, 1),
                end_sec=round(abs_end, 1),
                timestamp_label=f"{_fmt_time(abs_start)} - {_fmt_time(abs_end)}",
                duration_sec=round(moment.end_sec - moment.start_sec, 1),
                layout=settings.layout,
                filename=filename,
                rel_path=f"{folder_name}/{filename}",
                file_size_mb=size_mb,
                source_title=video.title,
                source_url=video.url,
                created_at=time.time(),
                hashtags=moment.hashtags,
                is_live_source=video.is_live,
                facecam=asdict(settings.facecam),
                gameplay=asdict(settings.gameplay),
                captions=settings.captions,
                loudnorm=settings.loudnorm,
                watermark=settings.watermark,
                hook_banner=settings.hook_banner,
            )
            with self._lock:
                job.clips.append(clip_obj)
            created.append(clip_obj)
            self._append_history(clip_obj)

        self._log(
            job,
            f"Done! Created {total} viral vertical clip(s) ready to post.",
            progress=100,
            status="done",
        )
        return created

    # -----------------------------------------------------------------------
    # Feature 4: Fine-Tune Start/End Trim & Re-Render (0 Gemini API calls!)
    # -----------------------------------------------------------------------

    def rerender_clip(self, clip_id: str, new_start: float, new_end: float) -> dict[str, Any]:
        """Re-cut and re-render a saved clip with adjusted start/end seconds."""
        items = self.list_saved_clips()
        target = next((c for c in items if c.get("id") == clip_id), None)
        if not target:
            raise ValueError("Clip not found.")
        if new_end <= new_start + 3.0:
            raise ValueError("End time must be at least 3 seconds after start time.")
        if new_end - new_start > 120.0:
            raise ValueError("Trimmed clip length cannot exceed 120 seconds.")

        temp_dir = self.work_dir / f"retrim_{clip_id}_{int(time.time())}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        try:
            raw_mp4 = temp_dir / "raw_trim.mp4"
            source_url = target["source_url"]
            if target.get("is_live_source"):
                video = inspect_youtube_url(source_url)
                if video.is_live:
                    v_url, a_url = pick_live_hls_urls(video)
                    if not v_url:
                        raise RuntimeError("Live DVR stream is no longer available for re-trimming.")
                    vpl = parse_hls_playlist(v_url)
                    sep = bool(a_url and a_url != v_url)
                    apl = parse_hls_playlist(a_url) if sep else vpl
                    if sep:
                        sync_hls_playlists(vpl, apl)
                    download_live_dvr_clip(vpl, apl if sep else None, new_start, new_end, raw_mp4, sep)
                else:
                    download_vod_clip(source_url, new_start, new_end, raw_mp4)
            else:
                download_vod_clip(source_url, new_start, new_end, raw_mp4)

            old_file = self.clips_dir / target["rel_path"]
            folder = old_file.parent
            new_filename = f"{_slugify(target.get('title', 'clip'), 36)}_{clip_id}_{int(time.time()) % 10000}.mp4"
            new_file = folder / new_filename

            fc_d = target.get("facecam") or {"x": 0.02, "y": 0.62, "w": 0.28, "h": 0.34}
            gp_d = target.get("gameplay") or {"x": 0.25, "y": 0.0, "w": 0.50, "h": 1.0}
            layout: LayoutMode = target.get("layout", "no_facecam")  # type: ignore[assignment]
            opts = RenderOptions(
                captions=bool(target.get("captions", False)),
                loudnorm=bool(target.get("loudnorm", True)),
                watermark=str(target.get("watermark", "")),
                hook_text=str(target.get("title", "")) if target.get("hook_banner") else "",
            )

            if layout == "no_facecam":
                render_tracked_vertical_clip(raw_mp4, new_file, options=opts)
            else:
                render_vertical_clip(
                    raw_mp4,
                    new_file,
                    CropBox(fc_d["x"], fc_d["y"], fc_d["w"], fc_d["h"]),
                    CropBox(gp_d["x"], gp_d["y"], gp_d["w"], gp_d["h"]),
                    layout=layout,
                    options=opts,
                )

            if old_file.exists() and old_file != new_file:
                old_file.unlink(missing_ok=True)

            rel_folder = Path(target["rel_path"]).parent.as_posix()
            updated = {
                **target,
                "start_sec": round(new_start, 1),
                "end_sec": round(new_end, 1),
                "duration_sec": round(new_end - new_start, 1),
                "timestamp_label": f"{_fmt_time(new_start)} - {_fmt_time(new_end)}",
                "filename": new_filename,
                "rel_path": f"{rel_folder}/{new_filename}",
                "file_size_mb": round(new_file.stat().st_size / (1024 * 1024), 2),
                "created_at": time.time(),
            }
            self._replace_history_item(clip_id, updated)
            return updated
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    # -----------------------------------------------------------------------
    # Feature 7: Continuous Live Stream Watcher ("Watch Live Mode")
    # -----------------------------------------------------------------------

    def get_live_watch_status(self) -> dict[str, Any]:
        with self._lock:
            remaining = max(0, int(round(self._watch.next_scan_at - time.time()))) if self._watch.active else 0
            return {
                "active": self._watch.active,
                "url": self._watch.url,
                "title": self._watch.title,
                "interval_minutes": self._watch.interval_minutes,
                "scans_completed": self._watch.scans_completed,
                "clips_created": self._watch.clips_created,
                "status_message": self._watch.status_message,
                "seconds_to_next_scan": remaining,
                "last_error": self._watch.last_error,
            }

    def stop_live_watch(self) -> dict[str, Any]:
        self._watch_stop.set()
        with self._lock:
            self._watch.active = False
            self._watch.status_message = "Stopped"
            self._watch.next_scan_at = 0.0
        return self.get_live_watch_status()

    def start_live_watch(
        self,
        url: str,
        api_key: str,
        settings: ClipSettings,
        interval_minutes: int = 15,
    ) -> dict[str, Any]:
        self.stop_live_watch()
        self._watch_stop = threading.Event()
        self._watch_clipped_ranges = []
        interval_minutes = max(5, min(60, int(interval_minutes)))

        with self._lock:
            self._watch = LiveWatchState(
                active=True,
                url=url,
                title=url,
                interval_minutes=interval_minutes,
                status_message="Starting Watch Live Mode...",
            )

        t = threading.Thread(
            target=self._watch_loop,
            args=(url, api_key, settings, interval_minutes),
            daemon=True,
        )
        t.start()
        return self.get_live_watch_status()

    def _watch_loop(
        self,
        url: str,
        api_key: str,
        settings: ClipSettings,
        interval_minutes: int,
    ) -> None:
        watch_settings = ClipSettings(
            layout=settings.layout,
            facecam=settings.facecam,
            gameplay=settings.gameplay,
            lookback_minutes=max(interval_minutes + 3, 15),
            max_clips=min(settings.max_clips, 3),
            clip_length=settings.clip_length,
            topic_filter=settings.topic_filter,
            captions=settings.captions,
            loudnorm=settings.loudnorm,
            watermark=settings.watermark,
            hook_banner=settings.hook_banner,
        )

        while not self._watch_stop.is_set():
            temp_dir = self.work_dir / f"watch_{uuid.uuid4().hex[:8]}"
            temp_dir.mkdir(parents=True, exist_ok=True)
            try:
                with self._lock:
                    self._watch.status_message = "Scanning live stream for new viral moments..."
                    self._watch.last_error = None
                video = inspect_youtube_url(url)
                with self._lock:
                    self._watch.title = video.title

                if not video.is_live:
                    with self._lock:
                        self._watch.active = False
                        self._watch.status_message = "Stream ended — Watch Live Mode stopped."
                    break

                dummy_job = JobState(job_id="watch", source_title=video.title, is_live=True)
                created = self._run_live_pipeline(
                    job=dummy_job,
                    video=video,
                    api_key=api_key,
                    settings=watch_settings,
                    temp_dir=temp_dir,
                    skip_ranges=self._watch_clipped_ranges,
                )
                for c in created:
                    self._watch_clipped_ranges.append((c.start_sec, c.end_sec))

                with self._lock:
                    self._watch.scans_completed += 1
                    self._watch.clips_created += len(created)
                    self._watch.next_scan_at = time.time() + interval_minutes * 60
                    self._watch.status_message = (
                        f"Found {len(created)} new viral clip(s) — waiting {interval_minutes}m for next scan."
                        if created
                        else f"No new viral spikes yet — scanning again in {interval_minutes}m."
                    )
            except Exception as exc:
                with self._lock:
                    self._watch.last_error = str(exc)
                    self._watch.next_scan_at = time.time() + interval_minutes * 60
                    self._watch.status_message = f"Scan warning (will retry in {interval_minutes}m): {exc}"
            finally:
                shutil.rmtree(temp_dir, ignore_errors=True)

            # Wait cleanly until next cycle or Stop clicked
            if self._watch_stop.wait(timeout=interval_minutes * 60):
                break
