"""100% Local AI Subject Tracker (YuNet CNN Face Detection + Spotlight/Motion Locking + Scene-Cut Snapping)."""

from __future__ import annotations

import urllib.request
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"
YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)

WORK_W = 640
WORK_H = 360


def _ensure_yunet_model() -> Optional[Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / YUNET_FILENAME
    if path.exists() and path.stat().st_size > 50_000:
        return path
    try:
        urllib.request.urlretrieve(YUNET_URL, str(path))
        if path.exists() and path.stat().st_size > 50_000:
            return path
    except Exception:
        pass
    return None


def _compute_hist(bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def _is_scene_cut(prev_hist: Optional[np.ndarray], cur_hist: np.ndarray, prev_gray: Optional[np.ndarray], cur_gray: np.ndarray) -> bool:
    if prev_hist is None or prev_gray is None:
        return False
    corr = float(cv2.compareHist(prev_hist, cur_hist, cv2.HISTCMP_CORREL))
    mad = float(np.mean(cv2.absdiff(prev_gray, cur_gray)))
    return corr < 0.62 or (corr < 0.78 and mad > 34.0)


def track_main_subject_local(
    video_path: Path,
    sample_fps: float = 3.0,
    progress_cb: Optional[Callable[[str], None]] = None,
) -> tuple[list[tuple[float, float]], list[float]]:
    """
    Track the main subject's horizontal center cx (0.0 to 1.0) locally using OpenCV YuNet CNN.
    Returns (track_points[(t, cx_norm), ...], scene_cuts[t_cut, ...]).
    Uses ZERO Gemini API calls and processes a 45s clip in ~1-2 seconds.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [(0.0, 0.5)], []

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    if src_fps <= 1.0 or src_fps > 240.0:
        src_fps = 30.0
    step = max(1, int(round(src_fps / max(1.0, sample_fps))))

    yunet_path = _ensure_yunet_model()
    detector = None
    if yunet_path is not None:
        try:
            detector = cv2.FaceDetectorYN.create(
                str(yunet_path), "", (WORK_W, WORK_H), 0.48, 0.3, 500
            )
        except Exception:
            detector = None

    haar_cascade = None
    if detector is None:
        haar_file = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        if haar_file.exists():
            haar_cascade = cv2.CascadeClassifier(str(haar_file))

    track: list[tuple[float, float]] = []
    cuts: list[float] = []

    prev_hist: Optional[np.ndarray] = None
    prev_gray: Optional[np.ndarray] = None
    locked_cx: Optional[float] = None
    frames_since_face = 0

    frame_idx = 0
    while True:
        grabbed = cap.grab()
        if not grabbed:
            break

        if frame_idx % step != 0:
            frame_idx += 1
            continue

        ret, frame = cap.retrieve()
        if not ret:
            break

        t = frame_idx / src_fps
        small = cv2.resize(frame, (WORK_W, WORK_H), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        hist = _compute_hist(small)

        # Detect hard camera cuts
        cut_now = _is_scene_cut(prev_hist, hist, prev_gray, gray)
        if cut_now and (not cuts or t - cuts[-1] > 0.35):
            cuts.append(round(t, 3))
            locked_cx = None  # Reset lock so we snap instantly to the new shot's subject

        motion_map = (
            cv2.absdiff(prev_gray, gray)
            if (prev_gray is not None and not cut_now)
            else None
        )

        # Detect faces
        faces: list[tuple[float, float, float, float, float]] = []
        if detector is not None:
            _, dets = detector.detect(small)
            if dets is not None:
                for row in dets:
                    fx, fy, fw, fh = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                    conf = float(row[14])
                    if fw >= 14 and fh >= 14:
                        faces.append((fx, fy, fw, fh, conf))
        elif haar_cascade is not None:
            rects = haar_cascade.detectMultiScale(gray, 1.15, 4, minSize=(20, 20))
            for (fx, fy, fw, fh) in rects:
                faces.append((float(fx), float(fy), float(fw), float(fh), 0.75))

        chosen_cx: Optional[float] = None
        if faces:
            best_score = -1e9
            frame_mean_luma = float(np.mean(gray)) + 1.0

            for (fx, fy, fw, fh, conf) in faces:
                cx = (fx + fw * 0.5) / WORK_W
                cy = (fy + fh * 0.5) / WORK_H
                area_norm = (fw * fh) / float(WORK_W * WORK_H)

                # Local brightness (spotlighted comedian pops out from dark crowd)
                x1, y1 = max(0, int(fx)), max(0, int(fy))
                x2, y2 = min(WORK_W, int(fx + fw)), min(WORK_H, int(fy + fh))
                roi_gray = gray[y1:y2, x1:x2]
                luma_ratio = (float(np.mean(roi_gray)) / frame_mean_luma) if roi_gray.size else 1.0

                # Motion energy around face + upper body (active speaker / performer)
                motion_bonus = 0.0
                if motion_map is not None:
                    my2 = min(WORK_H, int(fy + fh * 2.4))
                    mx1 = max(0, int(fx - fw * 0.4))
                    mx2 = min(WORK_W, int(fx + fw * 1.4))
                    m_roi = motion_map[y1:my2, mx1:mx2]
                    if m_roi.size:
                        motion_bonus = min(1.5, float(np.mean(m_roi)) / 18.0)

                # Penalize very bottom-edge audience heads (cy > 0.78)
                vertical_score = 1.0 if 0.12 <= cy <= 0.68 else (0.55 if cy <= 0.78 else 0.15)

                # Main subject prominence score
                score = (
                    (area_norm ** 0.55) * 18.0
                    + conf * 2.2
                    + min(2.0, luma_ratio) * 1.1
                    + motion_bonus * 1.2
                ) * vertical_score

                # Strong temporal continuity within the same shot so camera stays locked on the comedian
                if locked_cx is not None:
                    dist = abs(cx - locked_cx)
                    score += max(0.0, 2.8 * (1.0 - dist / 0.35))

                if score > best_score:
                    best_score = score
                    chosen_cx = cx

            frames_since_face = 0
        else:
            frames_since_face += 1
            # Fallback if subject briefly turns away: follow dominant upper-frame motion centroid
            if motion_map is not None and frames_since_face >= 3:
                upper = motion_map[: int(WORK_H * 0.78), :]
                mask = (upper > 24).astype(np.float32)
                col_energy = mask.sum(axis=0)
                total_energy = float(col_energy.sum())
                if total_energy > 120.0:
                    xs = np.arange(WORK_W, dtype=np.float32)
                    mot_cx = float((col_energy * xs).sum() / total_energy) / WORK_W
                    if locked_cx is None or abs(mot_cx - locked_cx) < 0.25:
                        chosen_cx = mot_cx

        if chosen_cx is not None:
            locked_cx = max(0.05, min(0.95, chosen_cx))
        elif locked_cx is None:
            locked_cx = 0.5

        track.append((round(t, 3), round(locked_cx, 4)))
        prev_hist = hist
        prev_gray = gray
        frame_idx += 1

    cap.release()
    if progress_cb:
        progress_cb(
            f"Tracked main subject across {len(track)} frames ({len(cuts)} camera cut(s) detected)"
        )
    return track or [(0.0, 0.5)], cuts
