"""CPU audiovisual engagement engine — no GPU, no TRIBE, runs offline.

Produces the SAME schema as the TRIBE path, so the app renders it identically.
Instead of a simulated brain response, it measures real bottom-up attention
signals from the video itself:

  * channel ``a`` — "visual interest": motion, scene cuts, spatial detail.
  * channel ``b`` — "audio energy": loudness (RMS) + onset strength.

These are established drivers of visual attention, so the per-second curve
genuinely spikes at hooks (a loud, high-motion cut). It's a measurement of the
stimulus, not a guess and not a brain — and it runs on a laptop in ~seconds.

Dependencies: opencv, librosa, imageio-ffmpeg (bundles ffmpeg), soundfile — all
CPU, all pip-installable.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from pipeline.schema import Channel, Engagement, TimelinePoint, VideoResult

log = logging.getLogger("cpu_engine")

CHANNELS = {
    "a": Channel("Visual interest", "#2dd4bf", "motion · scene cuts · detail"),
    "b": Channel("Audio energy", "#fb923c", "loudness · onsets"),
}


def _norm01(x: np.ndarray) -> np.ndarray:
    """Robust 0-1 scaling using the 5th/95th percentiles (outlier-tolerant)."""
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    lo, hi = np.percentile(x, 5), np.percentile(x, 95)
    if hi - lo < 1e-9:
        return np.zeros_like(x)
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


# --- video ------------------------------------------------------------------

def _analyze_video(video_path: Path, n_sec: int, analysis_fps: float = 8.0) -> dict[str, np.ndarray]:
    """Per-second motion, scene-cut rate, and spatial detail from the frames."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, int(round(fps / analysis_fps)))

    motion = [[] for _ in range(n_sec)]
    cuts = [[] for _ in range(n_sec)]
    detail = [[] for _ in range(n_sec)]

    prev = None
    idx = -1
    while True:
        ok = cap.grab()
        if not ok:
            break
        idx += 1
        if idx % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            break
        t = idx / fps
        sec = int(t)
        if sec >= n_sec:
            break
        gray = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        detail[sec].append(float(gray.std()) / 128.0)
        if prev is not None:
            d = float(np.abs(gray - prev).mean()) / 255.0
            motion[sec].append(d)
            cuts[sec].append(1.0 if d > 0.18 else 0.0)  # big frame delta => cut
        prev = gray
    cap.release()

    def per_sec(buckets, agg=np.mean):
        return np.array([agg(b) if b else 0.0 for b in buckets], dtype=np.float32)

    return {
        "motion": per_sec(motion),
        "cuts": per_sec(cuts, np.max),
        "detail": per_sec(detail),
    }


# --- audio ------------------------------------------------------------------

def _extract_audio(video_path: Path, sr: int = 22050) -> np.ndarray | None:
    """Decode the soundtrack to mono via the bundled ffmpeg. None if silent."""
    import imageio_ffmpeg
    import soundfile as sf

    ff = imageio_ffmpeg.get_ffmpeg_exe()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav = tmp.name
    try:
        proc = subprocess.run(
            [ff, "-y", "-i", str(video_path), "-vn", "-ac", "1", "-ar", str(sr), wav],
            capture_output=True,
        )
        if proc.returncode != 0 or not Path(wav).exists():
            return None
        y, _ = sf.read(wav, dtype="float32")
        return y if y.size else None
    except Exception as e:  # no audio stream, decode error, etc.
        log.warning("audio extract failed: %s", e)
        return None
    finally:
        Path(wav).unlink(missing_ok=True)


def _analyze_audio(y: np.ndarray | None, n_sec: int, sr: int = 22050) -> dict[str, np.ndarray]:
    if y is None or y.size == 0:
        return {"rms": np.zeros(n_sec, np.float32), "onset": np.zeros(n_sec, np.float32)}
    import librosa

    hop = 512
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    frame_t = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)

    def per_sec(vals):
        out = np.zeros(n_sec, np.float32)
        for s in range(n_sec):
            m = (frame_t >= s) & (frame_t < s + 1)
            if m.any():
                out[s] = float(np.mean(vals[: len(frame_t)][m]))
        return out

    return {"rms": per_sec(rms), "onset": per_sec(onset[: len(frame_t)])}


# --- orchestration ----------------------------------------------------------

def _probe_duration(video_path: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
    cap.release()
    return float(n / fps) if fps else 0.0


def analyze_video_cpu(
    video_path: Path,
    *,
    title: str | None = None,
    video_id: str | None = None,
    meta: dict | None = None,
) -> VideoResult:
    """Analyze one video on CPU and return a VideoResult (schema v2)."""
    video_path = Path(video_path)
    video_id = video_id or video_path.stem
    duration = _probe_duration(video_path)
    n_sec = max(1, int(duration))
    log.info("cpu-analyzing %s (%.1fs, %d buckets)", video_path, duration, n_sec)

    v = _analyze_video(video_path, n_sec)
    a_audio = _analyze_audio(_extract_audio(video_path), n_sec)

    # channel a: visual interest; channel b: audio energy
    a = _norm01(0.5 * _norm01(v["motion"]) + 0.3 * v["cuts"] + 0.2 * _norm01(v["detail"]))
    b = _norm01(0.6 * _norm01(a_audio["rms"]) + 0.4 * _norm01(a_audio["onset"]))
    g = 0.5 * (a + b)

    timeline = [
        TimelinePoint(t=float(s), a=round(float(a[s]), 3), b=round(float(b[s]), 3),
                      global_=round(float(g[s]), 3))
        for s in range(n_sec)
    ]

    t_arr = np.arange(n_sec, dtype=np.float32)
    hook_mask = t_arr < 3.0
    features = {
        "hook_3s": round(float(a[hook_mask].mean()) if hook_mask.any() else float(a[0]), 3),
        "visual_peak": round(float(a.max()), 3),
        "visual_mean": round(float(a.mean()), 3),
        "audio_mean": round(float(b.mean()), 3),
        "cuts_per_sec": round(float(v["cuts"].mean()), 3),
        "decay_slope": round(float(np.polyfit(t_arr, g, 1)[0]) if n_sec >= 2 else 0.0, 4),
        "variance": round(float(g.var()), 4),
        "duration": round(duration, 2),
    }
    drivers = [
        ("hook (first 3s)", features["hook_3s"]),
        ("visual peak", features["visual_peak"]),
        ("audio energy", features["audio_mean"]),
        ("cuts / sec", features["cuts_per_sec"]),
        ("sustained variance", features["variance"]),
    ]

    return VideoResult(
        video_id=video_id,
        title=title or (meta or {}).get("title", video_id),
        duration_sec=round(duration, 2),
        engagement=_engagement(meta),
        timeline=timeline,
        features=features,
        channels=CHANNELS,
        engine="audiovisual",
        viz="meter",
        drivers=drivers,
        prediction=_maybe_predict(features),
    )


def _maybe_predict(features: dict):
    """Score with the trained model if one exists, else None (panel hides)."""
    try:
        from pipeline.model import predict_engagement
        return predict_engagement(features)
    except Exception as e:  # never let scoring break analysis
        log.warning("prediction skipped: %s", e)
        return None


def _engagement(meta: dict | None) -> Engagement:
    if not meta:
        return Engagement(views=0, likes=0, subs=0, rate=0.0)
    views = int(float(meta.get("views", 0) or 0))
    likes = int(float(meta.get("likes", 0) or 0))
    subs = int(float(meta.get("subs", 0) or 0))
    return Engagement(views=views, likes=likes, subs=subs,
                      rate=round(likes / views, 6) if views else 0.0)


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)
    from pipeline.schema import validate_result

    res = analyze_video_cpu(Path(sys.argv[1]))
    errs = validate_result(res.to_dict())
    print(f"{len(res.timeline)} timeline pts, features={res.features}")
    print("VALID" if not errs else f"INVALID: {errs}")
