"""TRIBE inference: one video in, one results/<id>.json out. (Optional GPU path.)

RUNS ON A GPU (Kaggle T4/P100, or HF Spaces ZeroGPU). The web app never imports
this file. For the no-GPU path used by default, see ``pipeline/cpu_engine.py``.

Design constraints (see CLAUDE.md):
  * Peak VRAM must stay under 16 GB. TRIBE's `predict` runs three large frozen
    encoders (V-JEPA 2 / Wav2Vec-BERT / Llama-3.2) internally; on a 16 GB card
    watch the logged peak and use the repo's low-memory / reduced-precision
    options if it overflows. (The public API does the extraction for us — we do
    NOT manage per-modality loading by hand.)
  * The 5 s hemodynamic offset is applied via ``pipeline.hemodynamics`` — never
    inline here. See the note in ``run_tribe`` about not double-correcting.
  * ROI extraction uses the TRIBE repo's ``utils_fmri`` — we do NOT hand-roll
    vertex->region mapping.

NOTE: TRIBE v2 is CC-BY-NC (research / non-commercial only).

Usage:
    python pipeline/extract.py --video data/videos/test.mp4 \
        --metadata data/metadata.csv --out data/results
"""
from __future__ import annotations

import argparse
import csv
import gc
import logging
from pathlib import Path
from typing import Any

# NOTE: torch / tribev2 are imported lazily inside functions so that this module
# can be imported (and --help shown) on a machine with no GPU stack installed.

from pipeline.hemodynamics import HEMODYNAMIC_OFFSET_SEC, align_timeline_to_video
from pipeline.schema import (
    Channel,
    Engagement,
    Prediction,
    TimelinePoint,
    VideoResult,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("extract")

# fsaverage5 cortical ROIs. Nucleus accumbens is subcortical and NOT in this
# surface output — do not add it. vmPFC and anterior insula are cortical.
ROI_NAMES = ("vmpfc", "insula")

# TRIBE channels map to schema slots a=vmPFC (reward), b=anterior insula (salience).
TRIBE_CHANNELS = {
    "a": Channel("vmPFC", "#2dd4bf", "reward / value (cortical)"),
    "b": Channel("ant. insula", "#fb923c", "salience / arousal (cortical)"),
}

# Default TR (seconds per predicted timestep). TRIBE / Algonauts fMRI is ~1.49 s;
# override with --tr to match the exact model config you run with.
DEFAULT_TR_SEC = 1.49


# --- Memory helpers ----------------------------------------------------------

def _free(*objs: Any) -> None:
    """Drop references and empty the CUDA cache."""
    import torch

    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _log_peak_vram(stage: str) -> None:
    import torch

    if torch.cuda.is_available():
        peak_gb = torch.cuda.max_memory_allocated() / 1e9
        log.info("[vram] peak after %s: %.2f GB", stage, peak_gb)
        torch.cuda.reset_peak_memory_stats()


# --- TRIBE inference ---------------------------------------------------------

def run_tribe(video_path: Path, cache_dir: Path) -> tuple[Any, Any]:
    """Run TRIBE v2 on one video -> (preds, segments).

    This mirrors the real public API of github.com/facebookresearch/tribev2:

        model = TribeModel.from_pretrained("facebook/tribev2", cache_folder=...)
        df    = model.get_events_dataframe(video_path=...)
        preds, segments = model.predict(events=df)

    `predict` handles the trimodal feature extraction (V-JEPA 2 video,
    Wav2Vec-BERT audio, Llama-3.2 text) internally.

    preds: (n_timesteps, ~20k fsaverage5 cortical vertices), "average" subject.

    NOTE (verify against the repo): TRIBE documents its predictions as offset
    ~5 s for hemodynamic lag. If `predict` already applies that shift, run this
    pipeline with `--offset 0` to avoid double-correcting; if it returns the raw
    lagged signal, keep the default 5 s. Getting this wrong shifts every trace.
    """
    from tribev2 import TribeModel  # type: ignore

    cache_dir.mkdir(parents=True, exist_ok=True)
    model = TribeModel.from_pretrained(
        "facebook/tribev2",
        cache_folder=str(cache_dir / "hf"),
    )
    events = model.get_events_dataframe(video_path=str(video_path))
    preds, segments = model.predict(events=events)
    _log_peak_vram("tribe.predict")
    _free(model)
    return preds, segments


# --- Vertex -> ROI timeseries ------------------------------------------------

def preds_to_roi_timeseries(preds: Any) -> dict[str, list[float]]:
    """Reduce (n_timesteps, n_vertices) predictions to per-ROI timeseries.

    Uses the TRIBE repo's ROI machinery (utils_fmri) rather than a hand-rolled
    vertex->region map. Returns lists indexed by prediction timestep (NOT yet
    hemodynamically aligned) for vmpfc, insula, and a whole-cortex 'global' mean.
    """
    import numpy as np
    from tribev2 import utils_fmri  # type: ignore

    arr = np.asarray(preds, dtype=np.float32)  # (T, V)

    series: dict[str, list[float]] = {}
    for roi in ROI_NAMES:
        vertex_mask = utils_fmri.get_roi_vertices(roi, space="fsaverage5")
        series[roi] = arr[:, vertex_mask].mean(axis=1).tolist()
    series["global"] = arr.mean(axis=1).tolist()
    return series


# --- Feature engineering -----------------------------------------------------

def compute_features(timeline: list[TimelinePoint]) -> tuple[dict[str, float], list[tuple[str, float]]]:
    """Return (features dict, ordered drivers) for the TRIBE result.

    a = vmPFC (reward), b = anterior insula (salience).
    """
    import numpy as np

    if not timeline:
        raise ValueError("empty timeline; cannot compute features")

    t = np.array([p.t for p in timeline])
    vmpfc = np.array([p.a for p in timeline])
    insula = np.array([p.b for p in timeline])
    glob = np.array([p.global_ for p in timeline])

    hook_mask = t <= 3.0
    feats = {
        "hook_3s": round(float(vmpfc[hook_mask].mean()) if hook_mask.any() else float(vmpfc[0]), 3),
        "vmpfc_peak": round(float(vmpfc.max()), 3),
        "vmpfc_mean": round(float(vmpfc.mean()), 3),
        "insula_mean": round(float(insula.mean()), 3),
        "decay_slope": round(float(np.polyfit(t, glob, 1)[0]) if len(t) >= 2 else 0.0, 4),
        "response_variance": round(float(glob.var()), 4),
    }
    drivers = [
        ("hook (first 3s)", feats["hook_3s"]),
        ("vmPFC peak", feats["vmpfc_peak"]),
        ("vmPFC mean", feats["vmpfc_mean"]),
        ("insula mean", feats["insula_mean"]),
        ("decay slope", feats["decay_slope"]),
    ]
    return feats, drivers


# --- Metadata ----------------------------------------------------------------

def load_metadata(metadata_csv: Path, video_id: str) -> dict[str, Any] | None:
    if not metadata_csv.exists():
        return None
    with metadata_csv.open() as f:
        for row in csv.DictReader(f):
            if row.get("video_id") == video_id or Path(row.get("url", "")).stem == video_id:
                return row
    return None


def _engagement_from_meta(meta: dict[str, Any] | None) -> Engagement:
    if not meta:
        return Engagement(views=0, likes=0, subs=0, rate=0.0)
    views = int(float(meta.get("views", 0) or 0))
    likes = int(float(meta.get("likes", 0) or 0))
    subs = int(float(meta.get("subs", 0) or 0))
    rate = (likes / views) if views else 0.0
    return Engagement(views=views, likes=likes, subs=subs, rate=round(rate, 6))


# --- Orchestration -----------------------------------------------------------

def analyze_video(
    video_path: Path,
    *,
    tr_sec: float = DEFAULT_TR_SEC,
    offset_sec: float = HEMODYNAMIC_OFFSET_SEC,
    meta: dict[str, Any] | None = None,
    title: str | None = None,
    video_id: str | None = None,
    cache_dir: Path | None = None,
) -> VideoResult:
    """Run the full TRIBE -> ROI -> features chain and return a VideoResult.

    Shared core used by both the batch CLI (``process_video``) and the app's
    live-upload handler. It runs GPU inference, so callers on Kaggle / HF Spaces
    should invoke it inside their GPU context. Raises on failure — the caller
    decides how to degrade.
    """
    video_id = video_id or video_path.stem
    cache_dir = cache_dir or (Path("data/raw/cache") / video_id)
    log.info("analyzing %s (video_id=%s)", video_path, video_id)

    preds, _segments = run_tribe(video_path, cache_dir)

    roi_series = preds_to_roi_timeseries(preds)
    duration_sec = _probe_duration(video_path)

    aligned = align_timeline_to_video(roi_series, tr_sec, duration_sec, offset_sec)
    timeline = [
        TimelinePoint(t=p["t"], a=p["vmpfc"], b=p["insula"], global_=p["global"])
        for p in aligned
    ]
    features, drivers = compute_features(timeline)

    return VideoResult(
        video_id=video_id,
        title=title or (meta or {}).get("title", video_id),
        duration_sec=round(duration_sec, 2),
        engagement=_engagement_from_meta(meta),
        timeline=timeline,
        features=features,
        channels=TRIBE_CHANNELS,
        engine="tribe",
        viz="brain",
        drivers=drivers,
        brain_video=f"frames/{video_id}.mp4",  # optional, from pipeline/render.py
        prediction=_maybe_predict(features),
    )


def process_video(
    video_path: Path,
    out_dir: Path,
    metadata_csv: Path,
    cache_root: Path,
    tr_sec: float,
    offset_sec: float,
) -> Path:
    video_id = video_path.stem
    meta = load_metadata(metadata_csv, video_id)
    result = analyze_video(
        video_path,
        tr_sec=tr_sec,
        offset_sec=offset_sec,
        meta=meta,
        video_id=video_id,
        cache_dir=cache_root / video_id,
    )
    out_path = result.write(out_dir)
    log.info("wrote %s (%d timeline points)", out_path, len(result.timeline))
    return out_path


def _probe_duration(video_path: Path) -> float:
    """Video duration in seconds via ffprobe; falls back to 0.0 if unavailable."""
    import shutil
    import subprocess

    if not shutil.which("ffprobe"):
        log.warning("ffprobe not found; duration set to 0.0")
        return 0.0
    out = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(video_path),
        ],
        capture_output=True, text=True, check=True,
    )
    return float(out.stdout.strip() or 0.0)


def _maybe_predict(features: dict[str, float]) -> Prediction | None:
    """Score with the trained model if one applies, else None.

    The shipped model is trained on AUDIOVISUAL features, so it returns None for
    TRIBE's feature keys — a null result the app degrades to gracefully.
    """
    try:
        from pipeline.model import predict_engagement  # type: ignore
    except Exception:
        return None
    try:
        return predict_engagement(features)
    except Exception as e:  # never let a model hiccup break the JSON
        log.warning("prediction skipped: %s", e)
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description="TRIBE inference -> results JSON")
    ap.add_argument("--video", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/results"))
    ap.add_argument("--metadata", type=Path, default=Path("data/metadata.csv"))
    ap.add_argument("--cache", type=Path, default=Path("data/raw/cache"))
    ap.add_argument("--tr", type=float, default=DEFAULT_TR_SEC, help="seconds per fMRI timestep")
    ap.add_argument("--offset", type=float, default=HEMODYNAMIC_OFFSET_SEC,
                    help="hemodynamic lag to subtract (s)")
    args = ap.parse_args()

    process_video(
        video_path=args.video,
        out_dir=args.out,
        metadata_csv=args.metadata,
        cache_root=args.cache,
        tr_sec=args.tr,
        offset_sec=args.offset,
    )


if __name__ == "__main__":
    main()
