"""Generate illustrative results/*.json for the gallery (schema v2).

These are SAMPLE curves for the audiovisual engine, not analyses of real files —
they exist so the gallery/score UI has something to show out of the box. For a
real gallery, drop clips into data/videos/ and run pipeline.analyze_folder.

Run: python3 -m pipeline.make_mock
"""
from __future__ import annotations

import math
import random
from pathlib import Path

from pipeline.cpu_engine import CHANNELS
from pipeline.schema import (
    Engagement,
    Index,
    IndexEntry,
    Prediction,
    TimelinePoint,
    VideoResult,
)

RESULTS_DIR = Path("data/results")


def _curve(duration: float, *, hook: float, peak_t: float, base: float,
           decay: float, noise: float, seed: int) -> list[TimelinePoint]:
    """Per-second a (visual interest) / b (audio energy) with a hook spike."""
    rng = random.Random(seed)
    pts = []
    for s in range(int(duration)):
        bump = hook * math.exp(-((s - peak_t) ** 2) / (2 * 1.2 ** 2))
        trend = base + decay * s
        a = max(0.0, trend + bump + rng.uniform(-noise, noise))
        b = max(0.0, 0.6 * bump + 0.5 * trend + rng.uniform(-noise, noise))
        pts.append(TimelinePoint(t=float(s), a=round(a, 3), b=round(b, 3),
                                 global_=round(0.5 * (a + b), 3)))
    return pts


def _features_and_drivers(tl: list[TimelinePoint]):
    import numpy as np

    t = np.array([p.t for p in tl]); a = np.array([p.a for p in tl])
    b = np.array([p.b for p in tl]); g = np.array([p.global_ for p in tl])
    hook = a[t < 3.0]
    feats = {
        "hook_3s": round(float(hook.mean()) if hook.size else float(a[0]), 3),
        "visual_peak": round(float(a.max()), 3),
        "visual_mean": round(float(a.mean()), 3),
        "audio_mean": round(float(b.mean()), 3),
        "decay_slope": round(float(np.polyfit(t, g, 1)[0]) if len(t) >= 2 else 0.0, 4),
        "variance": round(float(g.var()), 4),
    }
    drivers = [
        ("hook (first 3s)", feats["hook_3s"]),
        ("visual peak", feats["visual_peak"]),
        ("audio energy", feats["audio_mean"]),
        ("sustained variance", feats["variance"]),
    ]
    return feats, drivers


def _build(video_id, title, duration, eng, pred, **curve_kw) -> VideoResult:
    tl = _curve(duration, **curve_kw)
    feats, drivers = _features_and_drivers(tl)
    return VideoResult(
        video_id=video_id, title=title, duration_sec=duration, engagement=eng,
        timeline=tl, features=feats, channels=CHANNELS, engine="audiovisual",
        viz="meter", drivers=drivers, prediction=pred,
    )


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    specs = [
        dict(video_id="hit_dogpark", title="Dog does a backflip (0:00 hook)  [sample]",
             duration=24.0, eng=Engagement(4_200_000, 511_000, 210_000, 0.1217),
             pred=Prediction(0.118, 94), hook=0.9, peak_t=2.0, base=0.12, decay=-0.004, noise=0.03, seed=1),
        dict(video_id="flop_unboxing", title="Slow unboxing, no hook (payoff 0:18)  [sample]",
             duration=26.0, eng=Engagement(38_000, 290, 210_000, 0.0076),
             pred=Prediction(0.014, 11), hook=0.3, peak_t=18.0, base=0.05, decay=-0.006, noise=0.04, seed=2),
        dict(video_id="mid_recipe", title="15-second pasta recipe  [sample]",
             duration=18.0, eng=Engagement(620_000, 41_000, 95_000, 0.0661),
             pred=Prediction(0.064, 63), hook=0.6, peak_t=3.5, base=0.10, decay=-0.003, noise=0.03, seed=3),
        dict(video_id="raw_streetart", title="Time-lapse street mural (no score)  [sample]",
             duration=30.0, eng=Engagement(150_000, 8_900, 44_000, 0.0593),
             pred=None, hook=0.5, peak_t=6.0, base=0.09, decay=-0.002, noise=0.05, seed=4),
    ]
    index = Index()
    for s in specs:
        r = _build(**s)
        r.write(RESULTS_DIR)
        index.videos.append(IndexEntry(video_id=r.video_id, title=r.title,
                                       duration_sec=r.duration_sec,
                                       score=r.prediction.score if r.prediction else None,
                                       engine=r.engine))
        print(f"wrote {r.video_id}.json ({len(r.timeline)} pts)")
    index.write(RESULTS_DIR)
    print(f"wrote index.json ({len(index.videos)} videos)")


if __name__ == "__main__":
    main()
