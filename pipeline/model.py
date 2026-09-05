"""Engagement prediction model.

Concrete method: **Ridge regression** on audiovisual features, predicting
``log(likes / views)`` (engagement RATE, not raw views — raw views mostly
measure channel size). This is what would turn the app's measured signals into
an actual prediction of whether a clip will catch attention.

Pipeline:
    collect.py  -> data/videos + metadata.csv        (labels)
    model.py --train                                  (features via cpu_engine,
                                                       fit Ridge, save artifact)
    app / analyze_folder -> predict_engagement(feats) (live scoring)

Honesty guards (per CLAUDE.md):
  * Split BEFORE fitting; report HOLDOUT metrics only.
  * If holdout Pearson r > ~0.85 at small N, warn about a likely leak.
  * A model below MIN_DEPLOY_R2 is NOT served live — the app shows measured
    signals with no score rather than dressing up noise.

NOTE: on a first real run (N=61 YouTube Shorts) this yielded holdout
R^2 = -0.188 — i.e. no usable signal. Short-video virality is genuinely hard to
predict from audiovisual features alone. That null result is reported honestly
rather than hidden.
"""
from __future__ import annotations

import csv
import logging
import math
from pathlib import Path

from pipeline.schema import Prediction

log = logging.getLogger("model")

MODEL_PATH = Path("data/model/engagement.joblib")
META = Path("data/metadata.csv")
VIDEOS = Path("data/videos")

# A model below this holdout R^2 is noise — we still SAVE it (for inspection),
# but predict_engagement refuses to serve scores from it, so the app honestly
# shows measured signals with no score rather than dressing up noise.
MIN_DEPLOY_R2 = 0.05

# Feature vector order the model is trained/served on (audiovisual engine keys).
FEATURE_ORDER = [
    "hook_3s", "visual_peak", "visual_mean", "audio_mean",
    "cuts_per_sec", "decay_slope", "variance", "duration",
]

_CACHE: dict | None = None


# --- dataset ----------------------------------------------------------------

def build_dataset(min_views: int = 1000) -> tuple[list[str], list[list[float]], list[float]]:
    """Analyze every labeled video and return (ids, X, y=log(likes/views)).

    Clips below ``min_views`` are dropped: likes/views is dominated by noise on
    micro-view videos (10 views / 4 likes = a bogus 0.4 "rate").
    """
    from pipeline.cpu_engine import analyze_video_cpu

    if not META.exists():
        raise FileNotFoundError(f"{META} not found — run pipeline.collect first.")

    ids: list[str] = []
    X: list[list[float]] = []
    y: list[float] = []
    skipped_lowview = 0
    with META.open() as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        vid = r.get("video_id")
        clip = VIDEOS / f"{vid}.mp4"
        try:
            views = float(r.get("views") or 0)
            likes = float(r.get("likes") or 0)
        except ValueError:
            continue
        if not vid or not clip.exists() or views <= 0 or likes <= 0:
            continue
        if views < min_views:
            skipped_lowview += 1
            continue
        result = analyze_video_cpu(clip, meta=r)
        feats = dict(result.features)
        feats["duration"] = result.duration_sec
        if any(k not in feats for k in FEATURE_ORDER):
            continue
        ids.append(vid)
        X.append([float(feats[k]) for k in FEATURE_ORDER])
        y.append(math.log(likes / views))
    log.info("built dataset: %d clips (skipped %d below %d views)",
             len(ids), skipped_lowview, min_views)
    return ids, X, y


# --- train ------------------------------------------------------------------

def train(test_size: float = 0.25, seed: int = 0) -> dict:
    import joblib
    import numpy as np
    from sklearn.linear_model import RidgeCV
    from sklearn.metrics import mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    ids, X, y = build_dataset()
    if len(ids) < 12:
        raise SystemExit(f"Only {len(ids)} labeled clips — collect more before training "
                         f"(aim for 100+). Run pipeline.collect.")

    X = np.asarray(X, float)
    y = np.asarray(y, float)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=seed)

    def fit(xx, yy):
        return make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))).fit(xx, yy)

    # holdout metrics
    m = fit(Xtr, ytr)
    pred = m.predict(Xte)
    r2 = r2_score(yte, pred)
    mae = mean_absolute_error(yte, pred)
    r = float(np.corrcoef(pred, yte)[0, 1]) if len(yte) > 1 else float("nan")

    # deployed model: refit on ALL data
    final = fit(X, y)
    deployable = bool(r2 >= MIN_DEPLOY_R2 and abs(r) <= 0.85)
    artifact = {
        "pipeline": final,
        "feature_order": FEATURE_ORDER,
        "y_ref": sorted(y.tolist()),          # for percentiles
        "n": len(ids),
        "deployable": deployable,             # predict_engagement honors this
        "metrics": {"holdout_r2": round(float(r2), 3),
                    "holdout_mae_lograte": round(float(mae), 3),
                    "holdout_pearson_r": round(r, 3)},
    }
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, MODEL_PATH)

    print(f"trained on {len(ids)} clips (holdout {len(yte)})")
    print(f"  holdout R^2 = {r2:.3f} | MAE(log-rate) = {mae:.3f} | Pearson r = {r:.3f}")
    if abs(r) > 0.85:
        print("  ⚠️  |r| > 0.85 at this N — suspect a data leak; NOT served live.")
    elif not deployable:
        print(f"  ℹ️  R² < {MIN_DEPLOY_R2}: little/no signal — saved for inspection but NOT")
        print("      served live. The app shows measured signals with no score (honest).")
    else:
        print("  ✓ signal above threshold — served live in the app.")
    print(f"  saved -> {MODEL_PATH}")
    global _CACHE
    _CACHE = artifact
    return artifact


# --- predict ----------------------------------------------------------------

def _load() -> dict | None:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    if not MODEL_PATH.exists():
        return None
    import joblib

    _CACHE = joblib.load(MODEL_PATH)
    return _CACHE


def predict_engagement(features: dict[str, float]) -> Prediction | None:
    """Score a feature dict. None if no model, model too weak, or features differ.

    Returns a Prediction whose ``score`` is the predicted engagement RATE
    (likes/views) and ``percentile`` is the rank within the training set.
    """
    art = _load()
    if art is None or not art.get("deployable", False):
        return None  # no model, or a model too weak to serve honestly
    order = art["feature_order"]
    if any(k not in features for k in order):
        return None  # e.g. TRIBE features — not this model's inputs
    import numpy as np

    x = np.asarray([[float(features[k]) for k in order]], float)
    yhat = float(art["pipeline"].predict(x)[0])          # predicted log-rate
    score = max(0.0, min(1.0, math.exp(yhat)))           # back to rate
    y_ref = art["y_ref"]
    pct = 100.0 * (sum(1 for v in y_ref if v < yhat) / len(y_ref)) if y_ref else 50.0
    return Prediction(score=round(score, 4), percentile=round(pct, 1))


if __name__ == "__main__":
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description="Train / inspect the engagement model")
    ap.add_argument("--train", action="store_true")
    args = ap.parse_args()
    if args.train:
        train()
    else:
        art = _load()
        print("no model trained yet." if art is None
              else f"model: n={art['n']} metrics={art['metrics']}")
