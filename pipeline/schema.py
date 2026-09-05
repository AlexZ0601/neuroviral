"""Frozen results schema for NeuroViral (v2, engine-neutral).

Single source of truth for ``data/results/<video_id>.json`` and ``index.json``.
The web app reads these files and nothing else from the pipeline.

v2 generalizes the timeline so two different engines can produce the same shape:
  * ``audiovisual`` — CPU signal-processing engine (no GPU); channels are
    "visual interest" and "audio energy". ``viz: "meter"``.
  * ``tribe`` — Meta TRIBE fMRI prediction (GPU, optional research mode);
    channels are "vmPFC" and "anterior insula". ``viz: "brain"``.

Two timeline channels ``a`` and ``b`` plus a ``global`` mean. What a/b MEAN is
described by ``channels``, so the UI labels/colors itself from the data.

Nothing here imports torch or any GPU dependency.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 2


@dataclass
class Channel:
    """Describes what a timeline channel means, for the UI to label itself."""

    label: str
    color: str
    desc: str = ""


@dataclass
class TimelinePoint:
    """One second of response. ``t`` is VIDEO time (seconds).

    ``a`` and ``b`` are the two engine channels; ``global_`` is their basis mean.
    """

    t: float
    a: float
    b: float
    global_: float

    def to_dict(self) -> dict[str, float]:
        return {"t": self.t, "a": self.a, "b": self.b, "global": self.global_}


@dataclass
class Engagement:
    views: int
    likes: int
    subs: int
    rate: float  # raw likes/views; the model TARGET is log(likes/views)


@dataclass
class Prediction:
    score: float       # predicted engagement (rate scale)
    percentile: float  # 0-100 rank within the precomputed set


@dataclass
class VideoResult:
    video_id: str
    title: str
    duration_sec: float
    engagement: Engagement
    timeline: list[TimelinePoint]
    features: dict[str, float]            # raw features (model input + reference)
    channels: dict[str, Channel]          # keys "a" and "b"
    engine: str                           # "audiovisual" | "tribe"
    viz: str = "meter"                    # "meter" | "brain"
    drivers: list[tuple[str, float]] | None = None  # ordered [label, value] for the UI
    brain_video: str = ""                 # optional pre-rendered viz mp4
    prediction: Prediction | None = None  # None => app hides the score panel
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "schema_version": self.schema_version,
            "video_id": self.video_id,
            "title": self.title,
            "duration_sec": self.duration_sec,
            "engine": self.engine,
            "viz": self.viz,
            "channels": {k: asdict(v) for k, v in self.channels.items()},
            "engagement": asdict(self.engagement),
            "timeline": [p.to_dict() for p in self.timeline],
            "features": self.features,
            "drivers": [list(d) for d in (self.drivers or [])],
            "brain_video": self.brain_video,
        }
        if self.prediction is not None:
            d["prediction"] = asdict(self.prediction)
        return d

    def write(self, results_dir: str | Path) -> Path:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / f"{self.video_id}.json"
        out.write_text(json.dumps(self.to_dict(), indent=2))
        return out


@dataclass
class IndexEntry:
    video_id: str
    title: str
    duration_sec: float
    score: float | None
    engine: str = "audiovisual"
    thumbnail: str | None = None


@dataclass
class Index:
    videos: list[IndexEntry] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        ordered = sorted(self.videos, key=lambda e: (e.score is None, -(e.score or 0.0)))
        return {"schema_version": self.schema_version, "videos": [asdict(e) for e in ordered]}

    def write(self, results_dir: str | Path) -> Path:
        results_dir = Path(results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        out = results_dir / "index.json"
        out.write_text(json.dumps(self.to_dict(), indent=2))
        return out


# --- Validation --------------------------------------------------------------

_REQUIRED_TOP = {"video_id", "title", "duration_sec", "engine", "channels",
                 "engagement", "timeline", "features"}
_REQUIRED_TIMELINE = {"t", "a", "b", "global"}


def validate_result(obj: dict[str, Any]) -> list[str]:
    """Return a list of human-readable problems; empty list means valid."""
    errors: list[str] = []

    missing = _REQUIRED_TOP - set(obj)
    if missing:
        errors.append(f"missing top-level keys: {sorted(missing)}")

    ch = obj.get("channels", {})
    if not ({"a", "b"} <= set(ch)):
        errors.append("channels must define 'a' and 'b'")

    timeline = obj.get("timeline")
    if not isinstance(timeline, list) or not timeline:
        errors.append("timeline must be a non-empty list")
    else:
        for i, pt in enumerate(timeline):
            bad = _REQUIRED_TIMELINE - set(pt)
            if bad:
                errors.append(f"timeline[{i}] missing keys: {sorted(bad)}")
                break
        ts = [pt.get("t") for pt in timeline if "t" in pt]
        if ts != sorted(ts):
            errors.append("timeline 't' values are not monotonically increasing")

    if "prediction" in obj and not {"score", "percentile"} <= set(obj["prediction"]):
        errors.append("prediction present but missing 'score'/'percentile'")

    return errors


if __name__ == "__main__":
    demo = VideoResult(
        video_id="demo",
        title="schema smoke test",
        duration_sec=2.0,
        engagement=Engagement(views=1_200_000, likes=89_000, subs=450_000, rate=0.0742),
        timeline=[
            TimelinePoint(t=0.0, a=0.31, b=0.12, global_=0.2),
            TimelinePoint(t=1.0, a=0.44, b=0.20, global_=0.3),
        ],
        features={"hook_3s": 0.52, "peak": 0.71, "decay_slope": -0.03},
        channels={"a": Channel("Visual interest", "#2dd4bf"),
                  "b": Channel("Audio energy", "#fb923c")},
        engine="audiovisual",
        viz="meter",
        drivers=[["hook (first 3s)", 0.52], ["peak", 0.71]],
        prediction=Prediction(score=0.68, percentile=82),
    )
    errs = validate_result(demo.to_dict())
    print(json.dumps(demo.to_dict(), indent=2))
    print("VALID" if not errs else f"INVALID: {errs}")
