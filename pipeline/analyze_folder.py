"""Build a REAL gallery from real videos, using the CPU engine (no GPU).

Drop short clips into data/videos/*.mp4, optionally add rows to data/metadata.csv
(video_id,url,title,views,likes,subs,duration), then:

    python3 -m pipeline.analyze_folder

Writes data/results/<id>.json + index.json from actual analysis of your files.
Entries get an engagement score only if a deployable model exists (see
pipeline/model.py); otherwise the app shows measured signals with no score.
"""
from __future__ import annotations

import csv
import logging
from pathlib import Path

from pipeline.cpu_engine import analyze_video_cpu
from pipeline.schema import Index, IndexEntry

log = logging.getLogger("analyze_folder")

VIDEOS = Path("data/videos")
RESULTS = Path("data/results")
META = Path("data/metadata.csv")


def _load_meta() -> dict[str, dict]:
    if not META.exists():
        return {}
    with META.open() as f:
        return {r["video_id"]: r for r in csv.DictReader(f) if r.get("video_id")}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    metas = _load_meta()
    clips = sorted(p for p in VIDEOS.glob("*.mp4"))
    if not clips:
        print(f"No .mp4 files in {VIDEOS}/ — add some short clips and re-run.")
        return

    index = Index()
    for clip in clips:
        meta = metas.get(clip.stem)
        result = analyze_video_cpu(clip, meta=meta)
        result.write(RESULTS)
        index.videos.append(IndexEntry(
            video_id=result.video_id, title=result.title,
            duration_sec=result.duration_sec,
            score=result.prediction.score if result.prediction else None,
            engine=result.engine,
        ))
        print(f"analyzed {clip.name} -> {result.video_id}.json ({len(result.timeline)} pts)")
    index.write(RESULTS)
    print(f"wrote index.json ({len(index.videos)} videos). Restart the app to see them.")


if __name__ == "__main__":
    main()
