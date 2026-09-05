"""Collect a training dataset of YouTube Shorts with yt-dlp.

Downloads short clips + engagement labels (views, likes, subs, duration) into
data/videos/*.mp4 and data/metadata.csv. Run this yourself — needs internet and
takes a while.

Sources (mix and match):
  --queries    keyword searches  (broad, but low-view-heavy)
  --channels   channel handles/URLs — a creator's /shorts tab is mostly HIGH view
  --playlists  playlist URLs      — e.g. trending/compilation lists

    # keyword + channels, keep only >=1000-view clips
    python3 -m pipeline.collect \
        --queries "funny,prank,football,makeup,cat,car" --per-query 120 \
        --channels "@zachking,@dylanlemay" --limit 100 --sleep 4

Every video's real view/like count is checked BEFORE downloading, so low-view
clips (where likes/views is noise) never hit disk. Metadata is written per-clip,
so Ctrl+C never loses labels. Re-running skips ids already present.

If YouTube rate-limits you, raise --sleep and lower --per-query / --limit.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

VIDEOS = Path("data/videos")
META = Path("data/metadata.csv")
META_FIELDS = ["video_id", "url", "title", "views", "likes", "subs", "duration"]


def _existing_ids() -> set[str]:
    ids = {p.stem for p in VIDEOS.glob("*.mp4")}
    if META.exists():
        with META.open() as f:
            ids |= {r["video_id"] for r in csv.DictReader(f) if r.get("video_id")}
    return ids


def _append_row(row: dict) -> None:
    """Append ONE row immediately (write header if new) so Ctrl+C never loses labels."""
    new = not META.exists()
    with META.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=META_FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()


def _read_meta() -> list[dict]:
    if not META.exists():
        return []
    with META.open() as f:
        return list(csv.DictReader(f))


def _watch_url(entry: dict) -> str | None:
    url = entry.get("url") or entry.get("webpage_url")
    vid = entry.get("id")
    if url and url.startswith("http"):
        return url
    if vid:
        return f"https://www.youtube.com/watch?v={vid}"
    return None


def _sources(queries, per_query, channels, playlists, limit) -> list[str]:
    src: list[str] = []
    for q in queries:
        src.append(f"ytsearch{per_query}:{q} #shorts")
    for c in channels:
        c = c.strip().rstrip("/")
        if c.startswith("http"):
            src.append(c if c.endswith("/shorts") else c + "/shorts")
        else:
            handle = c if c.startswith("@") else "@" + c
            src.append(f"https://www.youtube.com/{handle}/shorts")
    src += [p.strip() for p in playlists]
    return src


def collect(queries, per_query, max_duration, min_views, channels, playlists, limit, sleep) -> None:
    import yt_dlp

    VIDEOS.mkdir(parents=True, exist_ok=True)
    META.parent.mkdir(parents=True, exist_ok=True)
    have = _existing_ids()

    flat = yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "ignoreerrors": True,
                             "extract_flat": "in_playlist", "playlistend": limit})
    dl = yt_dlp.YoutubeDL({
        "format": "mp4[height<=480]/best[height<=480]/mp4",
        "outtmpl": str(VIDEOS / "%(id)s.%(ext)s"),
        "quiet": True, "no_warnings": True, "ignoreerrors": True,
        # polite pacing to avoid YouTube rate-limiting (the "-t sleep" advice)
        "sleep_interval_requests": sleep,
        "sleep_interval": sleep,
        "max_sleep_interval": max(sleep * 3, sleep + 2),
        "retries": 3, "extractor_retries": 2,
    })

    fails = 0  # consecutive metadata failures — a burst usually means rate-limited
    added = 0
    try:
        with flat, dl:
            for source in _sources(queries, per_query, channels, playlists, limit):
                print(f"[collect] source: {source}")
                listing = flat.extract_info(source, download=False)
                entries = (listing or {}).get("entries") or []
                for e in entries:
                    url = _watch_url(e or {})
                    vid = (e or {}).get("id")
                    if not url or not vid or vid in have:
                        continue
                    # full metadata (view/like counts) before committing a download
                    info = dl.extract_info(url, download=False)
                    if not info:
                        fails += 1
                        if fails >= 8:
                            print("\n[collect] many failures in a row — likely rate-limited by "
                                  "YouTube. Stopping cleanly. Wait ~1h, then re-run (progress is "
                                  "saved). Consider a larger --sleep.")
                            return
                        continue
                    fails = 0
                    likes, views = info.get("like_count"), info.get("view_count")
                    dur = info.get("duration") or 0
                    if likes is None or not views or views < min_views or dur > max_duration:
                        continue
                    try:
                        dl.download([url])
                    except Exception as ex:
                        print(f"  skip {vid}: {ex}")
                        continue
                    if not (VIDEOS / f"{vid}.mp4").exists():
                        continue
                    have.add(vid)
                    row = {
                        "video_id": vid, "url": url,
                        "title": (info.get("title") or "").replace("\n", " ")[:120],
                        "views": views, "likes": likes,
                        "subs": info.get("channel_follower_count") or 0,
                        "duration": round(float(dur), 1),
                    }
                    _append_row(row)  # write immediately — Ctrl+C safe
                    added += 1
                    print(f"  + {vid}  views={views} likes={likes}")
    except KeyboardInterrupt:
        print("\n[collect] stopped (Ctrl+C). Progress saved.")
    finally:
        usable = sum(1 for r in _read_meta() if float(r["views"]) >= min_views and float(r["likes"]) > 0)
        print(f"[collect] added {added} clips this run. Usable (>= {min_views} views): {usable}.")
        print("When usable is ~250+: python3 -m pipeline.model --train")


def main() -> None:
    ap = argparse.ArgumentParser(description="Collect Shorts + labels via yt-dlp")
    ap.add_argument("--queries", default="", help="comma-separated search queries")
    ap.add_argument("--channels", default="", help="comma-separated channel handles/URLs")
    ap.add_argument("--playlists", default="", help="comma-separated playlist URLs")
    ap.add_argument("--per-query", type=int, default=50)
    ap.add_argument("--limit", type=int, default=80, help="max videos to scan per channel/playlist")
    ap.add_argument("--max-duration", type=int, default=60)
    ap.add_argument("--min-views", type=int, default=1000,
                    help="skip clips below this view count (likes/views is noisy on tiny videos)")
    ap.add_argument("--sleep", type=float, default=2.0,
                    help="seconds to pause between requests (raise if rate-limited)")
    args = ap.parse_args()

    def split(s):
        return [x.strip() for x in s.split(",") if x.strip()]

    if not (args.queries or args.channels or args.playlists):
        ap.error("give at least one of --queries / --channels / --playlists")
    collect(split(args.queries), args.per_query, args.max_duration, args.min_views,
            split(args.channels), split(args.playlists), args.limit, args.sleep)


if __name__ == "__main__":
    main()
