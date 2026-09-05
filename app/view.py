"""HTML/JS island for the synced player.

The video, the viz, and the response timeline all read from ONE master clock so
they stay in sync without any Python round-trip per frame (CLAUDE.md: "video and
brain animation must stay in sync"). If a real source video exists it becomes
the clock; otherwise a synthetic clock drives an animated placeholder.

This module builds a string; it imports nothing heavy and never touches torch.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_TEMPLATE = Path(__file__).with_name("player.html").read_text()


def build_main_view(result: dict[str, Any], video_url: str | None) -> str:
    """Return a self-contained HTML/JS island rendering `result`."""
    payload = {
        "video": result,
        "videoUrl": video_url,
        "uid": "nv_" + result["video_id"],
    }
    return _TEMPLATE.replace("/*__PAYLOAD__*/null", json.dumps(payload))


def build_compare_view(hit: dict[str, Any], flop: dict[str, Any]) -> str:
    """Side-by-side hit-vs-flop timelines — the two-second thesis (CLAUDE.md)."""
    tmpl = Path(__file__).with_name("compare.html").read_text()
    payload = {"hit": hit, "flop": flop}
    return tmpl.replace("/*__PAYLOAD__*/null", json.dumps(payload))
