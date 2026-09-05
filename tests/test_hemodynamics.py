"""Unit tests for the hemodynamic offset correction (CLAUDE.md requires this).

Run: pytest -q   (from repo root)
"""
from pipeline.hemodynamics import (
    HEMODYNAMIC_OFFSET_SEC,
    align_timeline_to_video,
    brain_timestep_to_video_time,
)


def test_offset_default_is_five_seconds():
    assert HEMODYNAMIC_OFFSET_SEC == 5.0


def test_first_timestep_maps_before_video_start():
    # timestep 0 at brain time 0 -> video time -5 (response precedes nothing yet).
    assert brain_timestep_to_video_time(0, tr_sec=1.0) == -5.0


def test_offset_is_subtracted_not_added():
    # brain time 10s (10 steps @ 1s) with 5s lag -> video time 5s.
    assert brain_timestep_to_video_time(10, tr_sec=1.0) == 5.0


def test_tr_scales_brain_time():
    # 4 steps at TR=1.49 -> 5.96 brain, minus 5 lag = 0.96 video time.
    assert brain_timestep_to_video_time(4, tr_sec=1.49) == 5.96 - 5.0


def test_custom_offset_overrides_default():
    assert brain_timestep_to_video_time(3, tr_sec=2.0, offset_sec=1.0) == 5.0


def test_align_drops_negative_and_overshoot_times():
    # 10 timesteps at TR=1.0, 5s lag, duration 3s.
    # video times: -5,-4,-3,-2,-1,0,1,2,3,4 -> keep only 0,1,2,3.
    series = {"vmpfc": list(range(10)), "insula": list(range(10)), "global": list(range(10))}
    out = align_timeline_to_video(series, tr_sec=1.0, duration_sec=3.0)
    ts = [p["t"] for p in out]
    assert ts == [0.0, 1.0, 2.0, 3.0]
    # value at video t=0 must be the timestep whose brain time was 5s (index 5).
    assert out[0]["vmpfc"] == 5


def test_align_is_monotonic_in_video_time():
    series = {"vmpfc": [0.0] * 20, "global": [0.0] * 20}
    out = align_timeline_to_video(series, tr_sec=1.49, duration_sec=20.0)
    ts = [p["t"] for p in out]
    assert ts == sorted(ts)
    assert all(t >= 0 for t in ts)
