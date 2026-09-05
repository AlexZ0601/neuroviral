"""Hemodynamic offset correction.

TRIBE predicts the fMRI BOLD response, which lags the stimulus that caused it
by the hemodynamic response delay (~5 s). So the brain activity the model
reports for prediction timestep *i* was actually driven by whatever was on
screen ~5 s EARLIER in the video.

To line the brain timeline up with the video timeline we must subtract this
delay when converting a prediction timestep to a video timestamp. Getting the
sign or magnitude wrong shifts every ROI trace relative to the video and
silently corrupts every downstream feature (hook_3s, peak timing, decay slope),
so this correction lives alone here and is unit-tested.

No torch import: pure arithmetic, safe for the app to import.
"""
from __future__ import annotations

# Canonical hemodynamic lag used across the pipeline. TRIBE's predictions are
# offset ~5 s into the past relative to the stimulus.
HEMODYNAMIC_OFFSET_SEC: float = 5.0


def brain_timestep_to_video_time(
    timestep_index: int,
    tr_sec: float,
    offset_sec: float = HEMODYNAMIC_OFFSET_SEC,
) -> float:
    """Map an fMRI prediction timestep to the VIDEO timestamp that caused it.

    Args:
        timestep_index: 0-based index of the prediction timestep.
        tr_sec: Seconds per prediction timestep (the fMRI repetition time, TR).
        offset_sec: Hemodynamic lag to remove. Defaults to ``HEMODYNAMIC_OFFSET_SEC``.

    Returns:
        Video timestamp in seconds. Brain time is ``timestep_index * tr_sec``;
        we subtract the lag because the response trails its stimulus. Early
        timesteps therefore map to NEGATIVE video times (the response to
        pre-roll / the very start hasn't ramped up yet); callers drop those.
    """
    brain_time = timestep_index * tr_sec
    return brain_time - offset_sec


def align_timeline_to_video(
    values_by_roi: dict[str, list[float]],
    tr_sec: float,
    duration_sec: float,
    offset_sec: float = HEMODYNAMIC_OFFSET_SEC,
) -> list[dict[str, float]]:
    """Convert per-timestep ROI predictions into video-aligned timeline points.

    Args:
        values_by_roi: e.g. ``{"vmpfc": [...], "insula": [...], "global": [...]}``,
            each list indexed by prediction timestep.
        tr_sec: Seconds per timestep.
        duration_sec: Video length; timeline points past this are dropped.
        offset_sec: Hemodynamic lag.

    Returns:
        List of ``{"t": <video_sec>, <roi>: <value>, ...}`` with ``t >= 0`` and
        ``t <= duration_sec``, sorted by ``t``. ``t`` is video time, ready for
        the schema's ``timeline``.
    """
    rois = list(values_by_roi)
    n = min(len(v) for v in values_by_roi.values()) if values_by_roi else 0

    points: list[dict[str, float]] = []
    for i in range(n):
        t = brain_timestep_to_video_time(i, tr_sec, offset_sec)
        if t < 0 or t > duration_sec:
            continue
        pt = {"t": round(t, 3)}
        for roi in rois:
            pt[roi] = values_by_roi[roi][i]
        points.append(pt)
    return points
