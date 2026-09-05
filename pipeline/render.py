"""ROI/vertex data -> brain frame PNGs -> brain mp4.

STUB (optional, TRIBE mode only). Plan:
  * Render one brain frame per second server-side using the TRIBE repo's
    ``plotting/`` module with the Nilearn backend (surface plot of predicted
    activity on fsaverage5).
  * Stitch frames to mp4 with ffmpeg at a duration MATCHING the source video so
    the app can play both <video> elements in parallel under one control.

Output path convention (frozen in schema): ``frames/<video_id>.mp4``.
Runs offline; never called by the app at demo time. The live app draws its
viz on canvas instead, so this is only needed for a pre-rendered brain movie.
"""
