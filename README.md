# NeuroViral

Upload a short video → see what grabs attention in it, second by second, synced
to playback → get a breakdown of what drove it. Built for the PrincetonBuilds
Ideathon. See `CLAUDE.md` for the original build spec.

**Architecture:** two decoupled halves. Analysis writes static JSON to
`data/results/`; the web app reads only those files and never imports torch — so
it runs offline with no GPU.

```
[ analysis engine ] → data/results/*.json → [ Gradio app ]
```

## Two engines (hybrid)

- **Audiovisual (default, CPU) — the real working path.** `pipeline/cpu_engine.py`
  measures bottom-up attention from the video itself: motion / scene-cuts / detail
  (*visual interest*) + loudness / onsets (*audio energy*). Runs on a laptop,
  offline, in seconds. **Upload a clip → it actually analyzes it.**
- **TRIBE brain (optional, GPU) — research mode.** `pipeline/extract.py` predicts
  an fMRI response with Meta TRIBE v2. Needs CUDA; runs on HF Spaces (ZeroGPU).
  Off-GPU it explains why it can't run instead of faking data.
  **TRIBE v2 is CC-BY-NC — research/non-commercial only.**

Both emit the same schema v2 (`pipeline/schema.py`): timeline channels `a`/`b`
+ `global`, with a `channels` descriptor and a `viz` mode (`meter` vs `brain`),
so the UI labels itself from the data.

## Run

```bash
pip install -r requirements.txt

python3 -m pytest -q          # offset correction + schema tests
python3 -m pipeline.make_mock # seed the illustrative sample gallery
python3 app/app.py            # -> http://127.0.0.1:7860
```

## Analyze your own videos

**Right now, no GPU:** run the app, open "Analyze your own video", upload a short
clip, keep the engine on *Audiovisual*, click Analyze.

**Build a real gallery from your clips:** drop `.mp4`s into `data/videos/`
(optionally add rows to `data/metadata.csv`), then:

```bash
python3 -m pipeline.analyze_folder
```

## Predicting virality (the model)

Concrete method: **Ridge regression** on the audiovisual features, target
`log(likes/views)` — engagement RATE, not raw views (raw views mostly measure
channel size).

```bash
# 1) collect labeled Shorts (needs internet; raise --sleep if rate-limited)
python3 -m pipeline.collect --queries "funny,cat,food" --channels "@zachking" --sleep 4
# 2) train (prints holdout R² / MAE / Pearson r)
python3 -m pipeline.model --train
# 3) rebuild the gallery, then run the app
python3 -m pipeline.analyze_folder && python3 app/app.py
```

### Honest result

On a first real dataset (**N=61** YouTube Shorts) the model scored
**holdout R² = -0.188** — i.e. **no usable signal**. Short-video virality is
genuinely hard to predict from audiovisual features alone.

The pipeline reports this rather than hiding it, and `predict_engagement` refuses
to serve scores from a model below `MIN_DEPLOY_R2`. So the app shows **measured
signals with no score** instead of dressing up noise. Training also warns if
|Pearson r| > 0.85 at small N (likely data leak).

## Status

- **Phase 1 — pipeline:** schema v2, hemodynamic-offset correction (unit-tested),
  TRIBE path wired (untested without a GPU — verify `model.predict` and whether
  it already applies the 5 s offset on first real run).
- **Phase 2 — web app:** working. Synced player (one master clock drives video +
  viz + timeline), peak annotation, gallery, hit-vs-flop compare, live upload,
  methodology panel.
- **Phase 3 — model:** built and run; null result documented above.

## Notes

- Cortical surface only (fsaverage5) in TRIBE mode: vmPFC + anterior insula. No
  subcortical regions (nucleus accumbens is unavailable).
- The gallery ships *illustrative sample* curves (marked `[sample]`), not real
  analyses — use `analyze_folder` for real ones.
