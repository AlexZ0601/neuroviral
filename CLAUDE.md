# NeuroViral — Build Spec

You are building a demo web app for a 5-day solo hackathon (PrincetonBuilds Ideathon, presenting Sunday 8PM). Read this whole file before writing code.

**What it is:** upload or select a short-form video → see a simulated human brain response to it, second by second, animated in sync with playback → get a predicted engagement score with a breakdown of which brain regions drove it.

**Powered by:** Meta's TRIBE v2 (`facebook/tribev2`), an open-source multimodal model that predicts fMRI brain responses to video/audio/text.

---

## HARD CONSTRAINTS — do not violate these

1. **No paid GPU.** Free tier only: Kaggle notebooks (T4/P100, 16GB, 30hr/wk) for batch work, HF ZeroGPU for the hosted demo. Never write code that assumes an A100 or a rented pod.
2. **The demo must run with ZERO GPU at presentation time.** This is the single most important architectural requirement. All demo content is precomputed and served from static JSON + pre-rendered frames. Live inference is an optional bonus path, never a dependency.
3. **Solo dev, 5 days.** Prefer boring, working solutions over elegant ones. No microservices, no auth, no database — flat files and a single app.
4. **Build in phase order.** Phase 2 (the app) must be fully working on mock data before Phase 3 (the model) is attempted. Do not start Phase 3 until Phase 2 demos end to end.

---

## Architecture

Two decoupled halves. This decoupling is deliberate — it's what makes the demo survivable.

```
[ Kaggle notebook ]  →  data/results/*.json  →  [ Web app ]
   GPU, runs once         static artifacts        no GPU needed
```

The web app never imports torch and never calls TRIBE. It reads precomputed JSON. This means the app works on a laptop with no internet, which is what you want at 8PM Sunday.

---

## Repo structure

```
neuroviral/
├── CLAUDE.md
├── pipeline/
│   ├── extract.py          # TRIBE inference → raw preds (RUNS ON KAGGLE)
│   ├── rois.py             # vertex-level preds → ROI timeseries
│   ├── render.py           # ROI data → brain frame PNGs → mp4
│   └── model.py            # features → engagement regression
├── data/
│   ├── videos/             # downloaded .mp4 files
│   ├── metadata.csv        # url, views, likes, subs, duration
│   ├── raw/                # .npy prediction arrays (gitignored, large)
│   └── results/            # ← the app only ever reads this
│       ├── index.json
│       └── <video_id>.json
├── app/
│   ├── app.py              # Gradio app
│   └── static/
└── requirements.txt
```

---

## PHASE 1 — Inference pipeline

**Goal:** one video in, one `results/<id>.json` out.

### TRIBE API

```python
from tribev2 import TribeModel

model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")
df = model.get_events_dataframe(video_path="short.mp4")
preds, segments = model.predict(events=df)   # (n_timesteps, ~20k vertices)
```

Install: `pip install -e ".[plotting]"` from `github.com/facebookresearch/tribev2`.

### Critical correctness requirements

- **Hemodynamic offset:** predictions are offset **5 seconds into the past**. Subtract this when mapping brain timesteps to video timestamps. Getting this wrong silently corrupts every downstream number. Write this correction as a single named function with a comment, and unit-test it.
- **Surface space:** output is on the **fsaverage5 cortical surface** (~20k vertices). Nucleus accumbens is subcortical and is NOT available — do not write code that looks for it. Use **vmPFC** and **anterior insula**, both cortical.
- **ROI extraction:** use `utils_fmri.py` from the TRIBE repo, which has ROI analysis built in. Do not hand-roll vertex→region mapping.

### 16GB memory strategy (required)

Peak VRAM with all three feature extractors loaded simultaneously exceeds 16GB. Load them **sequentially**:

```
extract video features → del model, torch.cuda.empty_cache()
extract audio features → del model, torch.cuda.empty_cache()
extract text features  → del model, torch.cuda.empty_cache()
run encoder on cached features
```

Use fp16/bf16. Cache intermediate features to disk so a crash doesn't cost a full re-run. Log peak memory with `torch.cuda.max_memory_allocated()` so we can verify we're under budget.

### Output schema — `data/results/<video_id>.json`

Define this early and freeze it. The app depends on it.

```json
{
  "video_id": "abc123",
  "title": "...",
  "duration_sec": 32.4,
  "engagement": { "views": 1200000, "likes": 89000, "subs": 450000, "rate": 0.0742 },
  "timeline": [
    { "t": 0.0, "vmpfc": 0.31, "insula": -0.12, "global": 0.08 },
    { "t": 1.0, "vmpfc": 0.44, "insula": -0.20, "global": 0.11 }
  ],
  "features": { "vmpfc_mean": 0.29, "vmpfc_peak": 0.71, "hook_3s": 0.52, "decay_slope": -0.03 },
  "prediction": { "score": 0.68, "percentile": 82 },
  "brain_video": "frames/abc123.mp4"
}
```

**Acceptance:** running `python pipeline/extract.py --video data/videos/test.mp4` produces a valid JSON matching this schema, under 16GB peak VRAM.

---

## PHASE 2 — The web app (build this second, on mock data)

**Do this before Phase 3.** Generate 3–4 fake `results/*.json` files by hand, build the entire app against them, confirm it demos beautifully. Only then wire in real data.

### Stack

**Gradio**, deployed to **HF Spaces**. Reasons: free hosting, free ZeroGPU if we want live inference later, single Python file, public URL that works on stage. Do not build a separate React frontend — not worth the time.

### The main view — this is what wins the competition

A synced side-by-side:

```
┌─────────────────┬─────────────────┐
│   video plays   │  brain animates │
│                 │   in sync       │
├─────────────────┴─────────────────┤
│  ROI signal timeline (scrubbing    │
│  playhead, vmPFC + insula traces)  │
├────────────────────────────────────┤
│  Predicted score  │  what drove it │
└────────────────────────────────────┘
```

Requirements:
- Video and brain animation **must stay in sync**. Simplest reliable approach: pre-render brain frames at 1fps server-side (`pipeline/render.py`, using the repo's `plotting/` module with the Nilearn backend), stitch to mp4 with ffmpeg at matching duration, play both in parallel `<video>` elements driven by one control.
- The ROI timeline should have a playhead that tracks video position. Plotly or a small canvas chart, updated on `timeupdate`.
- Highlight the moment of peak predicted response — annotate it on the timeline. This is the "look, the brain spikes right at the hook" moment that sells the whole thing.

### Gallery view

Grid of precomputed videos, sorted by predicted score. Click → main view. Include an obvious **hit vs. flop comparison** pairing, since that single visual makes the thesis legible in two seconds.

### Live upload path (optional, build last)

If time allows: upload → run inference on ZeroGPU → render. Wrap in `@spaces.GPU(duration=120)`.

**Guard it:** ZeroGPU free tier gives ~5 min/day and ~60s per call. This path WILL fail during a live demo if quota is exhausted. Implement it with an explicit timeout and a graceful fallback message pointing at the precomputed gallery. Never make the main demo flow depend on it.

**Acceptance:** app runs with `python app/app.py`, loads mock JSON, video and brain play in sync, gallery navigates, and it works with wifi disabled.

---

## PHASE 3 — Prediction model

Only start this once Phase 2 demos cleanly.

- Target variable: **`log(likes / views)`** — engagement rate, NOT raw views.
- **Why this matters:** raw view count mostly measures channel size. A 10M-sub channel gets views regardless of content quality. Using raw views would mean the model learns "big channel = viral," which is both trivially true and indefensible when a judge asks. Normalize, and surface the reasoning in the app's methodology section.
- Features: `vmpfc_mean`, `vmpfc_peak`, `insula_mean`, `hook_3s` (mean response first 3s), `decay_slope`, response variance.
- Model: Ridge regression first. Gradient-boosted trees only if Ridge underfits. N will be ~50 — do not reach for anything deep.
- **Report holdout performance only.** Split before any feature exploration. If cross-validated r exceeds ~0.85 at this sample size, assume a data leak and hunt for it before believing it.
- If there's no signal: the app still ships showing brain responses without a score. Make the score panel conditional on `prediction` being present in the JSON, so a null result degrades gracefully instead of breaking the UI.

---

## PHASE 4 — Demo hardening (Sunday morning)

- Record a full screen capture of the working demo. Non-negotiable.
- Verify the app runs with wifi off.
- Add a methodology panel covering: simulated not measured brains, N=50, cortical regions only, CC-BY-NC license.
- Freeze code. Anything broken gets cut, not fixed.

---

## Do NOT do these

- Do not import torch in `app/`.
- Do not make the presentation demo depend on live GPU inference.
- Do not use raw view counts as the target variable.
- Do not build auth, user accounts, or a database.
- Do not attempt to fine-tune TRIBE. Inference only.
- Do not look for subcortical regions in cortical-surface output.
- Do not skip the 5-second hemodynamic offset correction.

---

## Context you may need

- TRIBE is **CC-BY-NC** — research/competition use only, no commercial path. Note this in the app footer.
- A 2026 paper found a *global* predicted-fMRI signal does NOT predict YouTube replay heatmaps (arxiv.org/pdf/2607.01400). This project's differentiator is using **region-specific** features instead of a global average. Frame it that way in the methodology panel — it turns a known negative result into motivation rather than a weakness.
- Dataset comes from `yt-dlp`: `--dump-json` gives view/like counts, and a normal download gives the video file. YouTube Shorts, not TikTok, because one tool gives both halves.

---

## Suggested first message to Claude Code

> Read CLAUDE.md. Start with Phase 1: scaffold the repo structure, write `pipeline/extract.py` with sequential feature extraction and the hemodynamic offset correction, and define the results JSON schema. Don't write the app yet.
