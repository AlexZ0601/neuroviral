"""NeuroViral — Gradio app.

Reads ONLY data/results/*.json. No torch at module load. Runs offline.

    python app/app.py

The synced video + viz + timeline is a self-contained HTML/JS island
(app/view.py -> player.html, driven by app/head.js); Gradio here provides the
shell, the gallery, the upload path, the comparison, and the methodology panel.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import gradio as gr

# make `pipeline` importable for schema validation, and `app` for view builders
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.view import build_compare_view, build_main_view  # noqa: E402

RESULTS = ROOT / "data" / "results"
VIDEOS = ROOT / "data" / "videos"
HEAD_JS = (Path(__file__).parent / "head.js").read_text()


# --- data loading ------------------------------------------------------------

def load_index() -> list[dict]:
    idx = json.loads((RESULTS / "index.json").read_text())
    return idx.get("videos", [])


def load_result(video_id: str) -> dict:
    return json.loads((RESULTS / f"{video_id}.json").read_text())


def video_url_for(video_id: str) -> str | None:
    """Serve a real source video if one exists, else None (sample => placeholder)."""
    mp4 = VIDEOS / f"{video_id}.mp4"
    if mp4.exists():
        return f"/gradio_api/file={mp4}"
    return None


# --- gallery -----------------------------------------------------------------

def gallery_card_label(entry: dict) -> str:
    if entry.get("score") is None:
        badge = "○ no score"
    elif entry["score"] >= 0.09:
        badge = f"▲ {entry['score']:.3f}"
    elif entry["score"] <= 0.03:
        badge = f"▼ {entry['score']:.3f}"
    else:
        badge = f"◆ {entry['score']:.3f}"
    return f"{badge}  ·  {entry['title']}"


METHODOLOGY = """
### Methodology & honest caveats
- **Two engines, same view.**
  - **Audiovisual (default, runs on CPU):** measures real bottom-up attention signals from your video — motion, scene cuts, spatial detail (*visual interest*) and loudness + onsets (*audio energy*). These are *measurements of the stimulus*, not guesses and not a brain. Grounded in attention research: motion, faces, audio onsets, and novelty are established drivers of visual attention.
  - **TRIBE brain (optional, needs a GPU):** *predicts* an fMRI response with Meta's TRIBE v2 — a simulation, no human is scanned. Region-specific (vmPFC / anterior insula), cortical surface only (no subcortical regions), with a ~5s hemodynamic correction. Runs on HF Spaces (ZeroGPU); CC-BY-NC.
- **The prediction is a Ridge regression** on the audiovisual features (`hook_3s`, `visual_peak`, `cuts_per_sec`, audio energy, decay, variance, duration), trained to predict `log(likes/views)` — engagement RATE, not raw views (raw views mostly measure channel size). Holdout-validated; a model that doesn't clear a minimum R² is **not served**, so the app shows measured signals with **no score** rather than guessing. The gallery uses *illustrative sample* curves.
- **Known null result:** on a first real dataset (N=61 YouTube Shorts) the model scored holdout R² = **-0.188** — no usable signal. Short-video virality is genuinely hard to predict from audiovisual features alone. Reported rather than hidden.
- **Why channel-specific:** a 2026 result found a *global* predicted-fMRI signal does not predict YouTube replay heatmaps — so we keep the two channels separate rather than averaging them away.
"""

FOOTER = (
    "TRIBE v2 is **CC-BY-NC** — research/competition use only, no commercial path.  ·  "
    "NeuroViral · PrincetonBuilds Ideathon demo"
)


def build_app() -> gr.Blocks:
    index = load_index()
    hit = next((e for e in index if (e.get("score") or 0) > 0.09), index[0] if index else None)
    flop = next((e for e in index if e.get("score") is not None and e["score"] <= 0.03), None)
    default_id = hit["video_id"] if hit else index[0]["video_id"]

    with gr.Blocks(title="NeuroViral", head=f"<script>{HEAD_JS}</script>") as demo:
        gr.Markdown(
            "# 🧠 NeuroViral\n"
            "**See what grabs attention in a short video, second by second — and what makes it engaging.**"
        )

        with gr.Row():
            with gr.Column(scale=1, min_width=260):
                gr.Markdown("### Gallery\n<small>sorted by predicted engagement</small>")
                card_btns = []
                for e in index:
                    b = gr.Button(gallery_card_label(e), size="sm")
                    card_btns.append((b, e["video_id"]))
                compare_btn = gr.Button("⚖️  Compare hit vs flop", variant="secondary", size="sm")

                with gr.Accordion("🎬  Analyze your own video (works here — no GPU)", open=True):
                    gr.Markdown(
                        "<small>The <b>Audiovisual</b> engine analyzes your clip on "
                        "CPU, offline, in seconds — real measured signals, not a "
                        "guess. <b>TRIBE brain</b> mode needs a GPU (HF Spaces).</small>"
                    )
                    upload = gr.Video(sources=["upload"], label="Short video (≤ ~30s)")
                    engine_sel = gr.Radio(
                        ["Audiovisual (CPU, works here)", "TRIBE brain (GPU / Spaces)"],
                        value="Audiovisual (CPU, works here)", label="Engine",
                    )
                    analyze_btn = gr.Button("Analyze", variant="primary", size="sm")
                    upload_status = gr.Markdown(visible=False)

            with gr.Column(scale=3):
                main = gr.HTML(build_main_view(load_result(default_id), video_url_for(default_id)))

        with gr.Accordion("Methodology & caveats", open=False):
            gr.Markdown(METHODOLOGY)
        gr.Markdown(f"<small>{FOOTER}</small>")

        # wiring: each gallery button loads its main view
        def make_loader(vid: str):
            return lambda: build_main_view(load_result(vid), video_url_for(vid))

        for btn, vid in card_btns:
            btn.click(make_loader(vid), outputs=main)

        # live upload -> analysis (audiovisual CPU by default; TRIBE optional)
        def run_upload(video_path, engine_label):
            from app.infer import analyze_upload  # lazy: keeps app torch-free

            engine = "tribe" if "TRIBE" in (engine_label or "") else "audiovisual"
            result, message = analyze_upload(video_path, engine=engine)
            if result is None:
                note = (message or "Couldn't analyze this video.").replace("\n", "<br>")
                return gr.update(), gr.update(
                    visible=True, value=f"> ⚠️ **Couldn't analyze**\n>\n> {note}"
                )
            url = f"/gradio_api/file={video_path}" if video_path else None
            return build_main_view(result, url), gr.update(visible=False, value="")

        analyze_btn.click(run_upload, inputs=[upload, engine_sel], outputs=[main, upload_status])

        if hit and flop:
            compare_btn.click(
                lambda: build_compare_view(load_result(hit["video_id"]), load_result(flop["video_id"])),
                outputs=main,
            )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(
        allowed_paths=[str(VIDEOS)],
        theme=gr.themes.Base(primary_hue="teal", neutral_hue="slate"),
    )
