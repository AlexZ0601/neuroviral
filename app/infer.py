"""Live-upload inference adapter.

Two engines, same output schema:

  * "audiovisual" (default) — CPU signal-processing engine. Runs anywhere,
    including a laptop, offline, in seconds. This is the real working path.
  * "tribe" — Meta TRIBE fMRI prediction. GPU-only; runs on HF Spaces (ZeroGPU)
    inside an @spaces.GPU allocation, or a local CUDA box. Optional research
    mode. Off-GPU it returns an honest message instead of fabricating data.

app.py imports this lazily so it never pulls torch (or cv2/librosa) at module
load — the heavy imports happen inside the handlers, only when someone uploads.
"""
from __future__ import annotations

from pathlib import Path

# Optional HF Spaces ZeroGPU decorator (TRIBE path only).
try:  # pragma: no cover - depends on deploy target
    import spaces  # type: ignore

    IN_SPACES = True
    _gpu = spaces.GPU(duration=120)
except Exception:
    IN_SPACES = False

    def _gpu(fn):  # type: ignore
        return fn


DEPLOY_HINT = (
    "TRIBE (research mode) needs a CUDA GPU. The upload → TRIBE → render path is "
    "wired and `@spaces.GPU`-ready — deploy to a Hugging Face Space with ZeroGPU "
    "to run it. The audiovisual engine (default) works here with no GPU."
)


# --- audiovisual (CPU) : the real working path -------------------------------

def analyze_upload_cpu(video_path: str | None) -> tuple[dict | None, str]:
    """Analyze an uploaded video on CPU. Works anywhere; never fakes data."""
    if not video_path:
        return None, "No video uploaded yet."
    try:
        from pipeline.cpu_engine import analyze_video_cpu  # lazy: cv2/librosa
    except Exception as e:
        return None, f"Audiovisual engine unavailable (missing dependency: {e})."
    try:
        result = analyze_video_cpu(Path(video_path), title=Path(video_path).stem)
        if not result.timeline:
            return None, "Couldn't read frames/audio from this file."
        return result.to_dict(), ""
    except Exception as e:
        return None, f"Analysis failed: {e}"


# --- tribe (GPU) : optional research mode ------------------------------------

def _tribe_preflight() -> tuple[bool, str]:
    try:
        import torch  # noqa: F401
    except Exception:
        return False, "PyTorch isn't installed in this environment."
    try:
        import tribev2  # type: ignore  # noqa: F401
    except Exception:
        return False, "The `tribev2` package isn't installed here."
    # Off-Spaces we can trust cuda.is_available(); on ZeroGPU we can't (CUDA is
    # only bound inside the @spaces.GPU call), so skip that check there.
    if not IN_SPACES:
        import torch

        if not torch.cuda.is_available():
            return False, "No CUDA GPU is available on this machine."
    return True, ""


@_gpu
def _run_tribe(video_path: str) -> dict:
    from pipeline.extract import analyze_video

    return analyze_video(Path(video_path), title=Path(video_path).stem).to_dict()


def analyze_upload_tribe(video_path: str | None) -> tuple[dict | None, str]:
    if not video_path:
        return None, "No video uploaded yet."
    ok, why = _tribe_preflight()
    if not ok:
        return None, f"{why}\n\n{DEPLOY_HINT}"
    try:
        return _run_tribe(video_path), ""
    except Exception as e:
        return None, f"TRIBE inference failed: {e}\n\n{DEPLOY_HINT}"


# --- dispatch ----------------------------------------------------------------

def analyze_upload(video_path: str | None, engine: str = "audiovisual") -> tuple[dict | None, str]:
    if engine == "tribe":
        return analyze_upload_tribe(video_path)
    return analyze_upload_cpu(video_path)
