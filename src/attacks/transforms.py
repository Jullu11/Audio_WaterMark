"""Deterministic waveform degradations (noise, band-limiting, time-scale, resampling).

Applied on ``[B, T]`` float32 waveforms at fixed sample rate (default 16 kHz LibriSpeech).

Advanced attacks added:
  - deepfilter       : generative speech enhancement (DeepFilterNet)
  - asr_tts          : full signal-chain break via Whisper ASR + gTTS
  - voice_conversion : speaker identity shift via pitch shifting (librosa)
  - deepafx_style    : Adobe DeepAFx-ST style / proxy effects (needs ``deepafx_st`` + USENIX ckpts)
  - audiosr          : latent-diffusion super-resolution (needs ``audiosr`` PyPI, heavy / GPU-friendly)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor

_ROOT_ATTACKS = Path(__file__).resolve().parents[2]
_DEEPAFX_SYSTEM: Any = None
_AUDIOSR_MODEL: Any = None
_AUDIOSR_MODEL_KEY: str | None = None
_WHISPER_MODELS: dict[str, Any] = {}


def _official_checkpoint_dir() -> Path:
    return (
        _ROOT_ATTACKS
        / "external"
        / "audiowatermark.github.io"
        / "code"
        / "checkpoint"
    )


def _pad_time_dim(w1d: Tensor, target: int) -> Tensor:
    t = w1d.numel()
    if t >= target:
        return w1d[:target].contiguous()
    return F.pad(w1d, (0, target - t))


def list_attack_names() -> list[str]:
    return [
        "none",
        "gaussian_noise",
        "time_stretch",
        "lowpass",
        "highpass",
        "resample_chain",
        "quantize",
        # --- NEW advanced attacks ---
        "deepfilter",
        "asr_tts",
        "voice_conversion",
        "deepafx_style",
        "audiosr",
        # --- Figure-12 style data-level attacks ---
        "strip",
        "shrinkpad",
    ]


def _ensure_bt(wav: Tensor) -> Tensor:
    if wav.dim() == 1:
        return wav.unsqueeze(0)
    return wav


def apply_waveform_attack(wav: Tensor, attack: str, sample_rate: int = 16_000, **kw: Any) -> Tensor:
    """
    Apply a named attack on a copy and return the degraded waveform.

    Parameters
    ----------
    wav : Tensor
        Shape ``[B, T]`` or ``[T]``.
    attack : str
        One of :func:`list_attack_names`.
    sample_rate : int
        Nominal Hz (LibriSpeech = 16000).
    **kw
        Attack-specific options (see branches below).

    Advanced attack parameters
    --------------------------
    deepfilter:
        atten_lim_db (float) — 6=light, 15=medium, 30=heavy  (default 15)

    asr_tts:
        quality (str) — "light" (tiny model), "medium" (base), "heavy" (small)
                        (default "medium")

    voice_conversion:
        n_steps (float) — semitones to shift: smaller is milder
                        (default 6)

    deepafx_style:
        max_input (int) — input length passed to DeepAFx (default 80217, USENIX layout)
        max_ref   (int) — reference length (default 89769)

    audiosr:
        model_name (str) — ``speech`` or ``basic`` (default ``speech``)
        ddim_steps (int)  — diffusion steps; lower is faster (default 20)
        guidance_scale (float) — default 3.5
    """
    name = attack.strip().lower()
    x = _ensure_bt(wav.clone())

    if name in ("none", "clean"):
        return x if wav.dim() == 2 else x.squeeze(0)

    # ------------------------------------------------------------------
    # Original attacks
    # ------------------------------------------------------------------
    if name == "gaussian_noise":
        snr_db = float(kw.get("snr_db", 20.0))
        eps = 1e-12
        p_sig = (x**2).mean(dim=-1, keepdim=True).clamp_min(eps)
        p_noise = p_sig / (10.0 ** (snr_db / 10.0))
        noise = torch.randn_like(x) * torch.sqrt(p_noise)
        out = (x + noise).clamp(-1.0, 1.0)

    elif name == "time_stretch":
        rate = float(kw.get("rate", 1.1))
        if rate <= 0:
            raise ValueError("time_stretch rate must be positive")
        _, t = x.shape
        new_t = max(1, int(round(t / rate)))
        xi = x.unsqueeze(1)
        y = F.interpolate(xi, size=new_t, mode="linear", align_corners=False).squeeze(1)
        if new_t >= t:
            start = (new_t - t) // 2
            out = y[:, start: start + t]
        else:
            out = F.pad(y, (0, t - new_t))

    elif name == "lowpass":
        cutoff_hz = float(kw.get("cutoff_hz", 4000.0))
        out = _biquad_lowpass_batch(x, sample_rate, cutoff_hz)

    elif name == "highpass":
        cutoff_hz = float(kw.get("cutoff_hz", 200.0))
        out = _biquad_highpass_batch(x, sample_rate, cutoff_hz)

    elif name == "resample_chain":
        mid_sr = int(kw.get("mid_sr", 8000))
        out = _resample_chain(x, sample_rate, mid_sr)

    elif name == "quantize":
        levels = int(kw.get("levels", 256))
        if levels < 2:
            raise ValueError("quantize levels must be >= 2")
        scale = (levels - 1) / 2.0
        out = torch.round(x.clamp(-1.0, 1.0) * scale) / scale

    # ------------------------------------------------------------------
    # Figure-12 style baseline: STRIP
    # Blend random pattern into the waveform to reduce consistency.
    # Strength via blend:
    #   light=0.05, medium=0.10, heavy=0.20
    # ------------------------------------------------------------------
    elif name == "strip":
        blend_strength = float(kw.get("blend", 0.10))
        blend_strength = max(0.0, min(1.0, blend_strength))
        noise = torch.randn_like(x)
        out = ((1.0 - blend_strength) * x + blend_strength * noise).clamp(-1.0, 1.0)

    # ------------------------------------------------------------------
    # Figure-12 style baseline: ShrinkPad
    # Temporally shrink then right-pad back to original length.
    # Strength via ratio:
    #   light=0.95, medium=0.90, heavy=0.80
    # ------------------------------------------------------------------
    elif name == "shrinkpad":
        ratio = float(kw.get("ratio", 0.90))
        ratio = max(0.05, min(1.0, ratio))
        _, t = x.shape
        new_t = max(1, int(round(t * ratio)))
        xi = x.unsqueeze(1)
        shrunk = F.interpolate(
            xi, size=new_t, mode="linear", align_corners=False
        ).squeeze(1)
        if new_t >= t:
            out = shrunk[:, :t]
        else:
            out = F.pad(shrunk, (0, t - new_t))

    # ------------------------------------------------------------------
    # NEW Attack 1: DeepFilterNet — generative speech enhancement
    # Regenerates parts of the audio signal using a neural network.
    # Potentially overwrites watermark components.
    #
    # Strength via atten_lim_db:
    #   light=6, medium=15, heavy=30
    # ------------------------------------------------------------------
    elif name == "deepfilter":
        try:
            import numpy as np
            import soundfile as sf
            import tempfile, os
            from df import enhance, init_df

            atten_lim_db = float(kw.get("atten_lim_db", 15.0))
            # Process each item in batch
            results = []
            for i in range(x.size(0)):
                wav_np = x[i].cpu().numpy().astype(np.float32)
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    sf.write(tmp.name, wav_np, sample_rate)
                    tmp_path = tmp.name
                try:
                    model_df, df_state, _ = init_df()
                    enhanced = enhance(
                        model_df, df_state, tmp_path,
                        atten_lim_db=atten_lim_db
                    )
                finally:
                    os.unlink(tmp_path)

                enhanced_t = torch.tensor(
                    enhanced, dtype=torch.float32
                )
                t = x.size(1)
                if enhanced_t.size(0) >= t:
                    enhanced_t = enhanced_t[:t]
                else:
                    enhanced_t = F.pad(enhanced_t, (0, t - enhanced_t.size(0)))
                results.append(enhanced_t)
            out = torch.stack(results, dim=0)

        except Exception as e:
            raise RuntimeError(
                "[deepfilter] dependency/runtime failure. "
                "Install compatible deepfilternet + torchaudio stack. "
                f"Original error: {e}"
            ) from e

    # ------------------------------------------------------------------
    # NEW Attack 2: ASR + TTS re-synthesis
    # Transcribes watermarked audio to text via Whisper, then synthesises
    # brand-new audio via gTTS.  Completely breaks the signal chain —
    # the output carries NO acoustic watermark.
    #
    # Strength via quality:
    #   "light"  = Whisper tiny model
    #   "medium" = Whisper base  (default)
    #   "heavy"  = Whisper small
    # ------------------------------------------------------------------
    elif name == "asr_tts":
        try:
            import gc
            import numpy as np
            import soundfile as sf
            import tempfile, os
            import whisper
            from gtts import gTTS
            from pydub import AudioSegment

            quality = str(kw.get("quality", "medium")).strip().lower()
            model_size = {"light": "tiny", "medium": "base", "heavy": "small"}
            key = model_size.get(quality, "base")
            # Whisper on CPU/macOS: fp16=False avoids FP16 warnings and reduces crash risk;
            # keep the model on CUDA only when actually available.
            _asr_dev = "cuda" if torch.cuda.is_available() else "cpu"
            _use_fp16 = _asr_dev == "cuda"
            if key not in _WHISPER_MODELS:
                _WHISPER_MODELS[key] = whisper.load_model(key, device=_asr_dev)
            asr = _WHISPER_MODELS[key]

            results = []
            for i in range(x.size(0)):
                wav_np = x[i].cpu().numpy().astype(np.float32)

                # Step 1: audio → text
                with tempfile.NamedTemporaryFile(
                    suffix=".wav", delete=False
                ) as tmp:
                    sf.write(tmp.name, wav_np, sample_rate)
                    tmp_path = tmp.name
                try:
                    result = asr.transcribe(
                        tmp_path, fp16=_use_fp16, verbose=False
                    )
                    text = result["text"].strip()
                finally:
                    os.unlink(tmp_path)

                if not text:
                    # Nothing transcribed — keep original
                    results.append(x[i])
                    continue

                # Step 2: text → brand-new audio (gTTS)
                with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_mp3:
                    mp3_path = tmp_mp3.name
                try:
                    gTTS(text=text, lang="en").save(mp3_path)
                    audio = (
                        AudioSegment.from_mp3(mp3_path)
                        .set_frame_rate(sample_rate)
                        .set_channels(1)
                    )
                    wav_out = np.array(
                        audio.get_array_of_samples(), dtype=np.float32
                    ) / 32768.0
                finally:
                    if os.path.exists(mp3_path):
                        os.unlink(mp3_path)

                new_t = torch.tensor(wav_out, dtype=torch.float32)
                t = x.size(1)
                if new_t.size(0) >= t:
                    new_t = new_t[:t]
                else:
                    new_t = F.pad(new_t, (0, t - new_t.size(0)))
                results.append(new_t)
                gc.collect()

            out = torch.stack(results, dim=0)

        except Exception as e:
            raise RuntimeError(
                "[asr_tts] dependency/runtime failure. "
                "Install openai-whisper gTTS pydub and system ffmpeg. "
                f"Original error: {e}"
            ) from e

    # ------------------------------------------------------------------
    # NEW Attack 3a: DeepAFx-ST (Adobe) — neural audio effects / style path
    # Same stack as the USENIX AUDIO WATERMARK repo (needs their checkpoints).
    # Env overrides: DEEPAFX_STYLE_CKPT, DEEPAFX_PEQ_CKPT, DEEPAFX_COMP_CKPT
    # Install: pip install "git+https://github.com/adobe-research/DeepAFx-ST.git"
    # ------------------------------------------------------------------
    elif name == "deepafx_style":
        try:
            from deepafx_st.system import System
            from deepafx_st.utils import DSPMode
        except ImportError as e:
            raise RuntimeError(
                "[deepafx_style] Install DeepAFx-ST: "
                "pip install \"git+https://github.com/adobe-research/DeepAFx-ST.git\" "
                f"(original error: {e})"
            ) from e

        ckpt_dir = _official_checkpoint_dir()
        style_ckpt = os.environ.get(
            "DEEPAFX_STYLE_CKPT",
            str(ckpt_dir / "deepafx_style.ckpt"),
        )
        peq_ckpt = os.environ.get(
            "DEEPAFX_PEQ_CKPT",
            str(ckpt_dir / "deepafx_peq.ckpt"),
        )
        comp_ckpt = os.environ.get(
            "DEEPAFX_COMP_CKPT",
            str(ckpt_dir / "deepafx_comp.ckpt"),
        )
        for label, path in (
            ("style", style_ckpt),
            ("peq", peq_ckpt),
            ("comp", comp_ckpt),
        ):
            if not Path(path).is_file():
                raise RuntimeError(
                    "[deepafx_style] Missing checkpoint "
                    f"{label} at {path!r}. Download the USENIX AUDIO WATERMARK "
                    "checkpoint bundle (see external/audiowatermark.github.io/README.md) "
                    "or set DEEPAFX_*_CKPT env vars."
                )

        global _DEEPAFX_SYSTEM
        dev = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        if _DEEPAFX_SYSTEM is None:
            try:
                loaded = System.load_from_checkpoint(
                    style_ckpt,
                    dsp_mode=DSPMode.INFER,
                    proxy_ckpts=[peq_ckpt, comp_ckpt],
                    map_location="cpu",
                )
            except Exception as e:
                raise RuntimeError(
                    "[deepafx_style] Failed to load DeepAFx-ST checkpoint "
                    f"(torch / lightning version mismatch is common on Apple Silicon). "
                    f"Original error: {e}"
                ) from e
            _DEEPAFX_SYSTEM = loaded.eval().to(dev)

        model = _DEEPAFX_SYSTEM
        l_in = int(kw.get("max_input", 80217))
        l_ref = int(kw.get("max_ref", 89769))

        results = []
        bsz = x.size(0)
        for i in range(bsz):
            wav = x[i].detach().float().cpu()
            ref_src = x[(i + 1) % bsz].detach().float().cpu()
            seg = _pad_time_dim(wav, l_in)
            ref = _pad_time_dim(ref_src, l_ref)
            xi = seg.view(1, 1, -1).to(dev)
            yi = ref.view(1, 1, -1).to(dev)
            with torch.no_grad():
                y_hat, _, _ = model(
                    xi,
                    y=yi,
                    dsp_mode=DSPMode.INFER,
                    sample_rate=sample_rate,
                )
            y1 = y_hat.squeeze(0).squeeze(0).detach().cpu()
            t_orig = x.size(1)
            if y1.numel() >= t_orig:
                y1 = y1[:t_orig]
            else:
                y1 = F.pad(y1, (0, t_orig - y1.numel()))
            results.append(y1.to(dtype=x.dtype, device=x.device))
        out = torch.stack(results, dim=0)

    # ------------------------------------------------------------------
    # NEW Attack 3b: AudioSR — diffusion super-resolution (48 kHz internal)
    # pip install audiosr   (first run downloads weights; use small ddim_steps on CPU)
    # ------------------------------------------------------------------
    elif name == "audiosr":
        global _AUDIOSR_MODEL, _AUDIOSR_MODEL_KEY
        try:
            from audiosr import build_model, super_resolution
        except ImportError as e:
            raise RuntimeError(
                "[audiosr] pip install audiosr  "
                "(see https://pypi.org/project/audiosr/). "
                f"Original error: {e}"
            ) from e

        import tempfile

        import numpy as np
        import soundfile as sf
        import torchaudio

        model_name = str(kw.get("model_name", "speech")).strip().lower()
        if model_name not in ("speech", "basic"):
            model_name = "speech"
        ddim_steps = int(kw.get("ddim_steps", 20))
        guidance_scale = float(kw.get("guidance_scale", 3.5))
        seed = int(kw.get("seed", 42))

        infer_dev = (
            "cuda"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        cache_key = f"{model_name}:{infer_dev}"
        if _AUDIOSR_MODEL is None or _AUDIOSR_MODEL_KEY != cache_key:
            _AUDIOSR_MODEL = build_model(model_name=model_name, device=infer_dev)
            _AUDIOSR_MODEL_KEY = cache_key

        sr_model = _AUDIOSR_MODEL
        up_sr = 48_000
        results = []
        for i in range(x.size(0)):
            wav_np = x[i].detach().cpu().numpy().astype(np.float32)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, wav_np, sample_rate)
                tmp_path = tmp.name
            try:
                gen = super_resolution(
                    sr_model,
                    tmp_path,
                    seed=seed + i,
                    ddim_steps=ddim_steps,
                    guidance_scale=guidance_scale,
                )
            finally:
                os.unlink(tmp_path)

            if isinstance(gen, torch.Tensor):
                w48 = gen.detach().float().cpu().squeeze()
            else:
                w48 = torch.tensor(np.asarray(gen), dtype=torch.float32).squeeze()

            if w48.dim() > 1:
                w48 = w48.reshape(-1)
            w16 = torchaudio.functional.resample(
                w48.unsqueeze(0), up_sr, sample_rate
            ).squeeze(0)
            t_orig = x.size(1)
            if w16.numel() >= t_orig:
                w16 = w16[:t_orig]
            else:
                w16 = F.pad(w16, (0, t_orig - w16.numel()))
            results.append(w16.to(dtype=x.dtype, device=x.device))
        out = torch.stack(results, dim=0)

    # ------------------------------------------------------------------
    # NEW Attack 3: Voice conversion via pitch shifting
    # Shifts speaker pitch to alter identity features (LTAF-based
    # watermark lives in speaker-style features).
    #
    # Strength via n_steps (semitones):
    #   light=3, medium=6, heavy=10  (default 6)
    # ------------------------------------------------------------------
    elif name == "voice_conversion":
        try:
            import numpy as np
            import librosa

            n_steps = float(kw.get("n_steps", 6.0))
            results = []
            for i in range(x.size(0)):
                wav_np = x[i].cpu().numpy().astype(np.float32)
                shifted = librosa.effects.pitch_shift(
                    wav_np, sr=sample_rate, n_steps=n_steps
                )
                shifted_t = torch.tensor(shifted, dtype=torch.float32)
                t = x.size(1)
                if shifted_t.size(0) >= t:
                    shifted_t = shifted_t[:t]
                else:
                    shifted_t = F.pad(shifted_t, (0, t - shifted_t.size(0)))
                results.append(shifted_t)
            out = torch.stack(results, dim=0)

        except ImportError as e:
            raise RuntimeError(
                "[voice_conversion] librosa is required for pitch-shift VC. "
                "Install: pip install librosa\n"
                f"(original error: {e})"
            ) from e

    else:
        raise ValueError(
            f"Unknown attack {attack!r}. Choose one of {list_attack_names()}"
        )

    return out if wav.dim() == 2 else out.squeeze(0)


# ------------------------------------------------------------------
# Helpers (unchanged)
# ------------------------------------------------------------------

def _biquad_lowpass_batch(x: Tensor, sr: int, cutoff_hz: float) -> Tensor:
    from scipy import signal
    cutoff_hz = min(cutoff_hz, sr * 0.499)
    b, a = signal.butter(4, cutoff_hz / (sr / 2), btype="low")
    return _apply_lfilter_batch(x, b, a)


def _biquad_highpass_batch(x: Tensor, sr: int, cutoff_hz: float) -> Tensor:
    from scipy import signal
    cutoff_hz = max(cutoff_hz, sr * 1e-4)
    b, a = signal.butter(4, cutoff_hz / (sr / 2), btype="high")
    return _apply_lfilter_batch(x, b, a)


def _apply_lfilter_batch(x: Tensor, b, a) -> Tensor:
    import numpy as np
    from scipy import signal
    xp = x.detach().cpu().numpy().astype(np.float64)
    out = signal.filtfilt(b, a, xp, axis=-1, padlen=3 * max(len(a), len(b)))
    return torch.from_numpy(out.astype(np.float32)).to(device=x.device, dtype=x.dtype)


def _resample_chain(x: Tensor, orig_sr: int, mid_sr: int) -> Tensor:
    import torchaudio
    if mid_sr <= 0 or mid_sr >= orig_sr:
        return x
    b, _ = x.shape
    y_list = []
    for i in range(b):
        seg = x[i: i + 1]
        down = torchaudio.functional.resample(seg, orig_sr, mid_sr)
        up = torchaudio.functional.resample(down, mid_sr, orig_sr)
        y_list.append(up.squeeze(0))
    out = torch.stack(y_list, dim=0)
    t = x.size(1)
    if out.size(1) >= t:
        start = (out.size(1) - t) // 2
        out = out[:, start: start + t]
    else:
        out = F.pad(out, (0, t - out.size(1)))
    return out
