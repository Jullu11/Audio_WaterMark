"""Deterministic waveform degradations (noise, band-limiting, time-scale, resampling).

Applied on ``[B, T]`` float32 waveforms at fixed sample rate (default 16 kHz LibriSpeech).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor


def list_attack_names() -> list[str]:
    return [
        "none",
        "gaussian_noise",
        "time_stretch",
        "lowpass",
        "highpass",
        "resample_chain",
        "quantize",
    ]


def _ensure_bt(wav: Tensor) -> Tensor:
    if wav.dim() == 1:
        return wav.unsqueeze(0)
    return wav


def apply_waveform_attack(wav: Tensor, attack: str, sample_rate: int = 16_000, **kw: Any) -> Tensor:
    """
    Apply a named attack in-place on a copy and return the degraded waveform.

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
    """
    name = attack.strip().lower()
    x = _ensure_bt(wav.clone())
    if name in ("none", "clean"):
        return x if wav.dim() == 2 else x.squeeze(0)

    if name == "gaussian_noise":
        snr_db = float(kw.get("snr_db", 20.0))
        eps = 1e-12
        p_sig = (x**2).mean(dim=-1, keepdim=True).clamp_min(eps)
        p_noise = p_sig / (10.0 ** (snr_db / 10.0))
        noise = torch.randn_like(x) * torch.sqrt(p_noise)
        y = x + noise
        out = y.clamp(-1.0, 1.0)
    elif name == "time_stretch":
        rate = float(kw.get("rate", 1.1))
        if rate <= 0:
            raise ValueError("time_stretch rate must be positive")
        # Interpret as playback speed: rate>1 => shorter duration => upsample then crop to T
        _, t = x.shape
        new_t = max(1, int(round(t / rate)))
        xi = x.unsqueeze(1)
        y = F.interpolate(xi, size=new_t, mode="linear", align_corners=False).squeeze(1)
        if new_t >= t:
            start = (new_t - t) // 2
            out = y[:, start : start + t]
        else:
            pad = t - new_t
            out = F.pad(y, (0, pad))
    elif name == "lowpass":
        cutoff_hz = float(kw.get("cutoff_hz", 4000.0))
        out = _biquad_lowpass_batch(x, sample_rate, cutoff_hz)
    elif name == "highpass":
        cutoff_hz = float(kw.get("cutoff_hz", 200.0))
        out = _biquad_highpass_batch(x, sample_rate, cutoff_hz)
    elif name == "resample_chain":
        # Downsample to intermediate rate then back (anti-alias simulation).
        mid_sr = int(kw.get("mid_sr", 8000))
        out = _resample_chain(x, sample_rate, mid_sr)
    elif name == "quantize":
        levels = int(kw.get("levels", 256))
        if levels < 2:
            raise ValueError("quantize levels must be >= 2")
        scale = (levels - 1) / 2.0
        out = torch.round(x.clamp(-1.0, 1.0) * scale) / scale
    else:
        raise ValueError(f"Unknown attack {attack!r}. Choose one of {list_attack_names()}")

    if wav.dim() == 1:
        return out.squeeze(0)
    return out


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
    """Apply zero-phase IIR along time axis (batch vectorized on CPU)."""
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
        seg = x[i : i + 1]
        down = torchaudio.functional.resample(seg, orig_sr, mid_sr)
        up = torchaudio.functional.resample(down, mid_sr, orig_sr)
        y_list.append(up.squeeze(0))
    out = torch.stack(y_list, dim=0)
    t = x.size(1)
    if out.size(1) >= t:
        start = (out.size(1) - t) // 2
        out = out[:, start : start + t]
    else:
        pad = t - out.size(1)
        out = F.pad(out, (0, pad))
    return out
