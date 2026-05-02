"""Waveform-level attacks / degradations for robustness evaluation."""

from __future__ import annotations

from src.attacks.transforms import apply_waveform_attack, list_attack_names

__all__ = ["apply_waveform_attack", "list_attack_names"]
