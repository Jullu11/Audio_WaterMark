"""Unit tests for waveform attack helpers (stdlib only)."""

from __future__ import annotations

import unittest

import torch

from src.attacks.transforms import apply_waveform_attack


class TestWaveformAttacks(unittest.TestCase):
    def test_none_is_identity(self) -> None:
        x = torch.linspace(-0.5, 0.5, steps=3200)
        y = apply_waveform_attack(x, "none", sample_rate=16_000)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.allclose(y, x))

    def test_batch_shape_preserved(self) -> None:
        x = torch.randn(4, 8000)
        y = apply_waveform_attack(x, "gaussian_noise", sample_rate=16_000, snr_db=40.0)
        self.assertEqual(y.shape, x.shape)

    def test_lowpass_finite(self) -> None:
        x = torch.randn(2, 32000)
        y = apply_waveform_attack(x, "lowpass", sample_rate=16_000, cutoff_hz=3000.0)
        self.assertEqual(y.shape, x.shape)
        self.assertTrue(torch.isfinite(y).all())


if __name__ == "__main__":
    unittest.main()
