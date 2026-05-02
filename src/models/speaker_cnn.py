"""Lightweight CNN speaker classifier on log-mel spectrograms."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LogMelFrontend(nn.Module):
    """Waveform [B, T] -> log-mel [B, n_mels, time]."""

    def __init__(self, sample_rate: int = 16_000, n_mels: int = 80) -> None:
        super().__init__()
        import torchaudio

        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=n_mels,
            f_min=20,
            f_max=7600,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB()

    def forward(self, wav: Tensor) -> Tensor:
        x = wav.unsqueeze(1)
        m = self.mel(x)
        # Mono batched input yields ``[B, 1, n_mels, time]``; Conv2d expects channel once.
        if m.dim() == 4 and m.size(1) == 1:
            m = m.squeeze(1)
        return self.to_db(m + 1e-6)


class SpeakerCNN(nn.Module):
    """Classifier on mel ``[B, n_mels, time]``."""

    def __init__(self, num_classes: int, n_mels: int = 80) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 2)),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, mel: Tensor) -> Tensor:
        x = mel.unsqueeze(1)
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.head(x)


class SpeakerModel(nn.Module):
    """End-to-end: waveform -> logits."""

    def __init__(self, num_classes: int, sample_rate: int = 16_000, n_mels: int = 80) -> None:
        super().__init__()
        self.front = LogMelFrontend(sample_rate=sample_rate, n_mels=n_mels)
        self.back = SpeakerCNN(num_classes=num_classes, n_mels=n_mels)

    def forward(self, wav: Tensor) -> Tensor:
        mel = self.front(wav)
        return self.back(mel)
