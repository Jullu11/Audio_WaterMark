"""PyTorch Dataset for LibriSpeech FLAC rows from manifest CSV."""

from __future__ import annotations

import csv
import random
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset


def load_manifest_filtered(
    path: Path,
    *,
    subset: str,
    split: str | None = None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("subset") != subset:
                continue
            if split is not None and r.get("split") != split:
                continue
            rows.append(r)
    return rows


def build_speaker_label_map(rows: list[dict[str, str]]) -> dict[str, int]:
    """Stable mapping speaker_id -> class index."""
    speakers = sorted({r["speaker_id"] for r in rows})
    return {s: i for i, s in enumerate(speakers)}


class LibriSpeechSpeakerDataset(Dataset):
    """
    Loads mono FLAC at native LibriSpeech sample rate (16 kHz).
    Crops / pads to ``segment_samples`` for fixed-length batches.
    """

    def __init__(
        self,
        rows: list[dict[str, str]],
        speaker_to_idx: dict[str, int],
        *,
        segment_seconds: float = 2.0,
        sample_rate: int = 16_000,
        augment: bool = False,
        seed: int = 0,
    ) -> None:
        self.rows = rows
        self.speaker_to_idx = speaker_to_idx
        self.segment_samples = int(segment_seconds * sample_rate)
        self.sample_rate = sample_rate
        self.augment = augment
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        import soundfile as sf
        import torchaudio

        r = self.rows[idx]
        path = Path(r["path"])
        # soundfile reads FLAC without TorchCodec (torchaudio.load may require torchcodec on some stacks).
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
        x = torch.from_numpy(data)
        if x.dim() == 2:
            x = x.mean(dim=1)
        if sr != self.sample_rate:
            x = torchaudio.functional.resample(x.unsqueeze(0), sr, self.sample_rate).squeeze(0)
        t = x.numel()
        if t >= self.segment_samples:
            if self.augment:
                start = self._rng.randint(0, t - self.segment_samples)
            else:
                start = max(0, (t - self.segment_samples) // 2)
            x = x[start : start + self.segment_samples]
        else:
            pad = self.segment_samples - t
            x = torch.nn.functional.pad(x, (0, pad))

        label = self.speaker_to_idx[r["speaker_id"]]
        return x, torch.tensor(label, dtype=torch.long)


def pick_device(preferred: str | None = None) -> torch.device:
    if preferred and preferred != "auto":
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
