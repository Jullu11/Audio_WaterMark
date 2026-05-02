#!/usr/bin/env python3
"""
Train a closed-set speaker classifier on LibriSpeech (train-clean-100) using
manifest splits from ``02_make_splits.py --strategy utterance``.

Rows with ``subset=dev-clean`` are disjoint speakers from train-clean-100 and are
skipped here (use embedding/open-set methods later if you need dev-clean scores).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset.torch_librispeech import (  # noqa: E402
    LibriSpeechSpeakerDataset,
    build_speaker_label_map,
    load_manifest_filtered,
    pick_device,
)
from src.models.speaker_cnn import SpeakerModel  # noqa: E402


def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    pred = logits.argmax(dim=1)
    return (pred == y).float().mean().item()


def main() -> int:
    p = argparse.ArgumentParser(description="Train speaker-ID CNN on LibriSpeech manifest splits.")
    p.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_with_split.csv"))
    p.add_argument("--subset", default="train-clean-100")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--segment-seconds", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--output-dir", type=Path, default=Path("results/speaker_baseline"))
    args = p.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    torch.manual_seed(args.seed)

    # Label space = all speakers in train-clean-100 (251 classes).
    rows_subset = load_manifest_filtered(manifest_path, subset=args.subset, split=None)
    if not rows_subset:
        print(f"No rows for subset={args.subset!r}", file=sys.stderr)
        return 1
    speaker_to_idx = build_speaker_label_map(rows_subset)
    num_classes = len(speaker_to_idx)

    train_rows = load_manifest_filtered(manifest_path, subset=args.subset, split="train")
    val_rows = load_manifest_filtered(manifest_path, subset=args.subset, split="val")
    if not train_rows or not val_rows:
        print(
            "Train or val split empty. Regenerate splits with:\n"
            "  PYTHONPATH=. python scripts/02_make_splits.py --strategy utterance --write-per-split",
            file=sys.stderr,
        )
        return 1

    train_ds = LibriSpeechSpeakerDataset(
        train_rows,
        speaker_to_idx,
        segment_seconds=args.segment_seconds,
        augment=True,
        seed=args.seed,
    )
    val_ds = LibriSpeechSpeakerDataset(
        val_rows,
        speaker_to_idx,
        segment_seconds=args.segment_seconds,
        augment=False,
        seed=args.seed,
    )

    device = pick_device(None if args.device == "auto" else args.device)
    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    model = SpeakerModel(num_classes=num_classes).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.CrossEntropyLoss()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "speaker_label_map.json", "w", encoding="utf-8") as f:
        json.dump(speaker_to_idx, f, indent=2)

    best_val = 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for wav, y in train_loader:
            wav = wav.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(wav)
            loss = crit(logits, y)
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for wav, y in val_loader:
                wav = wav.to(device)
                y = y.to(device)
                logits = model(wav)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.numel()
        val_acc = correct / max(1, total)
        best_val = max(best_val, val_acc)
        print(
            f"epoch {epoch}/{args.epochs}  train_loss={total_loss / max(1, n_batches):.4f}  "
            f"val_acc={val_acc:.4f}  device={device}"
        )

    ckpt = {
        "model_state": model.state_dict(),
        "num_classes": num_classes,
        "subset": args.subset,
        "manifest": str(manifest_path),
        "best_val_acc": best_val,
    }
    torch.save(ckpt, args.output_dir / "speaker_cnn.pt")
    print(f"Saved checkpoint to {args.output_dir / 'speaker_cnn.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
