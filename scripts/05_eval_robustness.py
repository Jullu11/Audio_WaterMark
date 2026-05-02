#!/usr/bin/env python3
"""
Evaluate speaker-ID accuracy under waveform attacks (robustness / BA degradation).

Uses the same manifest + checkpoint as ``04_eval_speaker_id.py``, but applies
``src.attacks.transforms`` to each batch before the forward pass.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.attacks.transforms import apply_waveform_attack, list_attack_names  # noqa: E402
from src.dataset.torch_librispeech import (  # noqa: E402
    LibriSpeechSpeakerDataset,
    load_manifest_filtered,
    pick_device,
)
from src.models.speaker_cnn import SpeakerModel  # noqa: E402


def parse_extra(arglist: list[str]) -> dict[str, float | int]:
    """Parse ``key=value`` pairs for attack hyperparameters."""
    out: dict[str, float | int] = {}
    for item in arglist:
        if "=" not in item:
            raise ValueError(f"Expected key=value, got {item!r}")
        k, v = item.split("=", 1)
        k = k.strip()
        v = v.strip()
        if v.lower() in ("true", "false"):
            raise ValueError("Boolean extras not supported; use numeric key=value only.")
        try:
            if "." in v or "e" in v.lower():
                out[k] = float(v)
            else:
                out[k] = int(v)
        except ValueError as e:
            raise ValueError(f"Bad value for {k}: {v}") from e
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Speaker-ID eval under waveform attacks.")
    p.add_argument("--manifest", type=Path, default=Path("data/processed/manifest_with_split.csv"))
    p.add_argument("--subset", default="train-clean-100")
    p.add_argument("--split", default="test")
    p.add_argument("--checkpoint", type=Path, default=Path("results/speaker_baseline/speaker_cnn.pt"))
    p.add_argument("--label-map", type=Path, default=Path("results/speaker_baseline/speaker_label_map.json"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--segment-seconds", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument(
        "--attack",
        default="none",
        help=f"Attack name. One of: {', '.join(list_attack_names())}",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        default=16_000,
        help="Must match LibriSpeech / training (16000).",
    )
    p.add_argument(
        "--attack-kw",
        nargs="*",
        default=[],
        metavar="KEY=VAL",
        help="Extra attack parameters, e.g. snr_db=10 rate=1.05 cutoff_hz=3000",
    )
    p.add_argument(
        "--metrics-json",
        type=Path,
        default=Path("results/speaker_baseline/robustness_metrics.json"),
        help="Write accuracy + attack config here.",
    )
    args = p.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1
    if not args.checkpoint.is_file():
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1
    if not args.label_map.is_file():
        print(f"Label map not found: {args.label_map}", file=sys.stderr)
        return 1

    extra = parse_extra(list(args.attack_kw))

    with open(args.label_map, encoding="utf-8") as f:
        speaker_to_idx: dict[str, int] = json.load(f)

    rows_all = load_manifest_filtered(manifest_path, subset=args.subset, split=args.split)
    rows = [r for r in rows_all if r["speaker_id"] in speaker_to_idx]
    skipped = len(rows_all) - len(rows)
    if not rows:
        print("No eval rows after filtering to known speakers.", file=sys.stderr)
        return 1

    ds = LibriSpeechSpeakerDataset(
        rows,
        speaker_to_idx,
        segment_seconds=args.segment_seconds,
        augment=False,
        seed=args.seed,
    )
    device = pick_device(None if args.device == "auto" else args.device)
    pin_memory = device.type == "cuda"
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    num_classes = int(ckpt["num_classes"])

    model = SpeakerModel(num_classes=num_classes).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    attack_name = args.attack.strip().lower()

    correct = 0
    total = 0
    with torch.no_grad():
        for wav, y in loader:
            wav = wav.to(device)
            degraded = apply_waveform_attack(
                wav,
                attack_name,
                sample_rate=args.sample_rate,
                **extra,
            )
            if degraded.device != device:
                degraded = degraded.to(device)
            y = y.to(device)
            logits = model(degraded)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()

    acc = correct / max(1, total)
    print(
        f"attack={attack_name!r}  subset={args.subset!r} split={args.split!r}  "
        f"n={len(rows)}  skipped_unknown_speaker={skipped}"
    )
    print(f"accuracy (closed-set under attack) = {acc:.6f}  device={device}  extra={extra}")

    out = {
        "attack": attack_name,
        "attack_params": extra,
        "sample_rate": args.sample_rate,
        "subset": args.subset,
        "split": args.split,
        "n_utterances": len(rows),
        "skipped_unknown_speaker": skipped,
        "accuracy": acc,
        "checkpoint": str(args.checkpoint.resolve()),
        "manifest": str(manifest_path),
    }
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    with open(args.metrics_json, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"Wrote {args.metrics_json.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
