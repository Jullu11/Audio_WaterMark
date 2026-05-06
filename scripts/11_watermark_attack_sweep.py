#!/usr/bin/env python3
"""
11_watermark_attack_sweep.py  (FIXED)
─────────────────────────────
Fixes applied vs original:
  1. Removed duplicate code (entire script was written twice)
  2. compute_BA now uses imported function instead of inline sum
  3. preds_attacked computed in ONE batched pass (was one-by-one — very slow)
  4. clean baseline probs collected ONCE and reused across all attacks

IMPORTANT — metrics semantics
------------------------------
- **BA** is closed-set speaker-ID accuracy (trained CNN), a **utility / survivability** proxy.
- **VSR** here uses :func:`compute_VSR` on *clean vs attacked* **speaker softmax probs**
  for the true label—the same *statistical* procedure as the paper, but **not** the
  official watermark detector. Label this **proxy VSR** (or “verification-style score”)
  unless you pipe in verifier logits from ``external/.../verify_watermark.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.attacks.transforms import apply_waveform_attack, list_attack_names
from src.dataset.torch_librispeech import (
    LibriSpeechSpeakerDataset,
    load_manifest_filtered,
    pick_device,
)
from src.metrics import (
    compute_BA,
    compute_MCD,
    compute_VSR,
    compute_harmful_degree,
)
from src.models.speaker_cnn import SpeakerModel

_SWEEP_EPILOG = """
Examples (each command runs all selected grid rows in one Python process, back-to-back):

  Full default grid (all strengths, one continuous run; omits brittle deps unless flagged):
    python scripts/11_watermark_attack_sweep.py --mode default \\
      --out-csv results/tables/attack_sweep.csv

  Include deepfilter / deepafx_style / audiosr (may fail without installs):
    python scripts/11_watermark_attack_sweep.py --mode default \\
      --with-optional-dep-attacks --out-csv results/tables/attack_sweep.csv

  One row per attack name (faster; still one continuous run):
    python scripts/11_watermark_attack_sweep.py --mode default --one-per-attack \\
      --out-csv results/tables/attack_sweep.csv

  Smoke test (limit batches per attack):
    python scripts/11_watermark_attack_sweep.py --mode default --max-batches 2 \\
      --out-csv results/tables/attack_sweep.csv
"""

# ── attack grid ──────────────────────────────────────────────────────────────
# Optional-dep attacks often fail on stock Mac/Python (deepfilternet/torchaudio,
# DeepAFX ckpts, audiosr weights). Default sweep omits them; use
# --with-optional-dep-attacks to include.

OPTIONAL_DEP_ATTACK_NAMES = frozenset({"deepfilter", "deepafx_style", "audiosr"})

ATTACK_GRID: list[tuple[str, dict, str]] = [
    # Figure-12 baselines (original paper)
    ("none",             {},                    "baseline"),
    ("gaussian_noise",   {"snr_db": 30},        "light"),
    ("gaussian_noise",   {"snr_db": 20},        "medium"),
    ("gaussian_noise",   {"snr_db": 10},        "heavy"),
    ("resample_chain",   {"mid_sr": 16000},     "light"),
    ("resample_chain",   {"mid_sr": 8000},      "medium"),
    ("resample_chain",   {"mid_sr": 4000},      "heavy"),
    ("lowpass",          {"cutoff_hz": 4000},   "light"),
    ("lowpass",          {"cutoff_hz": 2000},   "medium"),
    ("lowpass",          {"cutoff_hz": 1000},   "heavy"),
    ("quantize",         {"levels": 256},       "light"),
    ("quantize",         {"levels": 64},        "medium"),
    ("quantize",         {"levels": 16},        "heavy"),
    ("strip",            {"blend": 0.05},       "light"),
    ("strip",            {"blend": 0.10},       "medium"),
    ("strip",            {"blend": 0.20},       "heavy"),
    ("shrinkpad",        {"ratio": 0.95},       "light"),
    ("shrinkpad",        {"ratio": 0.90},       "medium"),
    ("shrinkpad",        {"ratio": 0.80},       "heavy"),
    # NEW advanced attacks (reliable in minimal env)
    ("voice_conversion", {"n_steps": 1},        "light"),
    ("voice_conversion", {"n_steps": 3},        "medium"),
    ("voice_conversion", {"n_steps": 5},        "heavy"),
    ("asr_tts",          {"quality": "light"},  "light"),
    ("asr_tts",          {"quality": "medium"}, "medium"),
    ("asr_tts",          {"quality": "heavy"},  "heavy"),
    # Optional heavy / brittle deps (see --with-optional-dep-attacks)
    ("deepfilter",       {"atten_lim_db": 6},   "light"),
    ("deepfilter",       {"atten_lim_db": 15},  "medium"),
    ("deepfilter",       {"atten_lim_db": 30},  "heavy"),
    ("deepafx_style",    {"max_input": 80217, "max_ref": 89769}, "medium"),
    ("audiosr",          {"model_name": "speech", "ddim_steps": 15}, "light"),
]


def _attack_grid_for_run(
    include_optional_dep: bool,
) -> list[tuple[str, dict, str]]:
    if include_optional_dep:
        return list(ATTACK_GRID)
    return [row for row in ATTACK_GRID if row[0] not in OPTIONAL_DEP_ATTACK_NAMES]

# Fine-grained grid for BA/VSR trade-off exploration.
FOCUSED_SEARCH_GRID: list[tuple[str, dict, str]] = [
    ("none",             {},                    "baseline"),
    ("shrinkpad",        {"ratio": 0.995},      "s1"),
    ("shrinkpad",        {"ratio": 0.99},       "s2"),
    ("shrinkpad",        {"ratio": 0.985},      "s3"),
    ("shrinkpad",        {"ratio": 0.98},       "s4"),
    ("shrinkpad",        {"ratio": 0.975},      "s5"),
    ("shrinkpad",        {"ratio": 0.97},       "s6"),
    ("shrinkpad",        {"ratio": 0.965},      "s7"),
    ("strip",            {"blend": 0.005},      "s1"),
    ("strip",            {"blend": 0.01},       "s2"),
    ("strip",            {"blend": 0.015},      "s3"),
    ("strip",            {"blend": 0.02},       "s4"),
    ("gaussian_noise",   {"snr_db": 50},        "s1"),
    ("gaussian_noise",   {"snr_db": 45},        "s2"),
    ("gaussian_noise",   {"snr_db": 40},        "s3"),
    ("gaussian_noise",   {"snr_db": 35},        "s4"),
    ("resample_chain",   {"mid_sr": 15_900},    "s1"),
    ("resample_chain",   {"mid_sr": 15_000},    "s2"),
    ("resample_chain",   {"mid_sr": 14_000},    "s3"),
    ("voice_conversion", {"n_steps": 0.5},      "s1"),
    ("voice_conversion", {"n_steps": 1.0},      "s2"),
    ("voice_conversion", {"n_steps": 1.5},      "s3"),
]


def load_model(checkpoint: Path, device: torch.device) -> SpeakerModel:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = SpeakerModel(num_classes=int(ckpt["num_classes"])).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def collect_samples(
    loader: DataLoader,
    attack: str,
    params: dict,
    device: torch.device,
    sample_rate: int = 16_000,
    max_batches: int | None = None,
) -> tuple[list[np.ndarray], list[np.ndarray], list[int]]:
    """Collect clean + attacked waveforms and labels."""
    clean_list, attacked_list, label_list = [], [], []
    with torch.no_grad():
        for batch_idx, (wav, y) in enumerate(loader):
            if max_batches and batch_idx >= max_batches:
                break
            wav = wav.to(device)
            attacked = apply_waveform_attack(
                wav, attack, sample_rate=sample_rate, **params
            ).to(device)
            for i in range(wav.size(0)):
                clean_list.append(wav[i].cpu().numpy())
                attacked_list.append(attacked[i].cpu().numpy())
                label_list.append(int(y[i]))
    return clean_list, attacked_list, label_list


def _flush_results_csv(out_csv: Path, rows: list) -> None:
    """Write partial results so long sweeps survive interruption."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def get_probs_and_preds(
    model: SpeakerModel,
    wavs: list[np.ndarray],
    labels: list[int],
    device: torch.device,
    batch_size: int = 32,
) -> tuple[list[float], list[int]]:
    """
    FIX 3: Single batched pass returns BOTH probs AND preds.
    Old code computed preds one-by-one in a loop (1663 forward passes).
    New code does it in batches of 32 — ~50x faster.
    """
    probs, preds = [], []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(wavs), batch_size):
            batch_wavs = wavs[start: start + batch_size]
            batch_lbls = labels[start: start + batch_size]
            wav_t = torch.tensor(
                np.stack(batch_wavs), dtype=torch.float32
            ).to(device)
            logits  = model(wav_t)
            softmax = torch.softmax(logits, dim=1)
            for j, lbl in enumerate(batch_lbls):
                probs.append(float(softmax[j, lbl]))
            preds.extend(logits.argmax(dim=1).cpu().tolist())
    return probs, preds


def main() -> int:
    p = argparse.ArgumentParser(
        description=(
            "Run attack sweep (BA / VSR / MCD). "
            "By default, ONE invocation executes the entire grid in sequence in this "
            "process—load the model once, then run every attack configuration one after "
            "another (not separate scripts). Use --attack / --exclude-attacks / "
            "--one-per-attack only if you want a subset."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_SWEEP_EPILOG,
    )
    p.add_argument("--manifest",   type=Path,
                   default=_ROOT / "data/processed/manifest_with_split.csv")
    p.add_argument("--subset",     default="train-clean-100")
    p.add_argument("--split",      default="test")
    p.add_argument("--checkpoint", type=Path,
                   default=_ROOT / "results/speaker_baseline/speaker_cnn.pt")
    p.add_argument("--label-map",  type=Path,
                   default=_ROOT / "results/speaker_baseline/speaker_label_map.json")
    p.add_argument("--model-name", default="speaker_baseline",
                   choices=["speaker_baseline", "resnet18", "vggm", "ecapa"],
                   help="Model run label and default checkpoint folder.")
    p.add_argument("--all-models", action="store_true",
                   help="Run sweep for all known model labels and combine results.")
    p.add_argument("--batch-size",      type=int,   default=16)
    p.add_argument("--segment-seconds", type=float, default=2.0)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--device",          default="auto")
    p.add_argument("--num-workers",     type=int,   default=0)
    p.add_argument("--sample-rate",     type=int,   default=16_000)
    p.add_argument("--max-batches",     type=int,   default=None,
                   help="Limit batches per attack for smoke test (e.g. 5)")
    p.add_argument("--attack",          default=None,
                   help="Run only this attack. Default: full grid.")
    p.add_argument(
        "--exclude-attacks",
        default="",
        help="Comma-separated attack names to skip (e.g. asr_tts for a fast grid).",
    )
    p.add_argument("--out-csv",    type=Path,
                   default=_ROOT / "results/tables/attack_sweep.csv")
    p.add_argument("--mode",       default="default",
                   choices=["default", "focused"],
                   help="default: full fixed grid; focused: fine-grained sweep.")
    p.add_argument(
        "--resume",
        action="store_true",
        help="If out-csv exists, load it and skip (model_name, attack, strength) rows already present.",
    )
    p.add_argument(
        "--one-per-attack",
        action="store_true",
        help="Keep only the first grid row per attack name (one strength/config each).",
    )
    p.add_argument(
        "--with-optional-dep-attacks",
        action="store_true",
        help=(
            "Include deepfilter, deepafx_style, audiosr in the default grid "
            "(needs deepfilternet/torchaudio stack, DeepAFX checkpoints, or audiosr)."
        ),
    )
    args = p.parse_args()

    # validate manifest path once
    if not args.manifest.is_file():
        print(f"[ERROR] Manifest not found: {args.manifest}", file=sys.stderr)
        return 1

    device = pick_device(None if args.device == "auto" else args.device)
    print(f"Device : {device}")

    rows_all = load_manifest_filtered(
        args.manifest, subset=args.subset, split=args.split
    )
    results: list = []
    done_keys: set[tuple[str, str, str]] = set()
    if args.resume and args.out_csv.is_file():
        prev = pd.read_csv(args.out_csv)
        results = prev.to_dict("records")
        for row in results:
            done_keys.add(
                (str(row["model_name"]), str(row["attack"]), str(row["strength"]))
            )
        print(f"[INFO] Resume: loaded {len(results)} existing rows from {args.out_csv}")
    # filter grid
    if args.mode == "focused":
        grid = list(FOCUSED_SEARCH_GRID)
    else:
        grid = _attack_grid_for_run(args.with_optional_dep_attacks)
        if not args.with_optional_dep_attacks:
            print(
                "[INFO] Default sweep skips optional-dep attacks "
                f"{sorted(OPTIONAL_DEP_ATTACK_NAMES)!r} "
                "(use --with-optional-dep-attacks to run them)."
            )
    if args.attack:
        grid = [(a, pr, s) for a, pr, s in grid if a == args.attack]
        if not grid:
            print(f"[ERROR] Unknown attack {args.attack!r}. "
                  f"Choose from: {list_attack_names()}", file=sys.stderr)
            return 1

    if args.exclude_attacks.strip():
        skip = {n.strip() for n in args.exclude_attacks.split(",") if n.strip()}
        before = len(grid)
        grid = [(a, pr, s) for a, pr, s in grid if a not in skip]
        print(f"[INFO] Excluded attacks {sorted(skip)!r}: {before} → {len(grid)} grid rows")

    if args.one_per_attack:
        seen_names: set[str] = set()
        slim: list[tuple[str, dict, str]] = []
        for a, pr, s in grid:
            if a not in seen_names:
                seen_names.add(a)
                slim.append((a, pr, s))
        print(f"[INFO] One-per-attack: {len(grid)} → {len(slim)} grid rows")
        grid = slim

    print(
        f"\n[INFO] This run will execute {len(grid)} attack configuration(s) in one "
        "process, in order (model loaded once per checkpoint).\n"
    )

    model_order = ["resnet18", "vggm", "ecapa"]
    model_runs = model_order if args.all_models else [args.model_name]

    for model_name in model_runs:
        default_ckpt = _ROOT / f"results/{model_name}/speaker_cnn.pt"
        default_label_map = _ROOT / f"results/{model_name}/speaker_label_map.json"

        checkpoint = args.checkpoint
        label_map = args.label_map
        if args.all_models or args.model_name != "speaker_baseline":
            checkpoint = default_ckpt
            label_map = default_label_map

        if not checkpoint.is_file() or not label_map.is_file():
            print(
                f"[WARN] Skipping model={model_name}: missing checkpoint/label map "
                f"({checkpoint}, {label_map})"
            )
            continue

        with open(label_map, encoding="utf-8") as f:
            speaker_to_idx: dict[str, int] = json.load(f)

        rows = [r for r in rows_all if r["speaker_id"] in speaker_to_idx]
        print(f"\nModel  : {model_name}")
        print(f"Weights: {checkpoint}")
        print(f"Eval utterances: {len(rows)}")

        ds = LibriSpeechSpeakerDataset(
            rows, speaker_to_idx,
            segment_seconds=args.segment_seconds,
            augment=False, seed=args.seed,
        )
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(device.type == "cuda"),
        )
        model = load_model(checkpoint, device)

        # collect clean baseline probs ONCE per model
        print("\nCollecting clean baseline probs (once) ...")
        clean_wavs_base, _, labels_base = collect_samples(
            loader, "none", {}, device, args.sample_rate, args.max_batches
        )
        clean_probs_base, _ = get_probs_and_preds(
            model, clean_wavs_base, labels_base, device
        )

        for attack_name, params, strength in grid:
            run_key = (model_name, attack_name, strength)
            if run_key in done_keys:
                print(f"\n[INFO] Skip (resume): {attack_name}  {strength}")
                continue

            print(f"\n{'─'*60}")
            print(f"Attack : {attack_name}  strength={strength}  params={params}")
            t0 = time.time()

            try:
                clean_wavs, attacked_wavs, labels = collect_samples(
                    loader, attack_name, params,
                    device, args.sample_rate, args.max_batches,
                )

                attacked_probs, preds_attacked = get_probs_and_preds(
                    model, attacked_wavs, labels, device
                )

                ba      = compute_BA(preds=preds_attacked, true_labels=labels)
                # Proxy VSR: paired test as in paper; probs = speaker softmax on true class.
                vsr     = compute_VSR(clean_probs=clean_probs_base,
                                      watermarked_probs=attacked_probs)
                mcd     = float(np.mean([
                    compute_MCD(c, a, sr=args.sample_rate)
                    for c, a in zip(clean_wavs[:50], attacked_wavs[:50])
                ]))
                harmful = compute_harmful_degree(
                    preds=preds_attacked, true_labels=labels
                )

                elapsed = time.time() - t0
                print(f"BA={ba:.1%}  VSR={vsr:.1%}  MCD={mcd:.2f}dB  "
                      f"Harmful={harmful:.3f}  ({elapsed:.1f}s)")

                results.append({
                    "model_name":     model_name,
                    "attack":         attack_name,
                    "strength":       strength,
                    "params":         str(params),
                    "BA":             round(ba,      4),
                    "VSR":            round(vsr,     4),
                    "MCD_dB":         round(mcd,     2),
                    "harmful_degree": round(harmful, 4),
                    "n_samples":      len(labels),
                    "elapsed_s":      round(elapsed, 1),
                })
                done_keys.add(run_key)
                _flush_results_csv(args.out_csv, results)

            except Exception as exc:
                print(f"[WARN] {attack_name}/{strength} failed: {exc}")
                results.append({
                    "model_name": model_name,
                    "attack": attack_name, "strength": strength,
                    "params": str(params), "BA": None, "VSR": None,
                    "MCD_dB": None, "harmful_degree": None,
                    "n_samples": 0,
                    "elapsed_s": None, "error": str(exc),
                })
                done_keys.add(run_key)
                _flush_results_csv(args.out_csv, results)

    # save (final; same as incremental flush)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv(args.out_csv, index=False)

    print(f"\n{'='*60}")
    print("SWEEP COMPLETE — SUMMARY")
    print(f"{'='*60}")
    print(df[["model_name","attack","strength","BA","VSR","MCD_dB"]]
          .to_string(index=False))
    print(f"\nSaved → {args.out_csv.resolve()}")
    ok = df["BA"].notna() & df["VSR"].notna()
    n_ok = int(ok.sum())
    n_failed = int((~ok).sum())
    print(f"\nCompleted runs: {n_ok} / {len(df)}")
    if n_failed:
        print(f"Failed runs (dependency/runtime issues): {n_failed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
