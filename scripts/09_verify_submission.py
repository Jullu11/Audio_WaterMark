#!/usr/bin/env python3
"""
Verify that exported metrics are internally consistent and expected artifacts exist.

Checks:
  - ``eval_test_metrics.json`` exists and parses.
  - ``robustness_sweep_deduped.csv`` (or deduped ``robustness_sweep.csv``) has a ``none``
    row whose accuracy matches baseline within ``--atol``.
  - Optional: ``results/figures/robustness_bars.png``, ``results/submission/README.md``.

Exit code ``0`` if all checks pass, ``1`` otherwise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.report.robustness_figure import dedupe_first, load_sweep_csv  # noqa: E402


def _find_clean_accuracy(rows: list[dict]) -> float | None:
    for r in rows:
        if r["attack"].strip().lower() in ("none", "clean") and (not r.get("params") or r["params"].strip() in ("{}", "")):
            return float(r["accuracy"])
    for r in rows:
        if r["attack"].strip().lower() in ("none", "clean"):
            return float(r["accuracy"])
    return None


def main() -> int:
    p = argparse.ArgumentParser(description="Verify metrics consistency and bundle files.")
    p.add_argument("--root", type=Path, default=_PROJECT_ROOT)
    p.add_argument("--atol", type=float, default=1e-5, help="Absolute tolerance for accuracy match.")
    p.add_argument("--skip-submission-dir", action="store_true")
    p.add_argument("--skip-figure", action="store_true")
    p.add_argument("--require-checkpoint", action="store_true", help="Fail if speaker_cnn.pt missing.")
    args = p.parse_args()
    root = args.root.resolve()
    errors: list[str] = []
    ok: list[str] = []

    eval_path = root / "results" / "speaker_baseline" / "eval_test_metrics.json"
    if not eval_path.is_file():
        errors.append(f"Missing {eval_path}")
    else:
        with open(eval_path, encoding="utf-8") as f:
            ev = json.load(f)
        baseline = float(ev["accuracy"])
        ok.append(f"Baseline accuracy from eval_test_metrics.json: {baseline:.6f}")

        deduced = root / "results" / "speaker_baseline" / "robustness_sweep_deduped.csv"
        sweep_full = root / "results" / "speaker_baseline" / "robustness_sweep.csv"
        rows: list[dict] | None = None
        if deduced.is_file():
            rows = load_sweep_csv(deduced)
            ok.append(f"Loaded deduped sweep: {deduced.name} ({len(rows)} rows)")
        elif sweep_full.is_file():
            rows = dedupe_first(load_sweep_csv(sweep_full))
            ok.append(f"Deduped from {sweep_full.name} ({len(rows)} rows)")
        else:
            errors.append("Missing both robustness_sweep_deduped.csv and robustness_sweep.csv")

        if rows is not None:
            clean = _find_clean_accuracy(rows)
            if clean is None:
                errors.append("No 'none' / clean row found in sweep data.")
            else:
                diff = abs(clean - baseline)
                if diff > args.atol:
                    errors.append(
                        f"Mismatch: eval baseline {baseline:.9f} vs sweep clean {clean:.9f} (|Δ|={diff:.2e} > atol={args.atol})"
                    )
                else:
                    ok.append(f"Sweep 'clean' matches baseline (|Δ|={diff:.2e}).")

    fig = root / "results" / "figures" / "robustness_bars.png"
    if not args.skip_figure:
        if not fig.is_file():
            errors.append(f"Missing figure {fig} (run scripts/07_plot_robustness_sweep.py)")
        else:
            ok.append(f"Figure present: {fig.name}")

    sub = root / "results" / "submission"
    if not args.skip_submission_dir:
        readme = sub / "README.md"
        dedup = sub / "robustness_sweep_deduped.csv"
        if not sub.is_dir():
            errors.append(f"Missing directory {sub} (run scripts/08_bundle_submission.py)")
        else:
            if not readme.is_file():
                errors.append(f"Missing {readme}")
            if not dedup.is_file():
                errors.append(f"Missing {dedup}")
            if readme.is_file() and dedup.is_file():
                ok.append("results/submission bundle looks populated.")

    ckpt = root / "results" / "speaker_baseline" / "speaker_cnn.pt"
    if args.require_checkpoint:
        if not ckpt.is_file():
            errors.append(f"Missing checkpoint {ckpt}")
        else:
            ok.append("Checkpoint present.")

    for line in ok:
        print(f"OK  {line}")
    for line in errors:
        print(f"ERR {line}", file=sys.stderr)

    if errors:
        print(f"\nFailed {len(errors)} check(s).", file=sys.stderr)
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
