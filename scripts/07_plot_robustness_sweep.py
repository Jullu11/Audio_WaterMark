#!/usr/bin/env python3
"""
Build a paper-style figure from ``results/speaker_baseline/robustness_sweep.csv``.

Deduplicates duplicate sweep runs (same attack + params), optionally writes
``robustness_sweep_deduped.csv``, saves ``results/figures/robustness_bars.png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.report.robustness_figure import (  # noqa: E402
    dedupe_first,
    load_sweep_csv,
    plot_accuracy_bars,
    write_sweep_csv,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Plot robustness sweep as horizontal bar chart.")
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("results/speaker_baseline/robustness_sweep.csv"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/figures/robustness_bars.png"),
    )
    p.add_argument(
        "--write-deduped",
        type=Path,
        default=Path("results/speaker_baseline/robustness_sweep_deduped.csv"),
        help="Write deduplicated CSV.",
    )
    p.add_argument(
        "--no-write-deduped",
        action="store_true",
        help="Do not write deduplicated CSV.",
    )
    p.add_argument("--title", default="Closed-set speaker-ID accuracy under waveform attacks")
    args = p.parse_args()

    src = args.csv.resolve()
    if not src.is_file():
        print(f"CSV not found: {src}", file=sys.stderr)
        return 1

    raw = load_sweep_csv(src)
    deduped = dedupe_first(raw)
    print(f"Rows: {len(raw)} -> deduped {len(deduped)}")

    if not args.no_write_deduped:
        dp = Path(args.write_deduped).resolve()
        write_sweep_csv(dp, deduped)
        print(f"Wrote {dp}")

    out = args.out.resolve()
    plot_accuracy_bars(deduped, out, title=args.title)
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
