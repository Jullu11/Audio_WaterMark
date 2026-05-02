#!/usr/bin/env python3
"""Write ``results/submission/table_robustness.md`` from deduped sweep CSV (for appendices / Word)."""

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
    short_label,
    sort_key,
)


def main() -> int:
    p = argparse.ArgumentParser(description="Export Markdown table from robustness sweep CSV.")
    p.add_argument(
        "--csv",
        type=Path,
        default=Path("results/speaker_baseline/robustness_sweep_deduped.csv"),
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("results/submission/table_robustness.md"),
    )
    args = p.parse_args()
    src = (args.csv if args.csv.is_absolute() else _PROJECT_ROOT / args.csv).resolve()
    if src.is_file():
        rows = load_sweep_csv(src)
    else:
        full = _PROJECT_ROOT / "results" / "speaker_baseline" / "robustness_sweep.csv"
        if full.is_file():
            rows = dedupe_first(load_sweep_csv(full))
        else:
            print(f"CSV not found: {src} (and no {full.name})", file=sys.stderr)
            return 1
    rows = sorted(rows, key=sort_key)

    lines = [
        "# Robustness table (generated)",
        "",
        "| Condition | Closed-set accuracy |",
        "|-----------|--------------------:|",
    ]
    for r in rows:
        cond = short_label(r["attack"], r["params"])
        acc = float(r["accuracy"])
        lines.append(f"| {cond} | {acc:.4f} |")
    lines.append("")

    out = (args.out if args.out.is_absolute() else _PROJECT_ROOT / args.out).resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
