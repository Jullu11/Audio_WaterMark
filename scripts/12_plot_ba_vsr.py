#!/usr/bin/env python3
"""Plot BA vs VSR trade-off from attack sweep CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _strength_order_value(v: str) -> int:
    order = {
        "baseline": 0,
        "light": 1,
        "medium": 2,
        "heavy": 3,
    }
    return order.get(str(v).strip().lower(), 99)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot BA vs VSR trade-off.")
    parser.add_argument(
        "--in-csv",
        type=Path,
        default=Path("results/tables/attack_sweep.csv"),
        help="Input attack sweep CSV.",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("results/figures/ba_vs_vsr.png"),
        help="Output figure path.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model_name filter if CSV contains multiple models.",
    )
    args = parser.parse_args()

    if not args.in_csv.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.in_csv}")

    df = pd.read_csv(args.in_csv)
    df = df.dropna(subset=["BA", "VSR", "attack", "strength"])

    if args.model_name and "model_name" in df.columns:
        df = df[df["model_name"] == args.model_name]

    if df.empty:
        raise RuntimeError("No rows available to plot after filtering.")

    # Stable strength ordering for each attack trajectory
    df["strength_rank"] = df["strength"].map(_strength_order_value)

    fig, ax = plt.subplots(figsize=(10, 6))
    attacks = sorted(df["attack"].unique())
    colors = plt.cm.tab20.colors

    for i, attack in enumerate(attacks):
        sub = df[df["attack"] == attack].sort_values(
            by=["strength_rank", "strength"]
        )
        color = colors[i % len(colors)]
        ax.plot(
            sub["BA"],
            sub["VSR"],
            marker="o",
            label=attack,
            color=color,
            linewidth=1.5,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                str(row["strength"]),
                (row["BA"], row["VSR"]),
                fontsize=8,
                xytext=(3, 3),
                textcoords="offset points",
            )

    # Reference region for narrative (high BA, low VSR): BA > 0.8 and VSR < 0.7
    ax.axhline(y=0.70, color="red", linestyle="--", linewidth=1.0)
    ax.axvline(x=0.80, color="blue", linestyle="--", linewidth=1.0)
    ax.fill_between([0.80, 1.0], 0.0, 0.70, alpha=0.10, color="green")

    title = "BA vs VSR Trade-off Per Attack"
    if args.model_name:
        title = f"{title} ({args.model_name})"
    ax.set_title(title)
    ax.set_xlabel("Benign Accuracy (BA)")
    ax.set_ylabel("Verification Success Rate (VSR)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=150)
    print(f"Saved -> {args.out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
