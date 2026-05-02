"""Load robustness sweep CSV, dedupe rows, build axis labels, plot accuracy bars."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def load_sweep_csv(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(
                {
                    "attack": r["attack"].strip(),
                    "params": r["params"].strip(),
                    "accuracy": float(r["accuracy"]),
                    "metrics_json": r.get("metrics_json", ""),
                }
            )
    return rows


def dedupe_first(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep first row per (attack, params); strips accidental duplicate sweep runs."""
    seen: set[tuple[str, str]] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        key = (r["attack"], r["params"])
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def short_label(attack: str, params_json: str) -> str:
    if attack in ("none", "clean"):
        return "clean (no attack)"
    try:
        p = json.loads(params_json) if params_json else {}
    except json.JSONDecodeError:
        return f"{attack}"
    if attack == "gaussian_noise":
        return f"noise {p.get('snr_db', '?')} dB SNR"
    if attack == "time_stretch":
        return f"time stretch ×{p.get('rate', '?')}"
    if attack == "lowpass":
        return f"low-pass {p.get('cutoff_hz', '?')} Hz"
    if attack == "highpass":
        return f"high-pass {p.get('cutoff_hz', '?')} Hz"
    if attack == "resample_chain":
        return f"resample {p.get('mid_sr', '?')} Hz → 16 kHz"
    if attack == "quantize":
        return f"quantize {p.get('levels', '?')} levels"
    return attack


def sort_key(row: dict[str, Any]) -> tuple[int, str, str]:
    """Clean row first, then attack name, then params for stable ordering."""
    a = row["attack"]
    pri = 0 if a in ("none", "clean") else 1
    return (pri, a, row["params"])


def write_sweep_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["attack", "params", "accuracy", "metrics_json"])
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "attack": r["attack"],
                    "params": r["params"],
                    "accuracy": r["accuracy"],
                    "metrics_json": r.get("metrics_json", ""),
                }
            )


def plot_accuracy_bars(
    rows: list[dict[str, Any]],
    out_png: Path,
    *,
    title: str = "Closed-set speaker-ID accuracy under waveform attacks",
    dpi: int = 150,
) -> None:
    import matplotlib.pyplot as plt

    rows = sorted(rows, key=sort_key)
    labels = [short_label(r["attack"], r["params"]) for r in rows]
    acc = [r["accuracy"] for r in rows]

    fig_h = max(4.0, 0.35 * len(labels) + 1.5)
    fig, ax = plt.subplots(figsize=(9, fig_h), layout="constrained")
    y_pos = range(len(labels))
    colors = ["#2ca02c" if r["attack"] in ("none", "clean") else "#1f77b4" for r in rows]
    ax.barh(list(y_pos), acc, color=colors, height=0.65)
    ax.set_yticks(list(y_pos), labels=labels, fontsize=9)
    ax.set_xlabel("Accuracy (closed-set test)")
    ax.set_title(title)
    ax.set_xlim(0.0, 1.05)
    if rows and rows[0]["attack"] in ("none", "clean"):
        ax.axvline(acc[0], color="#888", linestyle=":", linewidth=1)
    for i, v in enumerate(acc):
        ax.text(min(v + 0.02, 0.98), i, f"{v:.3f}", va="center", fontsize=8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
