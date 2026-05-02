#!/usr/bin/env python3
"""
Run a preset grid of waveform attacks; append one row per run to a CSV.

Invokes ``05_eval_robustness.py`` as a subprocess so behavior matches the CLI exactly.
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

# (attack_name, extra_cli_tokens)
PRESETS: list[tuple[str, list[str]]] = [
    ("none", []),
    ("gaussian_noise", ["--attack-kw", "snr_db=30"]),
    ("gaussian_noise", ["--attack-kw", "snr_db=20"]),
    ("gaussian_noise", ["--attack-kw", "snr_db=10"]),
    ("time_stretch", ["--attack-kw", "rate=1.05"]),
    ("time_stretch", ["--attack-kw", "rate=1.1"]),
    ("lowpass", ["--attack-kw", "cutoff_hz=4000"]),
    ("lowpass", ["--attack-kw", "cutoff_hz=2000"]),
    ("highpass", ["--attack-kw", "cutoff_hz=200"]),
    ("resample_chain", ["--attack-kw", "mid_sr=8000"]),
    ("quantize", ["--attack-kw", "levels=256"]),
    ("quantize", ["--attack-kw", "levels=32"]),
]


def main() -> int:
    out_csv = _PROJECT_ROOT / "results" / "speaker_baseline" / "robustness_sweep.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["attack", "params", "accuracy", "metrics_json"]
    new_file = not out_csv.is_file()
    f = open(out_csv, "a", newline="", encoding="utf-8")
    w = csv.DictWriter(f, fieldnames=fieldnames)
    if new_file:
        w.writeheader()

    py = sys.executable
    script = _PROJECT_ROOT / "scripts" / "05_eval_robustness.py"

    for idx, (attack, extra) in enumerate(PRESETS):
        mj = _PROJECT_ROOT / "results" / "speaker_baseline" / f"robustness_tmp_{idx:02d}_{attack}.json"
        cmd = [
            py,
            str(script),
            "--attack",
            attack,
            *extra,
            "--metrics-json",
            str(mj),
        ]
        env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
        r = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), env=env, capture_output=True, text=True)
        if r.returncode != 0:
            print(r.stdout)
            print(r.stderr, file=sys.stderr)
            print(f"FAILED: {' '.join(cmd)}", file=sys.stderr)
            f.close()
            return r.returncode
        with open(mj, encoding="utf-8") as jf:
            data = json.load(jf)
        params = json.dumps(data.get("attack_params") or {})
        w.writerow(
            {
                "attack": data.get("attack", attack),
                "params": params,
                "accuracy": data.get("accuracy"),
                "metrics_json": str(mj),
            }
        )
        f.flush()
        print(f"ok {attack} {params} -> {data.get('accuracy')}")

    f.close()
    print(f"Appended sweep rows to {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
