#!/usr/bin/env python3
"""
Regenerate paper artifacts in order: optional eval/sweep, then plot, bundle, verify.

Default (fast): ``07`` → ``08`` → ``09``  
With ``--with-sweep``: ``06`` first (re-runs full eval grid; slow).  
With ``--with-eval``: ``04`` first (refreshes ``eval_test_metrics.json``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]


def run_script(name: str, extra: list[str] | None = None) -> None:
    cmd = [sys.executable, str(_PROJECT_ROOT / "scripts" / name)]
    if extra:
        cmd.extend(extra)
    print(f"\n→ {' '.join(cmd)}\n", flush=True)
    env = {**os.environ, "PYTHONPATH": str(_PROJECT_ROOT)}
    env.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".matplotlib_cache"))
    (_PROJECT_ROOT / ".matplotlib_cache").mkdir(parents=True, exist_ok=True)
    r = subprocess.run(cmd, cwd=str(_PROJECT_ROOT), env=env)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> int:
    p = argparse.ArgumentParser(description="Reproduce figures + submission bundle + verification.")
    p.add_argument("--with-eval", action="store_true", help="Run 04_eval_speaker_id.py first.")
    p.add_argument("--with-sweep", action="store_true", help="Run 06_robustness_sweep.py first (slow).")
    p.add_argument(
        "--require-checkpoint",
        action="store_true",
        help="Pass through to 09: fail if speaker_cnn.pt is missing.",
    )
    args = p.parse_args()

    if args.with_eval:
        run_script("04_eval_speaker_id.py")
    if args.with_sweep:
        run_script("06_robustness_sweep.py")
    run_script("07_plot_robustness_sweep.py")
    run_script("08_bundle_submission.py")
    verify_extra = ["--require-checkpoint"] if args.require_checkpoint else []
    run_script("09_verify_submission.py", verify_extra or None)
    print("\nDone: figures, submission bundle, and verification.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
