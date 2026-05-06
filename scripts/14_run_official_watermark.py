#!/usr/bin/env python3
"""Thin wrapper around official AUDIO WATERMARK entrypoints.

This helps run the vendored official code with explicit checks and cleaner errors.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str], cwd: Path) -> int:
    print("Running:", " ".join(cmd))
    proc = subprocess.run(cmd, cwd=str(cwd), env={**os.environ}, text=True)
    return int(proc.returncode)


def main() -> int:
    p = argparse.ArgumentParser(description="Run official AUDIO WATERMARK scripts.")
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root.",
    )
    p.add_argument(
        "--task",
        choices=["verify_generate", "verify_eval", "train_benign"],
        required=True,
        help="Official workflow task.",
    )
    p.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="Extra args passed through to official script.",
    )
    args = p.parse_args()

    code_dir = args.root / "external" / "audiowatermark.github.io" / "code"
    verify_py = code_dir / "verify_watermark.py"
    train_all_py = code_dir / "train_all.py"

    if not code_dir.is_dir():
        print(f"[ERROR] Official code folder missing: {code_dir}", file=sys.stderr)
        return 2

    if args.task.startswith("verify") and not verify_py.is_file():
        print(f"[ERROR] Missing official script: {verify_py}", file=sys.stderr)
        return 2
    if args.task == "train_benign" and not train_all_py.is_file():
        print(f"[ERROR] Missing official script: {train_all_py}", file=sys.stderr)
        return 2

    if args.task == "verify_generate":
        cmd = [sys.executable, str(verify_py), "-g", *args.extra]
    elif args.task == "verify_eval":
        # default values from authors' docs can be overridden via --extra
        cmd = [sys.executable, str(verify_py), "-e", "-pr", "0.1", "-epoch", "20", *args.extra]
    else:
        cmd = [sys.executable, str(train_all_py), "--tasks", "train_benign", "--sr_model", "resnet18", *args.extra]

    print(f"Official code dir: {code_dir}")
    rc = _run(cmd, cwd=code_dir)
    if rc != 0:
        print(
            "\n[HINT] If this fails on macOS, this is expected for the official dependency stack.\n"
            "Use Linux+CUDA conda env from external/audiowatermark.github.io/code/dependency.yml.",
            file=sys.stderr,
        )
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
