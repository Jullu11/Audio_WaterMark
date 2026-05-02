#!/usr/bin/env python3
"""Build a manifest CSV from a LibriSpeech tree (LibriSpeech/<subset>/.../*.flac)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset.librispeech import write_manifest_csv  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(
        description="Scan LibriSpeech FLAC files and write manifest CSV for downstream steps."
    )
    p.add_argument(
        "--librispeech-root",
        type=Path,
        required=True,
        help="Path to the LibriSpeech folder containing subsets (e.g. .../data/raw/LibriSpeech).",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output CSV path (e.g. data/processed/manifest_librispeech.csv).",
    )
    p.add_argument(
        "--subsets",
        nargs="*",
        default=None,
        help="Optional subset folder names to include (e.g. dev-clean train-clean-100). Default: all found.",
    )
    p.add_argument(
        "--path-column",
        choices=("absolute", "relative"),
        default="absolute",
        help="Whether CSV path column stores absolute paths or paths relative to LibriSpeech parent.",
    )
    args = p.parse_args()

    root = args.librispeech_root.resolve()
    if not root.is_dir():
        print(f"LibriSpeech root not found: {root}", file=sys.stderr)
        return 1

    n = write_manifest_csv(
        root,
        args.out,
        path_column=args.path_column,
        subsets=list(args.subsets) if args.subsets else None,
    )
    print(f"Wrote {n} rows to {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
