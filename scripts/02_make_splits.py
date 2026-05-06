#!/usr/bin/env python3
"""
Assign speaker-level splits on LibriSpeech manifest rows.

- Speakers from ``--split-subset`` (default ``train-clean-100``) are partitioned into
  train/val/test so no speaker appears in more than one split.
- Rows from ``--eval-as-test`` subsets (default ``dev-clean``) get ``split=test`` for
  held-out evaluation (common: train on train-clean-100, report BA on dev-clean).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset.speaker_splits import (  # noqa: E402
    assign_utterance_splits,
    read_manifest_rows,
    speaker_to_split,
    write_filtered_csv,
    write_manifest_with_split,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Add train/val/test split column: disjoint speakers (speaker) or utterances (utterance)."
    )
    p.add_argument(
        "--strategy",
        choices=("speaker", "utterance"),
        default="utterance",
        help="speaker = disjoint speakers across splits; utterance = same speakers in all splits (closed-set softmax).",
    )
    p.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/processed/manifest_librispeech.csv"),
        help="Input manifest from 01_prepare_manifest.py",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=Path("data/processed/manifest_with_split.csv"),
        help="Output manifest CSV including a ``split`` column.",
    )
    p.add_argument(
        "--split-subset",
        default="train-clean-100",
        help="Corpus subset whose *speakers* are split into train/val/test.",
    )
    p.add_argument(
        "--eval-as-test",
        nargs="*",
        default=["dev-clean"],
        help="Subset names whose rows are all labeled split=test (held-out eval).",
    )
    p.add_argument("--train-ratio", type=float, default=0.85)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--write-per-split",
        action="store_true",
        help="Also write data/processed/splits/manifest_{train,val,test}.csv",
    )
    args = p.parse_args()

    manifest_path = args.manifest.resolve()
    if not manifest_path.is_file():
        print(f"Manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    rows = read_manifest_rows(manifest_path)
    if not rows:
        print(
            "Manifest has no data rows (empty or header-only). "
            "Fix `01_prepare_manifest.py` first: it must find `*.flac` under "
            "`--librispeech-root` (usually after `scripts/download_librispeech.py`).\n"
            "Tip: chain steps so this script does not run if manifest generation fails:\n"
            "  python scripts/01_prepare_manifest.py ... && python scripts/02_make_splits.py ...",
            file=sys.stderr,
        )
        return 1

    eval_test = set(args.eval_as_test)

    if args.strategy == "utterance":
        for r in rows:
            sub = r.get("subset", "")
            if sub in eval_test:
                r["split"] = "test"
            elif sub != args.split_subset:
                r["split"] = "unused"
        assign_utterance_splits(
            rows,
            args.split_subset,
            seed=args.seed,
            train_r=args.train_ratio,
            val_r=args.val_ratio,
            test_r=args.test_ratio,
        )
    else:
        speakers_split_subset: list[str] = []
        for r in rows:
            if r.get("subset") == args.split_subset:
                speakers_split_subset.append(r["speaker_id"])

        if not speakers_split_subset:
            print(f"No rows with subset={args.split_subset!r}. Check manifest.", file=sys.stderr)
            return 1

        spk_map = speaker_to_split(
            speakers_split_subset,
            seed=args.seed,
            train_r=args.train_ratio,
            val_r=args.val_ratio,
            test_r=args.test_ratio,
        )

        for r in rows:
            sub = r.get("subset", "")
            if sub in eval_test:
                r["split"] = "test"
            elif sub == args.split_subset:
                r["split"] = spk_map[r["speaker_id"]]
            else:
                r["split"] = "unused"

    out_path = args.out
    if out_path.is_dir():
        print("--out must be a file path, not a directory", file=sys.stderr)
        return 1
    write_manifest_with_split(rows, out_path.resolve())

    n_by = {"train": 0, "val": 0, "test": 0, "unused": 0}
    for r in rows:
        n_by[r["split"]] = n_by.get(r["split"], 0) + 1

    print(f"Wrote {len(rows)} rows to {out_path.resolve()}")
    print("Rows per split:", {k: v for k, v in n_by.items() if v})

    if args.write_per_split:
        split_dir = out_path.parent / "splits"
        split_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train", "val", "test"):
            pth = write_filtered_csv(rows, name, split_dir)
            print(f"  {name}: {pth}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
