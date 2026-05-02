#!/usr/bin/env python3
"""Download and extract LibriSpeech subsets from OpenSLR (resource 12)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.dataset.librispeech import (  # noqa: E402
    LIBRISPEECH_SUBSETS,
    LibrispeechDownloadError,
    default_download_root,
    download_subset,
    extracted_subset_root,
    subset_url,
)


def main() -> int:
    p = argparse.ArgumentParser(
        description="Download LibriSpeech from https://www.openslr.org/12/ into data/raw/."
    )
    p.add_argument(
        "--subset",
        default="train-clean-100",
        choices=sorted(LIBRISPEECH_SUBSETS.keys()),
        help="Corpus subset (default: train-clean-100; use dev-clean for a quick smoke test).",
    )
    p.add_argument(
        "--download-root",
        type=Path,
        default=None,
        help="Directory where the .tar.gz is stored and LibriSpeech/ is extracted (default: <project>/data/raw).",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="Re-download and re-extract even if the subset folder already exists.",
    )
    p.add_argument(
        "--no-extract",
        action="store_true",
        help="Only download the archive; do not extract.",
    )
    p.add_argument(
        "--remove-archive",
        action="store_true",
        help="Delete the .tar.gz after successful extraction.",
    )
    p.add_argument(
        "--verify-archive",
        action="store_true",
        help="Before extracting, fully verify gzip integrity (slow on multi‑GB files; use after bad/interrupted downloads).",
    )
    args = p.parse_args()

    dl_root = args.download_root or default_download_root(_PROJECT_ROOT)
    url = subset_url(args.subset)
    print(f"Subset:      {args.subset}")
    print(f"URL:         {url}")
    print(f"Download to: {dl_root.resolve()}")

    try:
        out = download_subset(
            args.subset,
            dl_root,
            force=args.force,
            extract=not args.no_extract,
            remove_archive=args.remove_archive,
            verify_archive=args.verify_archive,
        )
    except (OSError, LibrispeechDownloadError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.no_extract:
        print(f"Done. Archive:\n  {out.resolve()}")
        return 0

    expected = extracted_subset_root(dl_root, args.subset)
    if out.resolve() != expected.resolve():
        print(f"Internal error: unexpected output path {out!s}", file=sys.stderr)
        return 1
    print(f"Done. Extracted subset root:\n  {out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
