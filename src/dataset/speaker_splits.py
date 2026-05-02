"""Speaker-level train/val/test assignment from manifest rows (no speaker leakage)."""

from __future__ import annotations

import csv
import random
from collections.abc import Iterable
from pathlib import Path


def partition_sizes(n_items: int, train_r: float, val_r: float, test_r: float) -> tuple[int, int, int]:
    """Split ``n_items`` into three counts proportional to ratios (sums to ``n_items``)."""
    if abs(train_r + val_r + test_r - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0, got {train_r + val_r + test_r}")
    if n_items <= 0:
        return 0, 0, 0
    if n_items == 1:
        return 1, 0, 0
    if n_items == 2:
        return 1, 1, 0

    # Floors + steal from train so small buckets still get val/test utterances.
    n_train = max(1, int(n_items * train_r))
    n_val = max(1, int(n_items * val_r))
    n_test = n_items - n_train - n_val
    while n_test < 1 and n_train > 1:
        n_train -= 1
        n_test += 1
    while n_test < 1 and n_val > 1:
        n_val -= 1
        n_test += 1
    if n_test < 1:
        n_test = 1
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_train -= 1
    assert n_train + n_val + n_test == n_items
    return n_train, n_val, n_test


def speaker_to_split(
    speaker_ids: Iterable[str],
    *,
    seed: int,
    train_r: float,
    val_r: float,
    test_r: float,
) -> dict[str, str]:
    """Map each speaker id to ``train``, ``val``, or ``test``."""
    ids = sorted(set(speaker_ids))
    rng = random.Random(seed)
    rng.shuffle(ids)
    n = len(ids)
    n_train, n_val, n_test = partition_sizes(n, train_r, val_r, test_r)
    out: dict[str, str] = {}
    i = 0
    for spk in ids[i : i + n_train]:
        out[spk] = "train"
    i += n_train
    for spk in ids[i : i + n_val]:
        out[spk] = "val"
    i += n_val
    for spk in ids[i : i + n_test]:
        out[spk] = "test"
    i += n_test
    assert i == n
    return out


def read_manifest_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_manifest_with_split(
    rows: list[dict[str, str]],
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No rows to write")
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def write_filtered_csv(rows: list[dict[str, str]], split_name: str, out_dir: Path) -> Path:
    sub = [r for r in rows if r.get("split") == split_name]
    out = out_dir / f"manifest_{split_name}.csv"
    write_manifest_with_split(sub, out)
    return out


def assign_utterance_splits(
    rows: list[dict[str, str]],
    subset_name: str,
    *,
    seed: int,
    train_r: float,
    val_r: float,
    test_r: float,
) -> None:
    """
    For rows with ``subset == subset_name``, assign ``split`` per *utterance* so each
    speaker appears in train/val/test (closed-set classification). Mutates ``rows``.
    """
    from collections import defaultdict

    by_spk: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        if r.get("subset") == subset_name:
            by_spk[r["speaker_id"]].append(i)

    rng = random.Random(seed)
    for idxs in by_spk.values():
        rng.shuffle(idxs)
        n = len(idxs)
        n_train, n_val, _ = partition_sizes(n, train_r, val_r, test_r)
        for j in idxs[:n_train]:
            rows[j]["split"] = "train"
        for j in idxs[n_train : n_train + n_val]:
            rows[j]["split"] = "val"
        for j in idxs[n_train + n_val :]:
            rows[j]["split"] = "test"
