"""LibriSpeech: official OpenSLR URLs, download/extract, and manifest generation."""

from __future__ import annotations

import csv
import sys
import tarfile
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None  # type: ignore[misc, assignment]

# Resource 12 — https://www.openslr.org/12/
LIBRISPEECH_SUBSETS: dict[str, str] = {
    "dev-clean": "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "test-clean": "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "train-clean-100": "https://www.openslr.org/resources/12/train-clean-100.tar.gz",
    "train-clean-360": "https://www.openslr.org/resources/12/train-clean-360.tar.gz",
    "train-other-500": "https://www.openslr.org/resources/12/train-other-500.tar.gz",
}


class LibrispeechDownloadError(RuntimeError):
    """Raised when download or extraction fails."""


def default_download_root(project_root: Path | None = None) -> Path:
    """Directory containing `LibriSpeech/` after extraction (typically `data/raw`)."""
    if project_root is None:
        project_root = Path(__file__).resolve().parents[2]
    return project_root / "data" / "raw"


def archive_path(download_root: Path, subset: str) -> Path:
    return download_root / f"{subset}.tar.gz"


def extracted_subset_root(download_root: Path, subset: str) -> Path:
    """Path to e.g. `.../data/raw/LibriSpeech/dev-clean`."""
    return download_root / "LibriSpeech" / subset


def subset_url(subset: str) -> str:
    if subset not in LIBRISPEECH_SUBSETS:
        allowed = ", ".join(sorted(LIBRISPEECH_SUBSETS))
        raise ValueError(f"Unknown subset {subset!r}. Choose one of: {allowed}")
    return LIBRISPEECH_SUBSETS[subset]


def _download_url_to_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "audio_watermark_robustness/1.0 (LibriSpeech downloader)"},
    )
    with urllib.request.urlopen(req) as resp:  # noqa: S310 — fixed official URL list only
        total = int(resp.headers.get("Content-Length") or 0)
        chunk = 1024 * 1024
        if tqdm is not None and total > 0:
            pbar = tqdm(total=total, unit="B", unit_scale=True, desc=dest.name)
        else:
            pbar = None
        with open(dest, "wb") as f:
            while True:
                buf = resp.read(chunk)
                if not buf:
                    break
                f.write(buf)
                if pbar is not None:
                    pbar.update(len(buf))
        if pbar is not None:
            pbar.close()

    size_on_disk = dest.stat().st_size
    if total > 0 and size_on_disk != total:
        raise LibrispeechDownloadError(
            f"Incomplete download: Content-Length was {total} bytes but file has {size_on_disk}. "
            f"Delete {dest} and run again (avoid suspending the process mid-download)."
        )


def verify_gzip_archive(path: Path) -> None:
    """
    Read the full gzip stream to detect truncation/corruption (slow for multi‑GB files).
    Run after suspicious/interrupted downloads.
    """
    import gzip as gzip_module

    try:
        with gzip_module.open(path, "rb") as gz:
            while True:
                chunk = gz.read(1024 * 1024)
                if not chunk:
                    break
    except (EOFError, OSError) as e:
        raise LibrispeechDownloadError(
            f"Archive failed gzip integrity check (truncated or corrupt): {path}. "
            f"Delete it and re-download. ({e})"
        ) from e


def _count_flac_files(root: Path) -> int:
    """Count ``*.flac`` under ``root`` (non-recursive edge cases yield 0)."""
    if not root.is_dir():
        return 0
    return sum(1 for _ in root.rglob("*.flac"))


def download_subset(
    subset: str,
    download_root: Path,
    *,
    force: bool = False,
    extract: bool = True,
    remove_archive: bool = False,
    verify_archive: bool = False,
) -> Path:
    """
    Download one LibriSpeech subset from OpenSLR and optionally extract it.

    After extraction, audio paths look like:
    ``{download_root}/LibriSpeech/{subset}/{speaker}/{chapter}/*.flac``.

    Returns:
        - If ``extract=True``: path to ``.../LibriSpeech/{subset}``.
        - If ``extract=False``: path to the downloaded ``{subset}.tar.gz`` archive.
    """
    url = subset_url(subset)
    out_root = extracted_subset_root(download_root, subset)
    # Do not skip download/extract when a stale folder exists but contains no audio.
    if extract and out_root.is_dir() and any(out_root.iterdir()) and not force:
        if _count_flac_files(out_root) > 0:
            return out_root
        print(
            f"Warning: {out_root} exists but contains no .flac files; "
            "re-extracting from the archive …",
            file=sys.stderr,
        )

    tgz = archive_path(download_root, subset)
    if not tgz.is_file() or force:
        if tgz.is_file() and force:
            tgz.unlink()
        _download_url_to_file(url, tgz)

    if not extract:
        if not tgz.is_file():
            raise LibrispeechDownloadError(f"Archive missing after download: {tgz}")
        return tgz

    if verify_archive:
        print(f"Verifying gzip integrity of {tgz.name} (may take several minutes) …")
        verify_gzip_archive(tgz)

    # Archive contains top-level ``LibriSpeech/<subset>/...``
    if not tarfile.is_tarfile(tgz):
        raise LibrispeechDownloadError(f"Not a valid tar.gz: {tgz}")
    print(f"Extracting {tgz.name} into {download_root} (this can take several minutes) …")
    extract_kw: dict = {}
    if sys.version_info >= (3, 12):
        # Mitigate path traversal / unsafe member types (PEP 706).
        extract_kw["filter"] = "data"
    with tarfile.open(tgz, "r:gz") as tf:
        tf.extractall(path=download_root, **extract_kw)
    if remove_archive:
        tgz.unlink(missing_ok=True)

    if not out_root.is_dir():
        raise LibrispeechDownloadError(f"Expected extracted directory missing: {out_root}")

    n_flac = _count_flac_files(out_root)
    if n_flac == 0:
        raise LibrispeechDownloadError(
            f"Extraction finished but found 0 .flac files under {out_root}. "
            f"Remove that folder and {tgz}, then run download_librispeech.py with --force."
        )
    return out_root


@dataclass(frozen=True)
class LibrispeechUtterance:
    """One utterance row for evaluation manifests."""

    path: Path
    speaker_id: str
    chapter_id: str
    utterance_id: str
    subset: str
    rel_path: str

    @property
    def is_watermarked(self) -> bool:
        return False


def iter_flac_utterances(librispeech_root: Path) -> Iterator[LibrispeechUtterance]:
    """
    Walk ``LibriSpeech/<subset>/<speaker>/<chapter>/*.flac``.

    ``librispeech_root`` should be ``.../LibriSpeech`` (parent of ``dev-clean``, etc.).
    """
    root = librispeech_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"LibriSpeech root not found: {root}")

    for flac in sorted(root.rglob("*.flac")):
        rel = flac.relative_to(root)
        parts = rel.parts
        # Normal layout: ``<subset>/<speaker>/<chapter>/<file>.flac`` (4 parts) under ``LibriSpeech/``.
        # If users pass ``--librispeech-root data/raw`` instead of ``.../LibriSpeech``, paths look like
        # ``LibriSpeech/<subset>/<speaker>/<chapter>/<file>.flac`` (5 parts) — accept that too.
        if len(parts) == 5 and parts[0] == "LibriSpeech":
            subset, speaker_id, chapter_id, name = parts[1], parts[2], parts[3], parts[4]
        elif len(parts) == 4:
            subset, speaker_id, chapter_id, name = parts
        else:
            continue
        stem = Path(name).stem
        yield LibrispeechUtterance(
            path=flac,
            speaker_id=speaker_id,
            chapter_id=chapter_id,
            utterance_id=stem,
            subset=subset,
            rel_path=rel.as_posix(),
        )


def write_manifest_csv(
    librispeech_root: Path,
    out_csv: Path,
    *,
    path_column: str = "absolute",
    subsets: list[str] | None = None,
) -> int:
    """
    Write a CSV with columns used downstream: path, speaker_id, chapter_id, utterance_id,
    subset, rel_path, is_watermarked.
    """
    lib_root = librispeech_root.resolve()
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    allow = set(subsets) if subsets else None
    n = 0
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "path",
                "speaker_id",
                "chapter_id",
                "utterance_id",
                "subset",
                "rel_path",
                "is_watermarked",
            ]
        )
        for u in iter_flac_utterances(lib_root):
            if allow is not None and u.subset not in allow:
                continue
            if path_column == "absolute":
                p = str(u.path.resolve())
            elif path_column == "relative":
                # Paths relative to the LibriSpeech root: ``<subset>/<spk>/<chap>/<file>.flac``
                p = u.rel_path
            else:
                raise ValueError("path_column must be 'absolute' or 'relative'")
            w.writerow(
                [
                    p,
                    u.speaker_id,
                    u.chapter_id,
                    u.utterance_id,
                    u.subset,
                    u.rel_path,
                    "0",
                ]
            )
            n += 1
    return n
