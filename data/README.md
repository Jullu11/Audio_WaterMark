# Data layout

- **`raw/`** — downloaded corpora. Large files stay local (see `.gitignore`).
- **`processed/`** — manifests and derived lists (CSV), resampled clips if you add them later.

## LibriSpeech

Official corpus: [OpenSLR Resource 12](https://www.openslr.org/12/).

Download (from repository root):

```bash
cd "/Users/jainesh/Downloads/Data pravicy/audio_watermark_robustness"
source .venv/bin/activate   # if you use a venv
pip install -r requirements.txt
PYTHONPATH=. python scripts/download_librispeech.py --subset train-clean-100
```

Quick smoke test (~337 MB):

```bash
PYTHONPATH=. python scripts/download_librispeech.py --subset dev-clean
```

Extracted layout:

```text
data/raw/LibriSpeech/<subset>/<speaker-id>/<chapter-id>/*.flac
```

Build manifest:

```bash
PYTHONPATH=. python scripts/01_prepare_manifest.py \
  --librispeech-root data/raw/LibriSpeech \
  --out data/processed/manifest_librispeech.csv \
  --subsets train-clean-100
```

For multiple subsets, omit `--subsets` or pass several names.

## If `tar` or `gzip -t` says “truncated gzip” / “unexpected end of file”

The `.tar.gz` is **incomplete or corrupt** (often from stopping the download with Ctrl+Z or a dropped connection). Fix:

1. Remove the bad archive and extracted folder:

```bash
rm -f data/raw/train-clean-100.tar.gz
rm -rf data/raw/LibriSpeech/train-clean-100
```

2. Re-download (do **not** suspend the job) and optionally verify before extracting:

```bash
PYTHONPATH=. python scripts/download_librispeech.py --subset train-clean-100 --force --verify-archive
```

New downloads also check that the file size matches the server `Content-Length` header so truncated transfers fail fast.
