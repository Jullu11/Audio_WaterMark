# Audio Watermark Robustness (Course Project)

Systematic evaluation of **AUDIO WATERMARK** under advanced transformation attacks, with reproducible scripts and paper-ready outputs.

## Scope (this checkout)

- **Dataset:** **LibriSpeech only** (manifests under `data/processed/`). No second corpus is required for the default scripts.
- **Platform:** **macOS on Apple Silicon (e.g. M4)** is a supported dev/eval target. Use a normal Python venv + PyTorch with **MPS** (`--device auto` / `mps` in scripts that accept it). Some optional attacks (diffusion SR, large enhancement stacks) may be slow on CPU/MPS or need extra installs—skip them with the sweep flags documented in `scripts/11_watermark_attack_sweep.py` if they do not run locally.
- **Official USENIX pipeline** (`external/audiowatermark.github.io/code/`) is **optional** here: upstream conda envs target Linux/CUDA (see `external/README.md`). On Mac you can still run **proxy BA / proxy VSR** via the speaker-ID sweep; state that limitation clearly in the paper if you do not reproduce `verify_watermark.py` on a GPU machine.

## Layout

- `external/audiowatermark.github.io/` — official **AUDIO WATERMARK** code (pinned commit in `external/README.md`)
- `configs/` — experiment configs (seeds, strengths, paths)
- `data/` — local audio mirrors (see `data/README.md`) + manifests under `data/processed/`
- `external/` — vendored third-party code pinned by commit (optional)
- `src/` — library code (datasets, attacks, models, metrics, plotting)
- `scripts/` — CLI entrypoints (`scripts/README.md`)
- `results/` — logs, CSV tables, figures (generated)
- `paper/` — ACM source + exported PDF
- `tests/` — sanity checks

## Environment

```bash
cd "/path/to/Audio_WaterMark"   # repo root (directory that contains `src/` and `scripts/`)
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Always set **`PYTHONPATH`** to the project root when running scripts:

```bash
export PYTHONPATH="$(pwd)"
```

## LibriSpeech: download + manifest

See **`data/README.md`** for download commands. Summary:

```bash
PYTHONPATH=. python scripts/download_librispeech.py --subset train-clean-100
PYTHONPATH=. python scripts/01_prepare_manifest.py \
  --librispeech-root data/raw/LibriSpeech \
  --out data/processed/manifest_librispeech.csv \
&& PYTHONPATH=. python scripts/02_make_splits.py --write-per-split
```

Using `&&` ensures split generation does not run if manifest creation fails.

This produces `data/processed/manifest_with_split.csv` (and optional `data/processed/splits/manifest_{train,val,test}.csv`). Default **`--strategy utterance`** assigns utterances per speaker so train/val/test share the same **251 speakers** (closed-set softmax for `03_train_speaker_id.py`). **`dev-clean`** rows remain `split=test` but use **different speakers** than `train-clean-100` (not used by the closed-set classifier).

### Speaker-ID baseline (closed-set BA)

```bash
PYTHONPATH=. python scripts/03_train_speaker_id.py --epochs 5 --batch-size 32
```

Checkpoint: `results/speaker_baseline/speaker_cnn.pt`, label map: `speaker_label_map.json`.

### Evaluate closed-set test BA

```bash
PYTHONPATH=. python scripts/04_eval_speaker_id.py
```

Writes `results/speaker_baseline/eval_test_metrics.json` (defaults: `train-clean-100` + `split=test`).

### Robustness (BA under waveform attacks)

After training, measure **closed-set accuracy** when degradations are applied *before* the model (same manifest/checkpoint as `04`):

```bash
PYTHONPATH=. python scripts/05_eval_robustness.py --attack none
PYTHONPATH=. python scripts/05_eval_robustness.py --attack gaussian_noise --attack-kw snr_db=20
PYTHONPATH=. python scripts/05_eval_robustness.py --attack lowpass --attack-kw cutoff_hz=4000
```

Attack names: `none`, `gaussian_noise`, `time_stretch`, `lowpass`, `highpass`, `resample_chain`, `quantize`.  
Outputs JSON under `results/speaker_baseline/` (see `--metrics-json`).

Preset grid → CSV:

```bash
PYTHONPATH=. python scripts/06_robustness_sweep.py
```

Appends rows to `results/speaker_baseline/robustness_sweep.csv`.

### Figure + submission bundle (paper)

After you have `robustness_sweep.csv`, dedupe, plot, and copy artifacts for the write-up:

```bash
export MPLCONFIGDIR="$PWD/.matplotlib_cache"   # optional; avoids Matplotlib cache permission issues
mkdir -p "$MPLCONFIGDIR"
PYTHONPATH=. python scripts/07_plot_robustness_sweep.py
PYTHONPATH=. python scripts/08_bundle_submission.py
```

Outputs:

- `results/figures/robustness_bars.png` — horizontal bar chart (closed-set accuracy vs attack).
- `results/speaker_baseline/robustness_sweep_deduped.csv` — one row per attack (duplicate sweep runs removed).
- `results/submission/` — `README.md`, metrics JSON, deduped CSV, figure copy, optional `table_robustness_snippet.tex`, `bundle_manifest.json`.

### Main attack sweep used in this checkout (BA / proxy VSR / MCD)

The primary comparative run in this repo is:

```bash
# full default grid (skips optional dependency attacks by default)
PYTHONPATH=. python scripts/11_watermark_attack_sweep.py --device auto \
  --out-csv results/tables/attack_sweep.csv

# recommended on macOS if ASR+TTS is unstable:
PYTHONPATH=. python scripts/11_watermark_attack_sweep.py --device auto \
  --exclude-attacks asr_tts \
  --out-csv results/tables/attack_sweep.csv

# plot BA vs proxy VSR trade-off
PYTHONPATH=. python scripts/12_plot_ba_vsr.py \
  --in-csv results/tables/attack_sweep.csv \
  --out-png results/figures/ba_vs_vsr.png
```

**Important metric note:** `11_watermark_attack_sweep.py` reports **proxy VSR** (paired statistical protocol applied to speaker-model probabilities), not official `verify_watermark.py` VSR.

### Official watermark code (optional / Linux GPU)

See **`external/README.md`** for the cloned USENIX authors’ repo, conda setup, checkpoint downloads, and commands (`train_all.py`, `verify_watermark.py`).

Helper wrapper scripts added in this repo:

```bash
# proposal gap report (what is implemented vs promised)
PYTHONPATH=. python scripts/13_proposal_gap_report.py

# run official workflow entrypoints from vendored repo
PYTHONPATH=. python scripts/14_run_official_watermark.py --task train_benign
PYTHONPATH=. python scripts/14_run_official_watermark.py --task verify_generate
PYTHONPATH=. python scripts/14_run_official_watermark.py --task verify_eval
```


## Reproducing the report

As you finalize experiments, record commands that produce each table/figure:

| Artifact | Command |
|----------|---------|
| Closed-set test BA | `PYTHONPATH=. python scripts/04_eval_speaker_id.py` |
| Main comparative sweep (BA + proxy VSR + MCD) | `PYTHONPATH=. python scripts/11_watermark_attack_sweep.py --device auto --out-csv results/tables/attack_sweep.csv` |
| Main trade-off figure | `PYTHONPATH=. python scripts/12_plot_ba_vsr.py --in-csv results/tables/attack_sweep.csv --out-png results/figures/ba_vs_vsr.png` |
| Legacy BA-only robustness sweep | `PYTHONPATH=. python scripts/06_robustness_sweep.py` |
| Legacy figure + bundle | `PYTHONPATH=. python scripts/07_plot_robustness_sweep.py` then `08_bundle_submission.py` |
| Legacy verify metrics + files | `PYTHONPATH=. python scripts/09_verify_submission.py` |
| Legacy one-shot regenerate | `PYTHONPATH=. python scripts/10_reproduce_artifacts.py` |
| Markdown table for appendix | `PYTHONPATH=. python scripts/10_export_markdown_table.py` |
