# Audio Watermark Robustness (Course Project)

Systematic evaluation of **AUDIO WATERMARK** under advanced transformation attacks, with reproducible scripts and paper-ready outputs.

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
cd "/Users/jainesh/Downloads/Data pravicy/audio_watermark_robustness"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Always set **`PYTHONPATH`** to the project root when running scripts:

```bash
export PYTHONPATH="/Users/jainesh/Downloads/Data pravicy/audio_watermark_robustness"
```

## LibriSpeech: download + manifest

See **`data/README.md`** for download commands. Summary:

```bash
PYTHONPATH=. python scripts/download_librispeech.py --subset train-clean-100
PYTHONPATH=. python scripts/01_prepare_manifest.py \
  --librispeech-root data/raw/LibriSpeech \
  --out data/processed/manifest_librispeech.csv
PYTHONPATH=. python scripts/02_make_splits.py --write-per-split
```

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

### Official watermark code (optional / Linux GPU)

See **`external/README.md`** for the cloned USENIX authors’ repo, conda setup, checkpoint downloads, and commands (`train_all.py`, `verify_watermark.py`).

## After experiments: paper + submission

See **`NEXT_STEPS.md`** (checklist: narrative, figure refresh, ACM outline, final zip/PDF).

## Reproducing the report

As you finalize experiments, record commands that produce each table/figure:

| Artifact | Command |
|----------|---------|
| Closed-set test BA | `PYTHONPATH=. python scripts/04_eval_speaker_id.py` |
| BA under attacks + sweep CSV | `PYTHONPATH=. python scripts/06_robustness_sweep.py` |
| Figure + LaTeX snippet + bundle | `PYTHONPATH=. python scripts/07_plot_robustness_sweep.py` then `08_bundle_submission.py` |
| Verify metrics + files | `PYTHONPATH=. python scripts/09_verify_submission.py` |
| One-shot regenerate plot + bundle + verify | `PYTHONPATH=. python scripts/10_reproduce_artifacts.py` |
| Markdown table for appendix | `PYTHONPATH=. python scripts/10_export_markdown_table.py` |
