# Scripts

| Script | Purpose |
|--------|---------|
| `download_librispeech.py` | Download & extract LibriSpeech from OpenSLR into `data/raw/` |
| `01_prepare_manifest.py` | Build CSV manifest of FLAC utterances for downstream training/evaluation |
| `02_make_splits.py` | Add `train` / `val` / `test` (`--strategy utterance` = closed-set softmax; `speaker` = disjoint speakers) |
| `03_train_speaker_id.py` | Train small CNN speaker classifier (BA baseline on `train-clean-100`) |
| `04_eval_speaker_id.py` | Closed-set **test** accuracy + `eval_test_metrics.json` |
| `05_eval_robustness.py` | Same as `04`, but applies waveform attacks before forward (BA vs degradation) |
| `06_robustness_sweep.py` | Runs a preset attack grid; appends `robustness_sweep.csv` |
| `07_plot_robustness_sweep.py` | Dedupes sweep CSV, writes `robustness_bars.png` + `robustness_sweep_deduped.csv` |
| `08_bundle_submission.py` | Copies metrics, figure, deduped CSV, optional `.tex` table into `results/submission/` |
| `09_verify_submission.py` | Checks baseline vs sweep clean accuracy, figure + bundle files exist |
| `10_reproduce_artifacts.py` | Runs `07` → `08` → `09` (optional `--with-eval`, `--with-sweep`, `--require-checkpoint`) |
| `10_export_markdown_table.py` | Writes `results/submission/table_robustness.md` from deduped sweep |

Run from project root with `PYTHONPATH=.` (see main `README.md`). **Tests:** `PYTHONPATH=. python -m unittest tests.test_attacks -v`
