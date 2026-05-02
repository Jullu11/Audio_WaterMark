# Next steps (after code + results)

Use this as a **paper and submission** checklist. Experiments in this repo are already wired; what remains is **writing** and **packaging**.

## 1. Lock your story (30 min)

- **Claim:** You measure **closed-set speaker identification accuracy (BA)** on LibriSpeech under **waveform attacks** (noise, bandwidth, time-scale, resampling, quantization). This is a **downstream proxy** for “does identity-related behavior survive processing?” — say so explicitly.
- **Limitation:** Full **AUDIO WATERMARK** (Guo et al.) training/verification was **not** reproduced on macOS; cite their paper for watermark-specific BA/VSR and position your work as **orthogonal robustness analysis** unless you later run their code on **Linux + CUDA**.

## 2. Refresh figures and bundle (5 min)

From the project root:

```bash
source .venv/bin/activate
export PYTHONPATH="$PWD"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$PWD/.matplotlib_cache}"
mkdir -p "$MPLCONFIGDIR"

python scripts/10_reproduce_artifacts.py
python scripts/10_export_markdown_table.py
```

Or step-by-step: `07_plot_robustness_sweep.py` → `08_bundle_submission.py`.

Confirm:

- `results/figures/robustness_bars.png`
- `results/submission/` (CSV, JSON metrics, optional `table_robustness_snippet.tex`, `table_robustness.md`)

## 3. Write the ACM paper (~6 pages)

Suggested outline:

1. **Introduction** — Audio watermarks and robustness; motivation for attack suite.
2. **Background** — Brief summary of Guo et al. (USENIX Security 2025); your metric definitions (**BA** here = speaker-ID accuracy).
3. **Threat model / attacks** — Map each bar to `src/attacks/transforms.py` (parameters match sweep CSV).
4. **Experimental setup** — LibriSpeech subset, utterance splits, CNN+log-mel baseline (`SpeakerModel`).
5. **Results** — Table from `robustness_sweep_deduped.csv` + figure `robustness_bars.png`.
6. **Discussion** — Proxy task; no full watermark pipeline on this machine; future work (GPU Linux, MCD, watermarked audio).
7. **Reproducibility** — Copy commands from `README.md` and `results/submission/README.md`.

## 4. Final package for the course

- **Code:** This repository (or zip) with `README.md` and `NEXT_STEPS.md`.
- **Report:** PDF in `paper/` (or upload per course instructions).
- **Artifacts:** Point the grader to `results/submission/` and `results/figures/`.
- **Optional:** If the rubric requires “official” watermark numbers, ask the instructor **once** whether **proxy + honest limitations** is acceptable, or plan a **single** remote GPU run.

## 5. Last-day sanity check (10 min)

Automated consistency check (baseline vs sweep “clean”, figure + submission folder):

```bash
export PYTHONPATH="$PWD"
python scripts/09_verify_submission.py
```

Optional manual rerun:

```bash
python scripts/04_eval_speaker_id.py
python scripts/05_eval_robustness.py --attack none
```

Numbers should match `eval_test_metrics.json` and the “clean” row in your sweep (the verifier uses `--atol 1e-5` by default).

---

**Done when:** PDF submitted, zip/repo linked, and one paragraph in the paper states **exactly** what was measured (dataset, split, model, attacks, metric).
