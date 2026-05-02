# External: official **AUDIO WATERMARK** implementation

Source repository (project website + paper code):  
https://github.com/audiowatermark/audiowatermark.github.io  

**Pinned commit (this workspace clone):**  
`4dabc456baa2540fa45d2cca3fd371abff026a23`

To reproduce that checkout elsewhere:

```bash
git clone https://github.com/audiowatermark/audiowatermark.github.io.git
cd audiowatermark.github.io
git checkout 4dabc456baa2540fa45d2cca3fd371abff026a23
```

## Where the training code lives

All runnable Python entrypoints are under:

`external/audiowatermark.github.io/code/`

Upstream instructions are duplicated in:

- `external/audiowatermark.github.io/README.md`
- `external/audiowatermark.github.io/code/README.md`

### Dependencies

Authors recommend **Conda** using:

```bash
cd external/audiowatermark.github.io/code
conda env create -f dependency.yml
conda activate audiowatermark   # env name from their YAML (confirm inside file)
```

This stack is **separate** from this repo’s `.venv` (PyTorch/torchaudio versions may differ). Expect separate CUDA/conda setup.

### macOS (Apple Silicon / `osx-arm64`): `dependency.yml` usually **fails**

`dependency.yml` pins **Linux** packages (`ld_impl_linux-64`, `libgcc-ng`, `cudatoolkit`, …) and old **py36** builds. On macOS, `conda env create` returns **`LibMambaUnsatisfiableError`** (nothing provides those builds on `osx-arm64`). This is **expected** — the authors’ file targets **Linux + CUDA**.

**Practical options:**

1. **Linux with NVIDIA GPU** (lab server, cloud VM, or Docker) — use their YAML or a hand-built env with **torch 1.10.x + CUDA** per their pip section.
2. **Document in your report** that full reproduction was run on Linux/GPU; Mac was used for data prep and your own `audio_watermark_robustness` code only.
3. **Do not** expect `conda activate audiowatermark` until an environment is successfully created — if create fails, that env name does not exist.

### Data note (important)

Their README uses a **small 10‑speaker LibriSpeech subset** (Google Drive link in their README).  
Your course repo already has **full `train-clean-100` + `dev-clean`**. Integrating that with their loaders requires **path/format alignment** (`load_data/load_data.py`, `PoisonedLibriDataset`, etc.) — do this when you wire experiments, not before reading their dataset classes.

### Checkpoints (required for watermark generation)

Watermark generation / verification expects several checkpoints under `code/checkpoint/` (GST, DeepAFX, Style Wave‑U‑Net, benign SR model). URLs are listed in their README (Google Drive). Download into the folder layout they document next to `checkpoint/benign/`, etc.

### Typical command sequence (from authors)

1. Train or download benign speaker model: `python train_all.py --tasks train_benign --sr_model resnet18`
2. Generate watermarked samples: `python verify_watermark.py -g`
3. Poison / evaluate VSR-style behavior: `python verify_watermark.py -e -pr 0.1 -epoch 20`

Run these **from** `external/audiowatermark.github.io/code/` so imports resolve.

## Integration plan for *this* repository

1. Recreate their conda env and verify `train_all.py` runs on their 10‑speaker subset (smoke test).
2. Map their metrics (benign vs watermarked accuracy, pairwise protocol) to your report’s **BA / VSR** wording.
3. Replace or extend data roots so **your** manifests (`data/processed/*.csv`) feed generation / evaluation where feasible.
4. Implement **your attack pipeline** on exported FLAC from step 2–3 and compare against Figure‑12 baselines.
