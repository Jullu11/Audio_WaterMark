#!/usr/bin/env python3
"""Generate a proposal-vs-implementation gap report.

This script does not run heavy training. It inspects files, checkpoints, and outputs
to produce a clear status summary for the report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _yn(v: bool) -> str:
    return "YES" if v else "NO"


def main() -> int:
    p = argparse.ArgumentParser(description="Proposal gap status report.")
    p.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Project root.",
    )
    p.add_argument(
        "--out-md",
        type=Path,
        default=None,
        help="Optional markdown output path (default: results/submission/proposal_gap_report.md).",
    )
    args = p.parse_args()

    root = args.root
    out_md = args.out_md or (root / "results" / "submission" / "proposal_gap_report.md")
    out_md.parent.mkdir(parents=True, exist_ok=True)

    attack_csv = root / "results" / "tables" / "attack_sweep.csv"
    metrics_file = root / "src" / "metrics" / "__init__.py"
    transforms_file = root / "src" / "attacks" / "transforms.py"

    # Proposal targets
    proposed_advanced = {
        "voice_conversion": False,
        "neural_style_transfer_deepafx": False,
        "asr_tts": False,
        "audio_super_resolution": False,
        "generative_enhancement": False,
    }
    existing_attack_names: set[str] = set()
    if transforms_file.is_file():
        text = transforms_file.read_text(encoding="utf-8")
        # Minimal static checks
        if "voice_conversion" in text:
            proposed_advanced["voice_conversion"] = True
        if "asr_tts" in text:
            proposed_advanced["asr_tts"] = True
        if "deepafx" in text.lower():
            proposed_advanced["neural_style_transfer_deepafx"] = True
        if "audiosr" in text.lower():
            proposed_advanced["audio_super_resolution"] = True
        if "voicefixer" in text.lower() or "deepfilter" in text.lower():
            proposed_advanced["generative_enhancement"] = True

    # Models promised
    model_targets = ["resnet18", "vggm", "ecapa"]
    model_ckpts = {
        m: (root / "results" / m / "speaker_cnn.pt").is_file() for m in model_targets
    }

    # Dataset scope
    manifest = root / "data" / "processed" / "manifest_with_split.csv"
    has_librispeech = False
    has_voxceleb = False
    if manifest.is_file():
        mdf = pd.read_csv(manifest)
        subset_col = "subset" if "subset" in mdf.columns else None
        if subset_col:
            vals = {str(v).lower() for v in mdf[subset_col].dropna().unique()}
            has_librispeech = any("clean" in v or "librispeech" in v for v in vals)
            has_voxceleb = any("vox" in v for v in vals)

    # Official pipeline
    official_verify = root / "external" / "audiowatermark.github.io" / "code" / "verify_watermark.py"
    official_train = root / "external" / "audiowatermark.github.io" / "code" / "train_all.py"
    official_present = official_verify.is_file() and official_train.is_file()

    # Metrics and run outputs
    has_core_metrics = False
    if metrics_file.is_file():
        txt = metrics_file.read_text(encoding="utf-8")
        has_core_metrics = all(
            name in txt for name in ("compute_VSR", "compute_BA", "compute_MCD")
        )

    n_rows = 0
    n_completed = 0
    n_failed = 0
    if attack_csv.is_file():
        df = pd.read_csv(attack_csv)
        n_rows = len(df)
        if "BA" in df.columns and "VSR" in df.columns:
            ok = df["BA"].notna() & df["VSR"].notna()
            n_completed = int(ok.sum())
            n_failed = int((~ok).sum())
        if "attack" in df.columns:
            existing_attack_names = set(df["attack"].dropna().astype(str).tolist())

    lines = [
        "# Proposal Gap Report",
        "",
        "## Scope Status",
        f"- Advanced attack categories implemented (proposal 5): **{sum(proposed_advanced.values())}/5**",
        f"- LibriSpeech dataset present: **{_yn(has_librispeech)}**",
        f"- VoxCeleb dataset present: **{_yn(has_voxceleb)}**",
        f"- Promised model checkpoints present (`resnet18`, `vggm`, `ecapa`): **{sum(model_ckpts.values())}/3**",
        f"- Official AUDIO WATERMARK pipeline files present: **{_yn(official_present)}**",
        f"- Core metrics implemented (VSR, BA, MCD): **{_yn(has_core_metrics)}**",
        "",
        "## Advanced Attack Categories",
        f"- Voice Conversion: **{_yn(proposed_advanced['voice_conversion'])}**",
        f"- Neural Style Transfer (DeepAFX-ST): **{_yn(proposed_advanced['neural_style_transfer_deepafx'])}**",
        f"- ASR+TTS Re-synthesis: **{_yn(proposed_advanced['asr_tts'])}**",
        f"- Audio Super-Resolution (AudioSR): **{_yn(proposed_advanced['audio_super_resolution'])}**",
        f"- Generative Enhancement (DeepFilterNet/Voicefixer): **{_yn(proposed_advanced['generative_enhancement'])}**",
        "",
        "## Latest Sweep Snapshot",
        f"- attack_sweep rows: **{n_rows}**",
        f"- completed rows (BA/VSR present): **{n_completed}**",
        f"- failed rows (dependency/runtime): **{n_failed}**",
        f"- attack names in latest sweep: `{sorted(existing_attack_names)}`",
        "",
        "## Priority Next Steps",
        "1. **Official verifier (optional off-Mac):** If you have access to Linux/CUDA, run `verify_watermark.py` and report **official** VSR alongside **proxy** metrics. If you stay on **Mac (e.g. M4) only**, document that official reproduction was not run and rely on proxy BA/VSR from the speaker-ID sweep.",
        "2. **VoxCeleb:** Skip unless your write-up requires a second corpus; this repo’s defaults are **LibriSpeech-only**.",
        (
            "3. Install optional attack stacks where needed (DeepFilterNet, DeepAFX checkpoints, AudioSR) "
            "and re-run with `--with-optional-dep-attacks` if sweeps show failed rows."
            if sum(proposed_advanced.values()) < 5
            else "3. Optional: install optional-dep stacks (DeepFilterNet / DeepAFX / AudioSR) as needed; on Apple Silicon some rows may stay skipped if a stack is Linux/GPU-only."
        ),
        "4. In the paper, distinguish **proxy VSR** (speaker-ID softmax protocol in `11_watermark_attack_sweep.py`) from **official** verifier VSR when applicable.",
        "5. Add Figure-12-style attacks still missing from your comparison table (e.g. SCALE-UP, SNR/ANR as in the paper) only if your rubric requires them.",
    ]

    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved -> {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
