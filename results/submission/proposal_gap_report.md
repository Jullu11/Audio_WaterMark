# Proposal Gap Report

## Scope Status
- Advanced attack categories implemented (proposal 5): **5/5**
- LibriSpeech dataset present: **YES**
- VoxCeleb dataset present: **NO**
- Promised model checkpoints present (`resnet18`, `vggm`, `ecapa`): **0/3**
- Official AUDIO WATERMARK pipeline files present: **YES**
- Core metrics implemented (VSR, BA, MCD): **YES**

## Advanced Attack Categories
- Voice Conversion: **YES**
- Neural Style Transfer (DeepAFX-ST): **YES**
- ASR+TTS Re-synthesis: **YES**
- Audio Super-Resolution (AudioSR): **YES**
- Generative Enhancement (DeepFilterNet/Voicefixer): **YES**

## Latest Sweep Snapshot
- attack_sweep rows: **22**
- completed rows (BA/VSR present): **22**
- failed rows (dependency/runtime): **0**
- attack names in latest sweep: `['gaussian_noise', 'lowpass', 'none', 'quantize', 'resample_chain', 'shrinkpad', 'strip', 'voice_conversion']`

## Priority Next Steps
1. **Official verifier (optional off-Mac):** If you have access to Linux/CUDA, run `verify_watermark.py` and report **official** VSR alongside **proxy** metrics. If you stay on **Mac (e.g. M4) only**, document that official reproduction was not run and rely on proxy BA/VSR from the speaker-ID sweep.
2. **VoxCeleb:** Skip unless your write-up requires a second corpus; this repo’s defaults are **LibriSpeech-only**.
3. Optional: install optional-dep stacks (DeepFilterNet / DeepAFX / AudioSR) as needed; on Apple Silicon some rows may stay skipped if a stack is Linux/GPU-only.
4. In the paper, distinguish **proxy VSR** (speaker-ID softmax protocol in `11_watermark_attack_sweep.py`) from **official** verifier VSR when applicable.
5. Add Figure-12-style attacks still missing from your comparison table (e.g. SCALE-UP, SNR/ANR as in the paper) only if your rubric requires them.
