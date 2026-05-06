"""Evaluation metrics: VSR, BA, MCD, Harmful Degree."""

from __future__ import annotations

import numpy as np
from scipy import stats


def compute_VSR(
    clean_probs: list[float],
    watermarked_probs: list[float],
    n_repeats: int = 1000,
    n_pairs: int = 100,
    tau: float = 0.25,
) -> float:
    """
    Verification Success Rate via pairwise T-test (same randomization protocol as the paper).

    Expects two paired sequences of *detection* or *score* probabilities—typically
    benign vs watermarked outputs from the **official watermark verifier**. Scripts that
    pass **downstream-task** probabilities (e.g. speaker classifier softmax on the true
    label for clean vs attacked audio) are using this function as a **proxy** for the
    paper's VSR; results must be labeled as such in the write-up.

    H0: Pb = Pw + tau  →  reject if p < 0.01  →  watermark detected
    """
    clean = np.array(clean_probs)
    watermarked = np.array(watermarked_probs)
    n = min(len(clean), len(watermarked))

    if n < n_pairs:
        n_pairs = n

    successes = 0
    rng = np.random.default_rng(42)

    for _ in range(n_repeats):
        idx = rng.choice(n, size=n_pairs, replace=False)
        pb = clean[idx]
        pw = watermarked[idx] + tau   # shift by tau for null hypothesis
        _, p_value = stats.ttest_rel(pb, pw)
        if p_value < 0.01:
            successes += 1

    return successes / n_repeats


def compute_BA(
    preds: list[int],
    true_labels: list[int],
) -> float:
    """Standard closed-set accuracy."""
    correct = sum(p == l for p, l in zip(preds, true_labels))
    return correct / max(1, len(true_labels))


def compute_MCD(
    orig_wav: np.ndarray,
    attacked_wav: np.ndarray,
    sr: int = 16_000,
    n_mfcc: int = 13,
) -> float:
    """
    Mel Cepstral Distortion — lower = better audio quality.
    Same metric used in the original paper.
    """
    try:
        import librosa
        # Ensure 1-D float32 waveform in expected range.
        orig = orig_wav.astype(np.float32).flatten()
        attacked = attacked_wav.astype(np.float32).flatten()
        orig = np.clip(orig, -1.0, 1.0)
        attacked = np.clip(attacked, -1.0, 1.0)

        mfcc_o = librosa.feature.mfcc(y=orig, sr=sr, n_mfcc=n_mfcc)
        mfcc_a = librosa.feature.mfcc(y=attacked, sr=sr, n_mfcc=n_mfcc)
        min_len = min(mfcc_o.shape[1], mfcc_a.shape[1])
        diff = mfcc_o[:, :min_len] - mfcc_a[:, :min_len]
        mcd = (10.0 / np.log(10.0)) * np.sqrt(
            2.0 * np.mean(np.sum(diff**2, axis=0))
        )
        # Keep extreme outliers bounded, but avoid saturating most rows.
        return float(np.clip(mcd, 0.0, 200.0))
    except Exception:
        return 0.0


def compute_harmful_degree(
    preds: list[int],
    true_labels: list[int],
) -> float:
    """
    Fraction of watermarked samples misclassified.
    Lower = less harmful. Target: < 0.1 (same as original paper).
    """
    wrong = sum(p != l for p, l in zip(preds, true_labels))
    return wrong / max(1, len(true_labels))
