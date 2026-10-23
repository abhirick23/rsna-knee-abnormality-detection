"""Evaluation utilities for the RSNA Knee Abnormality Detection pipeline.

The central problem in this competition is that the only ground-truth labels
cover 58 studies.  Macro AUC on 58 studies has a standard error of roughly
0.03, and per-label AUC for a rare finding (~10 positives) has a standard
error near 0.09.  Most of the effects worth chasing are smaller than that.

These helpers exist to make that explicit:

  macro_auc            -- the competition metric
  per_label_auc        -- per-label breakdown (unreliable at n=58; see README)
  paired_bootstrap     -- compare two models on the SAME studies, which cancels
                          most shared variance and resolves ~0.015 instead of ~0.05
  bootstrap_ci         -- CI for a single model's macro AUC
  positives_per_label  -- sanity check: how many positives is each AUC built on

LABELS is the exact submission column order required by the competition.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA", "Effusion",
    "Synovitis", "Baker's", "Contusion", "Fracture",
]


def _binarise(T: np.ndarray) -> np.ndarray:
    """Soft targets -> hard labels at 0.5, as the competition metric assumes."""
    return np.asarray(T) >= 0.5


def per_label_auc(P: np.ndarray, T: np.ndarray) -> list[float]:
    """AUC for each of the 12 findings; NaN where a label is degenerate."""
    T = _binarise(T)
    out = []
    for k in range(T.shape[1]):
        pos = T[:, k].sum()
        out.append(roc_auc_score(T[:, k], P[:, k])
                   if 0 < pos < len(T) else np.nan)
    return out


def macro_auc(P: np.ndarray, T: np.ndarray) -> float:
    """Macro-averaged AUC over the labels that are estimable on this sample."""
    return float(np.nanmean(per_label_auc(P, T)))


def positives_per_label(T: np.ndarray) -> dict[str, int]:
    """Positive counts behind each per-label AUC.

    Use this before quoting a per-label number.  On the 58 gold studies,
    Lateral Meniscus has ~10 positives, which is why its AUC swung by 0.11
    between consecutive epochs of a single training run.
    """
    T = _binarise(T)
    return {lab: int(T[:, k].sum()) for k, lab in enumerate(LABELS[:T.shape[1]])}


def bootstrap_ci(P: np.ndarray, T: np.ndarray, n: int = 2000, seed: int = 0,
                 alpha: float = 0.05) -> tuple[float, float, float]:
    """Percentile bootstrap CI for one model's macro AUC.

    Returns (mean, lo, hi).  Resamples are skipped when they leave a label
    with no positives or no negatives, since AUC is undefined there.
    """
    rng = np.random.RandomState(seed)
    N = len(T)
    vals = []
    for _ in range(n):
        i = rng.randint(0, N, N)
        Ti = _binarise(T[i])
        if Ti.sum(0).min() == 0 or Ti.sum(0).max() == N:
            continue
        vals.append(macro_auc(P[i], T[i]))
    vals = np.asarray(vals)
    return (float(vals.mean()),
            float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


def paired_bootstrap(Pa: np.ndarray, Pb: np.ndarray, T: np.ndarray,
                     n: int = 2000, seed: int = 0,
                     alpha: float = 0.05) -> tuple[float, float, float]:
    """CI for macro_auc(A) - macro_auc(B), both scored on the same resample.

    Pairing is what buys the resolution: the two models share the same hard
    and easy studies, so resampling the studies cancels most of the variance
    that dominates an unpaired comparison.  On the 58 gold studies this takes
    the resolvable difference from roughly 0.05 down to roughly 0.015.

    Returns (mean difference, lo, hi).  A CI containing 0 means the data
    cannot distinguish the two models, whatever the point estimate says.
    """
    rng = np.random.RandomState(seed)
    N = len(T)
    diffs = []
    for _ in range(n):
        i = rng.randint(0, N, N)
        Ti = _binarise(T[i])
        if Ti.sum(0).min() == 0 or Ti.sum(0).max() == N:
            continue
        diffs.append(macro_auc(Pa[i], T[i]) - macro_auc(Pb[i], T[i]))
    diffs = np.asarray(diffs)
    return (float(diffs.mean()),
            float(np.percentile(diffs, 100 * alpha / 2)),
            float(np.percentile(diffs, 100 * (1 - alpha / 2))))


def mean_prediction_correlation(Pa: np.ndarray, Pb: np.ndarray) -> float:
    """Mean per-label Pearson correlation between two models' predictions.

    Ensembling pays in proportion to how decorrelated the members are.  Every
    pair in this project landed at 0.88-0.91, which is why averaging different
    backbones bought almost nothing (see results/ablations.md).
    """
    return float(np.mean([np.corrcoef(Pa[:, k], Pb[:, k])[0, 1]
                          for k in range(Pa.shape[1])]))


if __name__ == "__main__":
    # Demonstrates the resolution problem: the SAME true improvement is
    # invisible at n=58 and obvious at n=4349.
    rng = np.random.RandomState(0)
    for N in (58, 870, 4349):
        T = (rng.rand(N, 12) < 0.25).astype(float)
        signal = T * 1.0
        base = signal + rng.randn(N, 12) * 1.00         # noisier model
        better = signal + rng.randn(N, 12) * 0.92       # ~0.01 macro AUC better
        m, lo, hi = paired_bootstrap(better, base, T, n=500)
        print(f"n={N:5d}  diff {m:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  "
              f"{'resolvable' if lo > 0 else 'NOT resolvable'}")
