# Ablations

All eleven changes tried against the baseline training config (see
`README.md` § Training configuration), what instrument measured each, and
whether it was kept.

Instruments, by resolution (standard error on macro AUC):

| Instrument | SE | Why |
|---|---|---|
| gold-58 | ≈0.03 | only ground-truth labels available, but n=58 |
| val-870 (one fold) | ≈0.01 | held-out fold of the report-labeled set |
| OOF-4349 | ≈0.004 | every report-labeled study, each scored by the fold that held it out |
| LB-1300 | ≈0.008 | public leaderboard subset |

## The eleven experiments

| # | Change | Instrument | Result | Verdict |
|---|---|---|---|---|
| 1 | b0 → b2 | gold-58 | .876 → .877 | null |
| 2 | b0 → b3 | gold-58 | .876 → .881 | null |
| 3 | KU 16 → 24 + max-pool head | gold-58 | .876 → .870 | null |
| 4 | 3 → 5 planes (add PD/T1) | gold-58 | .876 → .865 | null |
| 5 | 3 → 5 planes, 5-fold | **OOF** | **.8515 → .8516** | null |
| 6 | 7 → 14 epochs | val-870 | .860 → .862 peak, then 8 epochs of decay | null |
| 7 | 3-backbone ensemble | gold-58 + paired bootstrap | +0.0092, CI [−0.0039, +0.0222] | inconclusive |
| 8 | + 5-plane model (4-model) | gold-58 + paired bootstrap | +0.0005, CI [−0.0052, +0.0063] | null |
| 9 | per-finding attention pooling (P1) | val-870 | .8600 → .8631 (peak .8667) | null |
| 10 | 2.5D, 3 slices as RGB (P2) | val-870 | .8600 → .8556 (peak .8598) | null |
| 11 | 4-table mean labels (P3) | val-870 | .8600 → .8604 (peak .8623) | null |
| — | **fused labels + 5-fold** | **LB** | **0.891 → 0.903** | **kept** |

Experiments 9–11 (the probe harness, `src/05_probe.py`) ran against a decision
threshold fixed **before** the runs: act only on val ≥ 0.870. All three missed
it. Probe spread across BASE/P1/P2/P3 was 0.0075 — one-fold noise, not signal.

Note: probes were single-fold (SE ≈ 0.01), so a true effect smaller than
about +0.01 would likely have been invisible to this harness. See
`measurement.md` and `README.md` § Limitations.

## Why #1–#4 and #9–#11 are "null" and not "failed"

Each of these changed one knob and measured on an instrument with a
standard error larger than the effect it was hoping to detect. A result
inside noise is not evidence the change doesn't help — it's evidence this
harness can't tell. The ones that mattered (#5, fused labels + 5-fold) were
re-run on a tighter instrument (OOF or LB) before being trusted.

## Probe epoch curves (fold 0, evaluated against a fixed v4_blend reference)

```
BASE  .796 .822 .834 .854 .856 .859 .860        final .8600
P1    .7976 .8336 .8533 .8634 .8667 .8637 .8631 final .8631  peak .8667
P2    .7661 .8239 .8426 .8501 .8598 .8566 .8556 final .8556  peak .8598
P3    .7870 .8248 .8527 .8514 .8623 .8596 .8604 final .8604  peak .8623
```

## 14-epoch run, fold 0

```
.776 .823 .835 .842 .845 .862 .858 .856 .847 .846 .845 .845 .845 .844
```

Peak at epoch 5, then eight consecutive epochs of decline. The 7-epoch run
(the kept config) ended at .860. The apparent "still improving at the last
epoch" pattern seen in 7-epoch runs is an artefact of OneCycle annealing the
LR to zero — it happens wherever the schedule ends, independent of whether
more training would actually help.

## 5-fold runs (v4_blend labels)

**3-plane** (the kept config)
- per-fold gold-58: .8677 .8695 .8755 .8803 .8577
- per-fold best val: .860 .853 .852 .851 .848
- **OOF .8515 · 5-fold gold .8914 · LB 0.903**

**5-plane**
- per-fold gold-58: .8614 .8596 .8469 .8608 .8644
- **OOF .8516 · 5-fold gold .8737**

The pair above (3-plane vs 5-plane, same folds, same labels) differs by
**0.0177 on gold-58** and **0.0001 on OOF**. This is the cleanest single
demonstration in the project that gold-58 cannot resolve most of the
decisions made against it — see `measurement.md`.

## Ensemble analysis (single-split models, gold-58)

| model | gold |
|---|---|
| b0 | 0.880 |
| b2 | 0.873 |
| b3 | 0.881 |
| 5-plane | 0.865 |

Mean per-label prediction correlation:

| pair | r |
|---|---|
| b0 vs b2 | 0.890 |
| b0 vs b3 | 0.905 |
| b2 vs b3 | 0.897 |
| b0 vs 5p | 0.883 |
| b2 vs 5p | 0.896 |
| b3 vs 5p | 0.886 |

| ensemble | gold |
|---|---|
| b0+b2 | 0.890 |
| b0+b3 | 0.887 |
| b2+b3 | 0.886 |
| all 3 | 0.891 |
| all 3, rank-average | 0.890 |
| all 4 (with 5-plane) | 0.891 |

Every pair sits at 0.88–0.91 correlation. Models that correlated that highly
had little independent error left for averaging to remove, which is the
direct explanation for why adding a fourth model moved nothing
(+0.0005, CI [−0.0052, +0.0063] by paired bootstrap, experiment #8).

Per-label, 3-model ensemble vs gold-58:

| Finding | AUC |
|---|---|
| ACL | 0.952 |
| MCL | 0.939 |
| Medial Meniscus | 0.879 |
| Lateral Meniscus | 0.780 |
| Medial OA | 0.975 |
| Lateral OA | 0.839 |
| PF OA | 0.853 |
| Effusion | 0.897 |
| Synovitis | 0.756 |
| Baker's | 0.975 |
| Contusion | 0.928 |
| Fracture | 0.912 |

Lateral Meniscus (0.780) is the one finding where the image-trained model
clearly underperforms the text-based gold labels (teacher AUC 0.881, see
`measurement.md`) — the signal is plausibly in the images and the model
isn't extracting it, likely a resolution problem on a small structure.
Synovitis (0.756) is the opposite case: text-label noise, not a model gap
(see `measurement.md`).
