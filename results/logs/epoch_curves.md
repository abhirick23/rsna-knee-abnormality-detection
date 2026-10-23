# Epoch curves

Raw per-epoch numbers behind the summarized results in `../ablations.md`.
Source: `EXPERIMENT_DATA.md` § 6.

## Probe epoch curves (fold 0, evaluated against a fixed v4_blend reference)

Pre-registered decision threshold: act only on val ≥ 0.870. All three probe
variants missed it.

```
BASE  .796 .822 .834 .854 .856 .859 .860        final .8600
P1    .7976 .8336 .8533 .8634 .8667 .8637 .8631 final .8631  peak .8667
P2    .7661 .8239 .8426 .8501 .8598 .8566 .8556 final .8556  peak .8598
P3    .7870 .8248 .8527 .8514 .8623 .8596 .8604 final .8604  peak .8623
```

- BASE: baseline config (see README § Training configuration)
- P1: per-finding attention pooling
- P2: 2.5D, 3 slices as RGB
- P3: 4-table mean labels

Probe spread across BASE/P1/P2/P3 was 0.0075 final-epoch — consistent with
single-fold noise (SE ≈ 0.01) rather than a real difference between variants.

## 14-epoch run, fold 0

```
.776 .823 .835 .842 .845 .862 .858 .856 .847 .846 .845 .845 .845 .844
```

Peak at epoch 5 (.862), then eight consecutive epochs of decline, ending at
.844. The kept 7-epoch config ends at .860 — near the peak, not because
7 epochs is specially tuned, but because OneCycleLR anneals the learning
rate to zero at the schedule's end, which happens to land near the epoch-5
peak for this run length. Running longer does not continue improving the
model; it mostly re-exposes it to overfitting once the LR has already
annealed down once at epoch 7's point in the original schedule.

## 5-fold runs (v4_blend labels)

**3-plane** (the kept config)

```
per-fold gold-58:    .8677 .8695 .8755 .8803 .8577
per-fold best val:   .860  .853  .852  .851  .848
OOF (all folds):     .8515
5-fold gold-58 mean: .8914
Public LB:           0.903
```

**5-plane**

```
per-fold gold-58:    .8614 .8596 .8469 .8608 .8644
OOF (all folds):     .8516
5-fold gold-58 mean: .8737
```

3-plane vs 5-plane: **0.0177 apart on gold-58, 0.0001 apart on OOF** — same
folds, same labels, only the plane count differs. See `../measurement.md`
for why this pair is the cleanest evidence in the project that gold-58
cannot resolve most of the comparisons made against it.

## Single-split model scores (gold-58), used in the ensemble analysis

```
b0        0.880
b2        0.873
b3        0.881
5-plane   0.865
```

(`model_b2` was independently re-evaluated at 0.877 during its own training
run with a different random slice subset — see `../measurement.md` point 1
for what that 0.877 vs 0.873 gap implies about measurement noise.)
