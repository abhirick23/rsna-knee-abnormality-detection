# RSNA Knee Abnormality Detection — raw experimental record

Everything below is measured output from the project. Source of truth for
`README.md`, `results/ablations.md` and `results/measurement.md`.

Competition: RSNA Knee Abnormality Detection, Kaggle, 30 Jul – 22 Oct 2026.
Metric: macro-averaged AUC ROC over 12 binary findings.
Hardware: single Kaggle T4, 30 GPU-hours/week.

---

## 1. Dataset

| | |
|---|---|
| Train studies | 4,407 |
| Series | 24,371 (~5.5/study) |
| DICOM files | 819,640 |
| Size | 569 GB |
| Studies with expert labels | **58** |
| Studies with only a free-text report | 4,349 |
| Reports | multilingual (Greek, Turkish, English, others) |
| Reports at test time | not provided |
| Test studies (hidden) | ~1,300 |

Label order required by the submission format:

`ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture`

### Series inventory

| Plane | Fluid-sensitive | Non-fluid-sensitive |
|---|---|---|
| Sagittal | 4,667 | 5,197 |
| Coronal | 4,624 | 3,985 |
| Axial | 4,719 | 1,179 |

Studies with ≥1 non-FS series: sagittal 4,266 (96.8%), coronal 3,406 (77.3%),
axial 857 (19.4%); 4,407/4,407 have at least one somewhere.

**`Fluid_Sensitive == 0` ⟺ `Fat_Suppression == 0`, exactly** — the two flags are
perfectly collinear, so the non-FS group is precisely the PD/T1 (non-fat-suppressed)
sequences, the ones menisci are normally read on.

Plane coverage of the 3-plane fluid-sensitive shards: `[0.942, 0.964, 1.000]`.
With the two non-FS planes appended: `[0.942, 0.964, 1.000, 0.968, 0.773]`.

---

## 2. Preprocessing

Per study, per plane: pick one fluid-sensitive series (most slices wins), sort
slices by `ImagePositionPatient · normal`, sample K=24 evenly, percentile-normalise
(0.5 / 99.5), pad to square, resize to S=192, store uint8.

- Shard format: `x (256, P, 24, 192, 192) uint8`, `m (256, P) uint8`, `uid (256,)`
- 18 shards, chunk size 256, ~12 GB for 3 planes, ~7.8 GB for the 2 non-FS planes
- ~0.55 s/study on 4 CPUs; DICOM read ~16 ms/slice, no extra decoders needed
- Required `cv2.setNumThreads(0)` and `maxtasksperchild=64` to avoid stalls

---

## 3. Training configuration (held constant unless an experiment says otherwise)

```
backbone        efficientnet_b0 (timm, pretrained, in_chans=1)
head            per-plane attention pooling -> concat(P*D + P) -> Linear(12)
slices used     KU = 16 of 24 (random subset in training, even grid at eval)
batch size      4
optimiser       AdamW, lr 3e-4, weight decay 1e-2
schedule        OneCycleLR, pct_start 0.1
precision       AMP fp16
augmentation    intensity jitter + random roll.  NO FLIPS - medial/lateral
                must stay fixed, flipping would swap the anatomy the labels name
loss            BCE on soft labels
epochs          7
split           5-fold KFold(shuffle=True, random_state=0) over the 4,349
                report-only studies; the 58 gold studies are NEVER trained on
```

Epoch timings (T4): b0 3-plane 282 s · b0 KU=24 415 s · b2 465 s ·
b3 (BS=3) 550 s · b0 5-plane 528 s · per-finding pooling 318 s · 2.5D 276 s.

---

## 4. Leaderboard results (public LB, ~1,300 studies, SE ≈ 0.008)

| # | Configuration | Labels | LB |
|---|---|---|---|
| 1 | b0, single 85/15 split, 3-plane | own Gemini (0.866) | **0.891** |
| 2 | b2, single 85/15 split, 3-plane | own Gemini (0.866) | **0.890** |
| 3 | b0, **5-fold**, 3-plane | **v4_blend (0.893)** | **0.903** |

Run 3 changed two things at once (labels and fold ensembling); the split
between them was never isolated. Stated as a limitation, not a finding.

---

## 5. Label quality (macro AUC of each label table against the 58 gold studies)

| Source | File | AUC | Coverage |
|---|---|---|---|
| own, Gemini 3.5-flash-lite | `report_labels.csv` | 0.866 | 1.00 |
| lixin73 | `labels_llm_gpt56sol.csv` | 0.835 | 1.00 |
| lixin73 | `report_labels_gpt56sol.csv` | 0.835 | 1.00 |
| Pilkwang Kim | `report_labels_v2.csv` | 0.870 | 1.00 |
| stevenleehans | `llm_labels_full.csv` | 0.878 | 1.00 |
| stevenleehans | `llm_labels_v2.csv` | 0.887 | 1.00 |
| **stevenleehans** | **`llm_labels_v4_blend.csv`** | **0.893** | 1.00 |

Naive fusion (mean of the top-n sources, ranked by the above):

| top-n | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| macro AUC | 0.890 | 0.890 | 0.890 | 0.886 | 0.887 | 0.888 |

No fusion beat the best single table. Caveat worth stating: all of these
differences are smaller than the gold-58 standard error, so this ranking is
itself weak evidence. v4_blend was adopted partly on that basis and partly
because it is what the strongest public solutions train on.

### Own Gemini labels, per finding, vs gold-58

| Finding | AUC |
|---|---|
| ACL | 0.960 |
| MCL | 0.954 |
| Medial Meniscus | 0.932 |
| Baker's | 0.917 |
| Medial OA | 0.927 |
| Lateral Meniscus | 0.881 |
| Fracture | 0.833 |
| Contusion | 0.813 |
| Effusion | 0.816 |
| PF OA | 0.819 |
| Lateral OA | 0.812 |
| Synovitis | 0.730 |

Synovitis is irreducible from text: the five worst misses never contain
"synovi-" anywhere in the report. The gold label there comes from image review,
so no prompt can recover it.

### Prompt-engineering experiment (single-label prompts with graded confidence anchors)

| Finding | baseline | rewritten | Δ |
|---|---|---|---|
| Lateral OA | 0.812 | 0.841 | +0.029 |
| PF OA | 0.819 | 0.793 | −0.026 |
| Effusion | 0.816 | 0.721 | −0.095 |
| Contusion | 0.813 | 0.745 | −0.068 |

Replication of the one apparent gain (Lateral OA):

| run | AUC |
|---|---|
| gemini-3.5-flash-lite, temp 0 | 0.841 |
| gemini-3.1-flash-lite, temp 0 | 0.775 |
| gemini-3.5-flash-lite, temp 0.3 | 0.846 |

Straddles the 0.812 baseline on both sides → the +0.029 was chance, from
running four tests on a 58-sample set. Relabeling abandoned.

---

## 6. The eleven experiments

Instruments: **gold-58** (SE ≈ 0.03 macro), **OOF-4349** (SE ≈ 0.004),
**val-870** one fold (SE ≈ 0.01), **LB-1300** (SE ≈ 0.008).

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

Experiments 9–11 ran against a decision threshold fixed **before** the runs:
act only on val ≥ 0.870. All three failed it. Probe spread was 0.0075 across
BASE/P1/P2/P3 — one-fold noise.

### Probe epoch curves (fold 0, evaluated against a fixed v4_blend reference)

```
BASE  .796 .822 .834 .854 .856 .859 .860        final .8600
P1    .7976 .8336 .8533 .8634 .8667 .8637 .8631 final .8631  peak .8667
P2    .7661 .8239 .8426 .8501 .8598 .8566 .8556 final .8556  peak .8598
P3    .7870 .8248 .8527 .8514 .8623 .8596 .8604 final .8604  peak .8623
```

### 14-epoch run, fold 0

```
.776 .823 .835 .842 .845 .862 .858 .856 .847 .846 .845 .845 .845 .844
```
Peak at epoch 5, then eight consecutive epochs of decline. The 7-epoch run
ended at .860. The apparent "still improving at the last epoch" in 7-epoch runs
is an artefact of OneCycle annealing the LR to zero — it happens wherever the
schedule ends.

### 5-fold runs (v4_blend labels)

**3-plane** — per-fold gold-58: .8677 .8695 .8755 .8803 .8577
per-fold best val: .860 .853 .852 .851 .848
**OOF .8515 · 5-fold gold .8914 · LB 0.903**

**5-plane** — per-fold gold-58: .8614 .8596 .8469 .8608 .8644
**OOF .8516 · 5-fold gold .8737**

Note the pair in bold: on gold-58 these two differ by **0.0177**; on OOF they
differ by **0.0001**. Same models, same folds, same labels.

### Ensemble analysis (single-split models, gold-58)

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

Every pair sits at 0.88–0.91. Members that correlated that highly had little
independent error left to average away, which is why adding a fourth model
moved nothing (+0.0005, CI [−0.0052, +0.0063]).

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

---

## 7. Evidence that gold-58 cannot support the decisions made on it

1. **Same checkpoint, two evaluations.** model_b2 scored 0.877 during training
   and 0.873 on re-evaluation. Only the random slice subset differed → ≈0.004
   of jitter before any model change is considered.

2. **Two models, two instruments, opposite answers.** 3-plane vs 5-plane 5-fold:
   gold-58 says 0.0177 apart, OOF-4349 says 0.0001 apart.

3. **A per-label AUC swinging 0.11 inside one run.** Lateral Meniscus across
   epochs of the 5-plane run: `.748 .805 .817 .706 .752 .740 .725`. Prevalence
   ≈0.168 → about 10 positives in 58 studies, giving SE ≈ 0.09. An early reading
   of 0.817 was briefly taken as confirmation of the PD/T1 hypothesis; the full
   curve shows it was noise.

4. **The gap the LB exposed.** Gold-58 rated 5-fold+v4_blend at 0.8914 against
   0.891 for the previous ensemble — "no change". The LB moved 0.891 → 0.903.

5. **Synthetic confirmation** (`stats.py` self-test): a true +0.014 macro AUC
   improvement, measured by paired bootstrap —

   ```
   n=  58  diff +0.0130  95% CI [-0.0410, +0.0669]  NOT resolvable
   n= 870  diff +0.0154  95% CI [-0.0001, +0.0308]  NOT resolvable
   n=4349  diff +0.0142  95% CI [+0.0072, +0.0208]  resolvable
   ```

---

## 8. Leaderboard context

The published 0.958 public blend is **inference-only**: it trains nothing and
loads three public checkpoint datasets (nartaa's CoAtNet-2 trained with OAI
external data, heliosli's ConvNeXt-small R6 with flip TTA, and goodpjw2008's
2.5D ConvNeXt reader at 1/10 weight). That notebook had 284 copies, and the
blend itself ranked 195 of 6,077 — i.e. a large share of the leaderboard above
this project is running shared weights rather than models they trained.

The relevant comparison for a competition-data-only model is goodpjw2008's
reader at **0.929**, trained on 5 folds with soft labels from four public tables
and the 58 gold studies held out. This project reached 0.903 under the same
constraints.

Final standing: ~4,000 of ~6,000.

---

## 9. Credits

Public report-label tables used or evaluated: **stevenleehans**, **Pilkwang Kim**,
**lixin73**. Leaderboard context from the public write-ups by **goodpjw2008**,
**heliosli**, **nartaa (Danial Zakaria)** and **Yash Bishnoi**.

---

## 10. Known gaps / limitations to state explicitly

- The 0.891 → 0.903 jump confounds two changes; neither was isolated.
- No physical-space resampling: every knee is padded and resized to 192 px
  regardless of field of view, so mm-per-pixel varies across studies. The
  0.929 reader resamples to 0.4 mm/px. Untested here.
- Single architecture family throughout (EfficientNet + attention pooling);
  no transformer over slices, no cross-series attention.
- 192 px and 24 stored slices were fixed after the first preprocessing pass
  and never revisited at higher resolution.
- Probes were single-fold (SE ≈ 0.01), so a true +0.005 effect would have been
  missed. The threshold of 0.870 was set to control false positives, at the
  cost of statistical power.
