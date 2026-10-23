# Measurement: why gold-58 can't carry this project alone

The competition provides expert ground-truth labels for only **58** of 4,407
training studies. Macro AUC on 58 studies has a standard error of roughly
0.03 (and much worse per-label — see below), which is larger than most of
the effects this project needed to evaluate. This document is the evidence
for that claim and the measurement strategy adopted in response.

## Label quality: own labels vs public tables (macro AUC against gold-58)

| Source | File | AUC | Coverage |
|---|---|---|---|
| own, Gemini 3.5-flash-lite | `report_labels.csv` | 0.866 | 1.00 |
| lixin73 | `labels_llm_gpt56sol.csv` | 0.835 | 1.00 |
| lixin73 | `report_labels_gpt56sol.csv` | 0.835 | 1.00 |
| Pilkwang Kim | `report_labels_v2.csv` | 0.870 | 1.00 |
| stevenleehans | `llm_labels_full.csv` | 0.878 | 1.00 |
| stevenleehans | `llm_labels_v2.csv` | 0.887 | 1.00 |
| **stevenleehans** | **`llm_labels_v4_blend.csv`** | **0.893** | 1.00 |

Naive fusion (mean of the top-n sources, ranked by the table above) never
beat the best single source:

| top-n | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|
| macro AUC | 0.890 | 0.890 | 0.890 | 0.886 | 0.887 | 0.888 |

Caveat: every difference in both tables above is smaller than gold-58's
standard error, so this ranking is itself weak evidence. `v4_blend` was
adopted partly on this basis and partly because it is what the strongest
public solutions train on (see README § Credits).

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

Synovitis is irreducible from report text: the five worst misses never
contain "synovi-" anywhere in the source report. The gold label there comes
from image review, so no prompt rewrite can recover it — confirmed directly
against the report text, not inferred.

### Prompt-engineering experiment on the four weakest labels

Tried a single-label prompt format with graded 0–100 confidence anchors,
scored against gold-58:

| Finding | baseline | rewritten | Δ |
|---|---|---|---|
| Lateral OA | 0.812 | 0.841 | +0.029 |
| PF OA | 0.819 | 0.793 | −0.026 |
| Effusion | 0.816 | 0.721 | −0.095 |
| Contusion | 0.813 | 0.745 | −0.068 |

Three of four got worse, one by a lot (−0.095 on Effusion — too large to be
noise, and a real signal that the 0–100 anchor-scale format compresses the
low end and destroys ranking information the original prompt preserved).

The one apparent gain (Lateral OA, +0.029) did not replicate:

| run | AUC |
|---|---|
| gemini-3.5-flash-lite, temp 0 | 0.841 |
| gemini-3.1-flash-lite, temp 0 | 0.775 |
| gemini-3.5-flash-lite, temp 0.3 | 0.846 |

The result straddles the 0.812 baseline on both sides across reruns — the
original +0.029 was chance from running four prompt variants on a 58-sample
set (a textbook multiple-comparisons trap: with four tests, expect the best
one to look positive even if all four are equally good). Relabeling was
abandoned; the original combined 12-label prompt stayed as the production
labeler (`src/02_label_reports.py`).

## Five pieces of direct evidence that gold-58 cannot support the decisions made on it

1. **Same checkpoint, two evaluations.** `model_b2` scored 0.877 during
   training and 0.873 on re-evaluation. Only the random slice subset
   differed between runs — about 0.004 of pure measurement jitter exists
   before any model change is even considered.

2. **Two models, two instruments, opposite answers.** 3-plane vs 5-plane,
   both 5-fold, same labels: gold-58 says they differ by **0.0177**;
   OOF-4349 (every report-labeled study, scored out-of-fold) says they
   differ by **0.0001**. Same models. The instrument, not the model, decided
   which answer you'd believe.

3. **A per-label AUC swinging 0.11 inside one run.** Lateral Meniscus across
   epochs of the 5-plane run: `.748 .805 .817 .706 .752 .740 .725`.
   Prevalence ≈0.168 in gold-58 → about 10 positives, giving a per-label
   standard error near 0.09. An early reading of 0.817 was briefly taken as
   confirming a hypothesis about PD/T1 sequences; the full epoch curve shows
   it was noise, not a trend.

4. **The gap the leaderboard exposed.** Gold-58 rated the 5-fold +
   v4_blend model at 0.8914 against 0.891 for the previous ensemble —
   reading as "no change." The public leaderboard moved 0.891 → 0.903 for
   the same change. Gold-58 missed an improvement the hidden test set
   confirmed.

5. **Synthetic confirmation** (`src/stats.py`, run as `python src/stats.py`):
   inject a known true +0.014 macro AUC improvement into synthetic data and
   measure it by paired bootstrap at three sample sizes:

   ```
   n=  58  diff +0.0130  95% CI [-0.0410, +0.0669]  NOT resolvable
   n= 870  diff +0.0154  95% CI [-0.0001, +0.0308]  NOT resolvable
   n=4349  diff +0.0142  95% CI [+0.0072, +0.0208]  resolvable
   ```

   The same true effect is invisible at n=58, borderline at n=870, and
   clearly resolvable at n=4,349. This is the general shape of every
   null/inconclusive result in `ablations.md`.

## What this means for how results were read

- **Paired bootstrap, not point estimates, for any gold-58 comparison.**
  Pairing two models on the same resampled studies cancels most of the
  shared variance and takes the resolvable difference from roughly 0.05
  down to roughly 0.015 (`src/stats.py::paired_bootstrap`). Still not
  enough for the smallest effects tried (experiment #8: +0.0005,
  CI [−0.0052, +0.0063] — genuinely inconclusive, not "no effect").
- **OOF-4349 or LB, not gold-58, decided anything load-bearing.** The one
  change that was kept (fused labels + 5-fold ensembling) was validated on
  OOF and then on the leaderboard, not on gold-58 alone.
- **A pre-registered decision threshold** (act only if val ≥ 0.870) was set
  before running the probe harness (experiments #9–11), specifically to
  avoid rationalizing a borderline single-fold result after seeing it.
- **Negative results are stated as null, not as "doesn't help."** A result
  landing inside an instrument's noise band means the instrument couldn't
  resolve it, not that the true effect is zero. See `ablations.md` and the
  Limitations section of `README.md`.
