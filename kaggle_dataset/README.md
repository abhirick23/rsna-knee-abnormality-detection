# RSNA Knee Report Labels (Gemini)

Soft labels for the 12 findings in the [RSNA Knee Abnormality Detection
competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection),
generated from the free-text radiology reports (not the images) for all
4,407 training studies.

## Method

Each report was scored for all 12 findings in a single structured-output
call to Gemini (`gemini-3.1-flash-lite` / `gemini-3.5-flash-lite` /
`gemini-3.8-flash` / `gemini-3-flash-preview` / `gemini-3.7-flash`, batched
10 reports/call), using a prompt with explicit per-finding definitions
(e.g. distinguishing a meniscus tear from degeneration-without-tear, or a
bone contusion from a frank fracture). Full prompt, schema and scheduling
code: [`src/02_label_reports.py`](https://github.com/abhirick23/rsna-knee-abnormality-detection/blob/master/src/02_label_reports.py)
in the linked repo.

## Quality

Scored against the competition's 58 expert-labeled ("gold") studies:
**0.866 macro AUC**. Per-finding breakdown, comparison against other public
label sources, and a documented failed attempt to improve the four weakest
labels with a different prompt format (which made three of four *worse* —
see the linked repo) are in
[`results/measurement.md`](https://github.com/abhirick23/rsna-knee-abnormality-detection/blob/master/results/measurement.md).

| Finding | AUC vs gold-58 |
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

Synovitis is weak because it's frequently undocumented in the report text
even when present on imaging (confirmed: the worst misses never contain
"synovi-" anywhere in the source report) — a limit of any report-based
labeler, not specific to this prompt.

## Caveat

Several other public label tables for this competition score higher
(up to 0.893 macro AUC vs gold-58) — see the credits section of the linked
repo's README. This dataset is published as one input among several, not
as the strongest available option; it's useful mainly because the full
generation pipeline is open and reproducible.

## Columns

`StudyInstanceUID` plus one probability (0–1) column per finding:
`ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA, PF OA,
Effusion, Synovitis, Baker's, Contusion, Fracture`.

## Full pipeline

https://github.com/abhirick23/rsna-knee-abnormality-detection
