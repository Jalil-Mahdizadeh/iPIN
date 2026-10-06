# TRAIN-degree baseline on the exact released STRING v11 human data

Completed: 2026-10-05T21:13:05.449077+00:00

**The earlier AP 0.8360 / AUROC 0.9635 result was already measured on these V11 files. This standalone assessment reproduces it independently from both PLM-interact humanV11's sequence CSVs and TUnA human seed 47's identifier/sequence TSVs.**

Both sources have 421,792 TRAIN rows (38,344 positive / 383,448 negative) and 52,725 human validation rows (4,794 positive / 47,931 negative). Their unordered exact-sequence-pair/label multisets, including duplicate multiplicities, match exactly. These are two source-format checks of the same dataset, not independent biological replications.

The files called `human_test` supply human validation for these releases. No STRING v12/v12.5 data, v5 HIPPIE/ILP training data, or nonhuman labels enter this calculation.

| Exact V11 source | Human validation rows | AP | AUROC |
|---|---:|---:|---:|
| PLM-interact humanV11 released CSVs | 52,725 | 0.836014 | 0.963460 |
| TUnA human seed-47 released TSVs | 52,725 | 0.836014 | 0.963460 |

These are the counting baseline's scores on each source dataset. They are not predictions from the two neural models.

**Method.** For every exact amino-acid sequence, count its occurrences in positive and negative TRAIN rows. Self pairs count once toward their endpoint. The score for a validation pair A–B is:

```text
log((d_train_positive(A)+1)/(d_train_negative(A)+1)) + log((d_train_positive(B)+1)/(d_train_negative(B)+1))
```

This is equivalent in ranking to multiplying the two smoothed positive/negative count ratios. The +1 smoothing was fixed before this assessment. All released TRAIN rows, aliases resolved by exact sequence, label conflicts, and repeated rows are retained in the primary calculation. There is no sequence embedding, learned parameter, hyperparameter search, score-direction choice, or validation-label input to the score. Values are ranking scores, not calibrated interaction probabilities.

**Sensitivity to repeated TRAIN/validation pairs.** Remove every validation row whose unordered exact sequence pair appears anywhere in TRAIN, irrespective of label. TRAIN counts and the score rule remain fixed.

| Validation cohort, shared by both sources | Rows | Positives | Negatives | AP | AUROC |
|---|---:|---:|---:|---:|---:|
| All released rows | 52,725 | 4,794 | 47,931 | 0.836014 | 0.963460 |
| Excluding TRAIN-pair overlaps | 52,636 | 4,783 | 47,853 | 0.836078 | 0.963509 |

The exclusion removes 89 rows: 82 same-label matches and seven opposite-label matches. The nearly unchanged result shows that direct pair repetition is not the main explanation for this baseline's high validation score. Protein identities still overlap.

**Interpretation.** Every one of the 15,351 distinct validation sequences already appears in TRAIN. TRAIN contains 15,631 distinct sequences, of which only 7,492 appear in its positive rows. Approximately 77.44% of negative TRAIN rows contain an endpoint absent from TRAIN positives. The baseline uses this difference in protein-level label frequencies, without learning whether particular partners physically bind.

As a simpler diagnostic, counting how many endpoints appeared in TRAIN positives gives AP 0.294138 / AUROC 0.873854. The full-cohort positive prevalence, the reference for a noninformative ranking, is 0.090925.

This is evidence of a substantial sampling signal in the human validation task. It does not establish that a degree lookup generalizes to unseen proteins or explain the native model's five-species results by itself. This study evaluates the human validation split only. No neural model or production training job was run. Full released row multiplicity is retained; metrics are deterministic point estimates without confidence intervals.

**Verification.** Six lossless archived inputs were checked against the recorded release SHA-256 hashes. Native CSVs and TUnA TSVs were separately parsed and scored; the complete labeled score multisets agree. AP was independently recomputed using grouped precision/recall thresholds, and AUROC using the Mann–Whitney rank formula; both agree with sklearn within 1e-12. Results also reproduce the earlier V11 assessment within 1e-12.

**Files.** [Source provenance](provenance/inputs.json); [full results and checks](results/summary.json); [metrics CSV](results/metrics.csv); [per-sequence TRAIN degrees](results/train-sequence-degrees.csv.gz); [native validation scores](results/validation-scores-native.csv.gz); [TUnA validation scores](results/validation-scores-tuna.csv.gz). Each score table preserves its original source row order. Compressed NPZ score arrays are also provided. [Completion manifest](results/COMPLETE.json) records artifact hashes.

**Reproduction.** From the iPIN project root, run:

```bash
bash literature/string-v11-train-degree-baseline/scripts/run.sh
```

This uses the existing native PLM-interact ARM64 SIF as a CPU Python environment; GPU passthrough is disabled. All six required data files are archived under this study's `inputs/` directory, so reproduction does not require access to the sibling TUnA checkout. The provenance identifies the pinned public releases; it does not reconstruct any unavailable private checkpoint-training history.
