# Vx-128 full TRAIN/VAL results

Decision: **no_go_or_inconclusive**.

Original depth-128 matched MSA, frozen encoders, two TRAIN-only heads. No shuffled encoding or TEST evaluation.

| VAL population | Rows | Eligible | Native AP | True128 AP | Quality AP |
|---|---:|---:|---:|---:|---:|
| assessment_new_rows | 13055 | 8731 | 0.647962 | 0.654981 | 0.651903 |
| assessment_all | 14013 | 9390 | 0.647211 | 0.655511 | 0.651661 |
| calibration | 15614 | 10018 | 0.635886 | 0.642647 | 0.637908 |
| crossing | 29633 | 19351 | 0.642747 | 0.651996 | 0.644994 |
| full_val | 59260 | 38759 | 0.642159 | 0.650538 | 0.644854 |
| pilot_val | 4000 | 2628 | 0.648354 | 0.658882 | 0.652053 |

Primary: previously unevaluated rows within the fixed assessment components. These can share proteins/families with the pilot and are not a newly blind dataset.

| Primary AP contrast | Estimate | 95% protein interval |
|---|---:|---|
| true_minus_baseline | +0.007019 | [+0.002428, +0.012237] |
| true_minus_quality | +0.003078 | [-0.001289, +0.007732] |
| quality_minus_baseline | +0.003941 | [+0.000939, +0.007153] |

Eligible TRAIN fit rows: 74576. Fusion alphas: {'true': 1, 'quality': 1}.

Full VAL and pilot-row results are descriptive. Calibration selects alpha; crossing pairs do not. All unavailable pairs retain exact native scores. Complete numerical and source verification is required before completion.

The predictive criterion is >=0.010 primary AP gain over native with a positive interval, AUROC loss <=0.005 and positive separation from quality/profile. No pairing-specific biological conclusion follows. TEST is not authorized by this run.
