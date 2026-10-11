# Vx-128 full TEST benchmark

Fixed practical criteria: **no_go_or_inconclusive**.

All 52,048 original TEST pairs are included; 32,340 receive MSA evidence. Unavailable pairs retain exact native scores.

| Frozen model | AP | AUROC |
|---|---:|---:|
| Native PLM-interact | 0.690319 | 0.699467 |
| Native + true Vx-128 | 0.698620 | 0.703791 |
| Native + quality/profile | 0.692603 | 0.702421 |

| Full TEST AP contrast | Estimate | 95% protein interval |
|---|---:|---|
| true_minus_baseline | +0.008302 | [+0.005190, +0.011384] |
| true_minus_quality | +0.006017 | [+0.002012, +0.009944] |
| quality_minus_baseline | +0.002284 | [-0.001123, +0.005430] |

Both encoders and both TRAIN-fitted heads are frozen. VAL-selected fusion weights are true = 1 and quality = 1. No TEST fitting, calibration, model selection, or shuffled encoding.

This evaluation was explicitly authorized after TRAIN/VAL missed the advancement criteria. The earlier no-go finding is unchanged. Original Bernett TEST has historical iPIN results; this is not a newly blind evaluation. Intervals condition on the fitted models and omit research-selection uncertainty.

Practical criteria: at least +0.010 AP over native with a positive 95% lower bound, AUROC loss at most 0.005, and a positive AP interval over quality/profile. This benchmark cannot establish pairing-specific co-evolution. Fixed strata, coverage reasons and depth/Neff summaries are in metrics.json. No outcome-dependent follow-up is automatic.

Independent verification and COMPLETE.json are required for accepted completion.
