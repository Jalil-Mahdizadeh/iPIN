# Vx v2 local-pooling ablation

Decision: **no_go_or_inconclusive**.

Fixed 4,000 TRAIN and 4,000 DEV pairs; original eligibility and baseline fallback preserved. No TEST access. Assessment was previously inspected, so this is exploratory.

| Assessment model | AP | AUROC |
|---|---:|---:|
| baseline | 0.640930 | 0.648144 |
| global_true | 0.664361 | 0.658658 |
| global_shuffled | 0.663248 | 0.661071 |
| quality | 0.650942 | 0.652725 |
| local_true | 0.655735 | 0.655391 |
| local_shuffled | 0.662517 | 0.659400 |
| scrambled_true | 0.651501 | 0.653388 |
| scrambled_shuffled | 0.657550 | 0.656363 |
| local_true_on_shuffled | 0.651379 | 0.654326 |
| scrambled_true_on_shuffled | 0.649087 | 0.653410 |

| Assessment AP contrast | Estimate | 95% matched protein interval |
|---|---:|---|
| pairing_gap_improvement | -0.007896 | [-0.023482, +0.007892] |
| local_minus_baseline | +0.014805 | [+0.000271, +0.026339] |
| local_minus_global_true | -0.008626 | [-0.021953, +0.004153] |
| local_minus_shuffled | -0.006783 | [-0.017395, +0.003384] |
| local_minus_quality | +0.004792 | [-0.011752, +0.022928] |
| locality_pairing_gap | -0.000734 | [-0.011045, +0.009565] |
| local_minus_spatially_scrambled_true | +0.004234 | [-0.005185, +0.012773] |
| same_head_pairing_gap | +0.004356 | [-0.001152, +0.010382] |
| spatially_scrambled_same_head_pairing_gap | +0.002414 | [-0.001766, +0.006325] |

The primary contrast is (local true − local shuffled) − (original global true − original global shuffled). A larger gap produced only by weakening the shuffled model is insufficient.

Selected fusion coefficients: `{"local_shuffled": 1, "local_true": 0.5, "scrambled_shuffled": 1, "scrambled_true": 0.5}`. All head transforms were fitted on the same 1,825 eligible TRAIN examples; alpha selection used calibration only.

See metrics.json for full fixed DEV results, common-prevalence AP, length/depth/coverage strata, diagnostic cutpoints, fixed-head counterfactuals and protein/family-witness influence.

Complete family annotations were unavailable within TRAIN/DEV-only inputs. Witness sensitivity is not complete-family independence. Pairing sensitivity would not by itself establish direct compensatory coevolution. No automatic continuation is enabled.
