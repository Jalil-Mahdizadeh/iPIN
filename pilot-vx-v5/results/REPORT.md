# Vx v5: matched MSA depth 256 versus 128

Decision: **material_depth_gain_not_supported_within_interval**.

Same original TRAIN/DEV cohort, masks, gates and readout. Depth includes the query. Historical depth-128 predictions reused unchanged. Exploratory assessment; no TEST access.

| Assessment arm | AP | AUROC |
|---|---:|---:|
| baseline | 0.640930 | 0.648144 |
| true | 0.664361 | 0.658658 |
| T256 | 0.659059 | 0.656865 |
| shuffled | 0.663248 | 0.661071 |
| S256 | 0.659753 | 0.660456 |
| quality | 0.650942 | 0.652725 |
| P256 | 0.650997 | 0.652769 |

| AP contrast | Estimate | 95% matched protein interval |
|---|---:|---|
| T256_minus_true | -0.005302 | [-0.018242, +0.007690] |
| S256_minus_shuffled | -0.003495 | [-0.015248, +0.010497] |
| P256_minus_quality | +0.000054 | [-0.001002, +0.001172] |
| T256_minus_S256 | -0.000694 | [-0.012211, +0.010797] |
| depth_pairing_interaction | -0.001807 | [-0.017161, +0.013390] |
| T256_minus_baseline | +0.018129 | [-0.000613, +0.033995] |
| S256_minus_baseline | +0.018823 | [+0.001436, +0.032559] |
| P256_minus_baseline | +0.010067 | [-0.008755, +0.023179] |
| T256_minus_P256 | +0.008062 | [-0.010654, +0.026718] |
| S256_minus_P256 | +0.008756 | [-0.008641, +0.025433] |
| true_minus_baseline | +0.023431 | [+0.003601, +0.037539] |
| shuffled_minus_baseline | +0.022318 | [+0.000798, +0.038365] |
| true_minus_shuffled | +0.001113 | [-0.009639, +0.013333] |
| quality_minus_baseline | +0.010012 | [-0.008808, +0.022976] |

New fusion alphas: {'T256': 1, 'S256': 1, 'P256': 1}. TRAIN-only fits on 1825 pairs; assessment 958 rows, 659 eligible.

Primary material benefit requires at least +0.010 AP with positive lower interval and AUROC loss no greater than 0.005 versus true128. An interval upper bound below +0.010 argues against that material benefit, not against every smaller gain. Non-significance is not equivalence.

True256 extends the exact true128 prefix using the unchanged accession matcher. Shuffled256 applies the original shuffler at its larger depth, so its old-prefix permutation can change. Naturally shallow inputs reuse verified original features; all three new heads are fitted once.

See depth-coverage.json for actual depth counts, input-diagnostics.json for diversity/coverage, qualification/real.json for same-pair timing at both depths, and metrics.json for length/depth/family strata and common-alpha diagnostics. No automatic full benchmark or TEST evaluation.
