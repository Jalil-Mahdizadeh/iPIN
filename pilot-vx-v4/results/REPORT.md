# Vx v4 independent homolog sampling

Decision: **no_go_or_inconclusive**.

One new I head; original Vx true/shuffled/quality, VZ Q and Vx v3 C predictions reused unchanged. Same masks, eligibility, depth, gate, and native PLM-interact. Assessment is exploratory; no TEST access. R is deferred.

| Assessment model | AP | AUROC |
|---|---:|---:|
| baseline | 0.640930 | 0.648144 |
| Q | 0.643861 | 0.651953 |
| C | 0.650079 | 0.651573 |
| I | 0.649347 | 0.648327 |
| shuffled | 0.663248 | 0.661071 |
| true | 0.664361 | 0.658658 |
| quality | 0.650942 | 0.652725 |

| Assessment AP contrast | Estimate | 95% matched protein interval |
|---|---:|---|
| I_minus_shuffled | -0.013901 | [-0.029273, +0.004231] |
| I_minus_true | -0.015014 | [-0.029861, -0.000989] |
| I_minus_Q | +0.005486 | [-0.011841, +0.018787] |
| I_minus_baseline | +0.008417 | [-0.009543, +0.021901] |
| I_minus_quality | -0.001595 | [-0.019616, +0.013809] |
| true_minus_I | +0.015014 | [+0.000989, +0.029861] |
| shuffled_minus_I | +0.013901 | [-0.004231, +0.029273] |
| true_minus_shuffled | +0.001113 | [-0.009639, +0.013333] |
| true_minus_baseline | +0.023431 | [+0.003601, +0.037539] |
| shuffled_minus_baseline | +0.022318 | [+0.000798, +0.038365] |
| Q_minus_shuffled | -0.019387 | [-0.035510, -0.000528] |
| Q_minus_true | -0.020500 | [-0.035194, -0.001869] |
| Q_minus_baseline | +0.002931 | [-0.004617, +0.010493] |
| Q_minus_quality | -0.007082 | [-0.020003, +0.011033] |
| true_minus_Q | +0.020500 | [+0.001869, +0.035194] |
| shuffled_minus_Q | +0.019387 | [+0.000528, +0.035510] |
| C_minus_shuffled | -0.013168 | [-0.029227, +0.009282] |
| C_minus_true | -0.014282 | [-0.031036, +0.004880] |
| C_minus_Q | +0.006219 | [-0.007665, +0.019111] |
| C_minus_baseline | +0.009149 | [-0.005773, +0.021985] |
| C_minus_quality | -0.000863 | [-0.016199, +0.017254] |
| true_minus_C | +0.014282 | [-0.004880, +0.031036] |
| shuffled_minus_C | +0.013168 | [-0.009282, +0.029227] |
| I_minus_C | -0.000732 | [-0.015472, +0.012257] |

Selected I alpha: **1**. Historical Q alpha remains 0.5; true/shuffled/C/quality alphas remain 1. All I transforms/head fitted on 1825 original eligible TRAIN pairs; assessment has 958 rows, 659 eligible.

Useful I gain: False; recovery within 0.010 AP margin: False; recovery beyond Q: False; eligible-only noninferiority: False; three-way equivalence: False; I beyond quality/profile interval: False.

Noninferiority permits up to 0.010 AP loss (roughly 45% of the original shuffled gain). It is distinct from equivalence, which requires the whole interval inside ±0.010. True-minus-shuffled is unchanged; adding I cannot strengthen its evidence.

I preserves query, original depth/masks/gate and intact homolog sequences. It selects A/B independently without the accession intersection. Profiles, diversity, coverage and matching change together; recovery supports practical sufficiency, not a unique biological mechanism. Additional availability is counted separately without new predictions. R remains deferred.

See input-diagnostics.json for selection/diversity/coverage diagnostics and coverage.json for label-free potential availability, and metrics.json for eligible-only and length/depth/coverage/family diagnostics, common-alpha and unfused-head results. No subgroup rescue, refitting, production continuation, or TEST evaluation is enabled.
