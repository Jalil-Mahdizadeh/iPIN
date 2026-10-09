# VZ query-only joint Pairformer ablation

Decision: **homolog_rows_add_practical_gain**.

One new Q head; original Vx true/shuffled/quality predictions reused unchanged. Same masks, eligibility, gate, and native PLM-interact. Assessment is exploratory; no TEST access.

| Assessment model | AP | AUROC |
|---|---:|---:|
| baseline | 0.640930 | 0.648144 |
| Q | 0.643861 | 0.651953 |
| shuffled | 0.663248 | 0.661071 |
| true | 0.664361 | 0.658658 |
| quality | 0.650942 | 0.652725 |

| Assessment AP contrast | Estimate | 95% matched protein interval |
|---|---:|---|
| Q_minus_shuffled | -0.019387 | [-0.035510, -0.000528] |
| Q_minus_true | -0.020500 | [-0.035194, -0.001869] |
| Q_minus_baseline | +0.002931 | [-0.004617, +0.010493] |
| Q_minus_quality | -0.007082 | [-0.020003, +0.011033] |
| true_minus_shuffled | +0.001113 | [-0.009639, +0.013333] |
| true_minus_baseline | +0.023431 | [+0.003601, +0.037539] |
| shuffled_minus_baseline | +0.022318 | [+0.000798, +0.038365] |
| true_minus_Q | +0.020500 | [+0.001869, +0.035194] |
| shuffled_minus_Q | +0.019387 | [+0.000528, +0.035510] |

Selected Q alpha: **0.5**. Historical true/shuffled/quality alphas remain 1. All transforms/head fitted on 1825 original eligible TRAIN pairs; assessment has 958 rows, 659 eligible.

Useful Q gain: False; recovery within 0.010 AP margin: False; eligible-only noninferiority: False; three-way equivalence: False; Q beyond quality/profile interval: False.

Noninferiority permits up to 0.010 AP loss (roughly 45% of the original shuffled gain). It is distinct from equivalence, which requires the whole interval inside ±0.010. True-minus-shuffled is unchanged; adding Q cannot strengthen its evidence.

Q removes homolog rows inside the encoder, while retaining MSA-derived masks and gate. Q success does not establish a fully MSA-free workflow or uniquely identify cross-chain biological signal. Q failure cannot separate loss of evolutionary context from input-depth effects.

See metrics.json for eligible-only and length/depth/coverage/family diagnostics, common-alpha and unfused-head results. No subgroup rescue, refitting, production continuation, or TEST evaluation is enabled.
