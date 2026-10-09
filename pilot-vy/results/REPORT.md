# Original Bernett VY pilot

Decision: **no_go_or_inconclusive**.

Independent monomer-MSA context, frozen native PLM-interact, and five small TRAIN-only heads. The reused assessment is exploratory. No TEST input or evaluation.

Available pairs: TRAIN 2486/4000; DEV 2957/4000; assessment 727/958. Required monomers: 3801; distinct TRAIN proteins fitted once per representation: 1881.

| Assessment model | AP | AUROC |
|---|---:|---:|
| baseline | 0.640930 | 0.648144 |
| vx_true | 0.664361 | 0.658658 |
| vx_shuffled | 0.663248 | 0.661071 |
| vx_quality | 0.650942 | 0.652725 |
| M | 0.640930 | 0.648144 |
| S | 0.640930 | 0.648144 |
| P | 0.644859 | 0.653294 |
| SP | 0.645863 | 0.653290 |
| U | 0.640930 | 0.648144 |
| M_on_S | 0.640930 | 0.648144 |

| Assessment AP contrast | Estimate | 95% matched protein interval |
|---|---:|---|
| M_minus_SP | -0.004933 | [-0.017920, +0.008214] |
| M_minus_baseline | +0.000000 | [+0.000000, +0.000000] |
| M_minus_S | +0.000000 | [+0.000000, +0.000000] |
| M_minus_P | -0.003929 | [-0.020598, +0.013107] |
| M_minus_U | +0.000000 | [+0.000000, +0.000000] |
| M_minus_vx_true | -0.023431 | [-0.040086, -0.004592] |
| M_minus_vx_shuffled | -0.022318 | [-0.038812, -0.001369] |
| M_minus_vx_quality | -0.010012 | [-0.022874, +0.008207] |
| M_same_head_context | +0.000000 | [+0.000000, +0.000000] |
| SP_minus_baseline | +0.004933 | [-0.008214, +0.017920] |
| P_minus_baseline | +0.003929 | [-0.013107, +0.020598] |
| S_minus_baseline | +0.000000 | [+0.000000, +0.000000] |
| U_minus_baseline | +0.000000 | [+0.000000, +0.000000] |

The primary contrast is M minus S+P: learned MSA context versus the combined query-only and profile control. All VY arms share availability and reliability. SP has 97 supervised parameters, M/S/P 49 each, U 17.

M = monomer-MSA; S = query only through the same encoder; P = fixed profile/quality; SP = concatenated S/P pair descriptors; U = additive unary MSA; M_on_S = M head evaluated on query-only descriptors without refitting.

Selected alpha values: `{'M': 0, 'S': 0, 'P': 1, 'SP': 0.5, 'U': 0}`. Native TRAIN logits were not used. Historical Vx predictions retain their original gate and heads and are descriptive references.

See metrics.json for fixed full DEV, coverage groups, length/diversity/quality strata, common-alpha sensitivity, evidence-only metrics and protein/family influence. Missing complete-family annotations limit generalization claims. A monomer-context benefit would not establish inter-protein co-evolution. No automatic continuation is enabled.
