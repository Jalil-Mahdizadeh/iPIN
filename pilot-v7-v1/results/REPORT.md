# V7 v1 frozen contact-head pilot

Decision: **no_go_or_inconclusive**.

Primary: native + frozen Vx + true contact evidence versus native + frozen Vx. All new fitting and fusion selection used eligible TRAIN only. VAL assessment remains the original 958 rows.

| Assessment score | AP | AUROC |
|---|---:|---:|
| native | 0.640930 | 0.648144 |
| vx | 0.664361 | 0.658658 |
| vx_shuffled | 0.663248 | 0.661071 |
| vx_quality | 0.650942 | 0.652725 |
| quality_length | 0.640930 | 0.648144 |
| combined_quality | 0.664361 | 0.658658 |
| contact_true | 0.641509 | 0.648048 |
| combined_true | 0.664466 | 0.658623 |
| contact_shuffled | 0.641098 | 0.647926 |
| combined_shuffled | 0.664296 | 0.658475 |
| contact_coordinate_true | 0.641927 | 0.648148 |
| combined_coordinate_true | 0.664367 | 0.658593 |
| contact_coordinate_shuffled | 0.640930 | 0.648144 |
| combined_coordinate_shuffled | 0.663955 | 0.658392 |
| contact_boundary_true | 0.641540 | 0.648100 |
| combined_boundary_true | 0.664399 | 0.658610 |
| contact_boundary_shuffled | 0.640930 | 0.648144 |
| combined_boundary_shuffled | 0.664361 | 0.658658 |
| combined_true_on_shuffled | 0.664457 | 0.658684 |
| combined_true_on_coordinate | 0.664517 | 0.658732 |
| combined_true_on_boundary | 0.664477 | 0.658619 |

| Assessment AP contrast | Estimate | 95% matched protein interval |
|---|---:|---|
| combined_true_minus_vx | +0.000105 | [-0.000627, +0.001011] |
| combined_true_minus_combined_quality | +0.000105 | [-0.000627, +0.001011] |
| combined_true_minus_native | +0.023536 | [+0.004160, +0.039150] |
| contact_true_minus_vx | -0.022852 | [-0.039499, -0.004067] |
| contact_true_minus_native | +0.000579 | [-0.001863, +0.003014] |
| contact_true_minus_quality | +0.000579 | [-0.001863, +0.003014] |
| combined_true_minus_shuffled | +0.000169 | [-0.000431, +0.000895] |
| combined_same_head_pairing_gap | +0.000008 | [-0.000470, +0.000623] |
| combined_true_minus_coordinate | +0.000098 | [-0.000374, +0.000894] |
| combined_same_head_coordinate_gap | -0.000051 | [-0.000820, +0.001009] |
| combined_boundary_minus_vx | +0.000038 | [-0.000667, +0.000819] |
| combined_fixed_head_boundary_minus_vx | +0.000116 | [-0.000625, +0.001100] |

The quality comparison adds the quality/length branch to the same frozen Vx incumbent. Coordinate and boundary variants are prespecified diagnostics, not alternate models selected on VAL.

The source contact-head training membership remains unresolved; no exposure-free or independently blind generalization claim is supported. Family witnesses are incomplete. Contact predictions are not calibrated human PPI probabilities.

Unavailable pairs retain exact native scores. Computational failures invalidate the run instead of becoming biological fallback. No TEST access or automatic continuation.
