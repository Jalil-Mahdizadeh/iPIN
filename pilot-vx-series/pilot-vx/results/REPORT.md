# Bernett Vx pilot results

Assessment AP change: +0.02343. Status: no-go or inconclusive.

No TEST labels, TEST predictions, or released structural contact heads were used.

| Population | Model | AP | AUROC | Covered / rows |
|---|---|---:|---:|---:|
| assessment | baseline | 0.64093 | 0.64814 | 659 / 958 |
| assessment | true | 0.66436 | 0.65866 | 659 / 958 |
| assessment | shuffled | 0.66325 | 0.66107 | 659 / 958 |
| assessment | quality | 0.65094 | 0.65272 | 659 / 958 |
| fixed_dev_sample | baseline | 0.64835 | 0.64953 | 2628 / 4000 |
| fixed_dev_sample | true | 0.65745 | 0.65346 | 2628 / 4000 |
| fixed_dev_sample | shuffled | 0.65531 | 0.65304 | 2628 / 4000 |
| fixed_dev_sample | quality | 0.65345 | 0.65242 | 2628 / 4000 |
| calibration | baseline | 0.64277 | 0.65177 | 692 / 1068 |
| calibration | true | 0.65121 | 0.65491 | 692 / 1068 |
| calibration | shuffled | 0.65057 | 0.65273 | 692 / 1068 |
| calibration | quality | 0.64968 | 0.65704 | 692 / 1068 |

The fixed DEV sample has 4,000 rows; it is not the complete 59,260-row DEV dataset. Calibration results are selection results. All unsupported examples retain the native baseline.

See metrics.json for matched protein-bootstrap intervals, strata and fusion coefficients. Family breadth and production cost require review before any continuation.
