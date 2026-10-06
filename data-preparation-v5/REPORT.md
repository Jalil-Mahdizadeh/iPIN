# V5 data preparation report

**Complete: all final integrity checks passed.**

The original Bernett test is unchanged. Its 26,024 positive cases are shared with the custom ILP-negative test.

| Split | Positive pairs | Negative pairs | Unique sequences |
| --- | ---: | ---: | ---: |
| train | 350,382 | 350,382 | 13,110 |
| val | 82,871 | 82,871 | 6,023 |
| test-ilp | 26,024 | 26,024 | 2,948 |

The ILP test shares 1,154 original negatives; 24,870 negative pairs need new scores per historical model.

Both backbone exports use identical physical pairs and full sequences. Model retraining and benchmark inference were not run.

Solver outcomes (a time-limited feasible solution is not a proven optimum):

| Split | Solver status | Actual relative gap | Positive GO mean | Negative GO mean |
| --- | --- | ---: | ---: | ---: |
| train | kOptimal | 0.00336223 | 0.016458 | 0.016458 |
| val | kOptimal | 0.000437623 | 0.018583 | 0.018584 |
| test-ilp | user_limit | 0.029808 | 0.044095 | 0.044095 |

See [normalization exclusions](reports/normalization-audit.json), [homology exclusions](reports/test-homology-filter.json), [positive split](reports/positive-split.json), [final audit](reports/final-audit.json), and [hashed completion manifest](completed.json).

The author degree/GO objective and candidate pools are retained. TRAIN/DEV recovery uses an equivalent scaled formulation, verified feasible starts and saved incumbents. The completed ILP test and positive split were reused. Candidate generation additionally enforces the full known-positive family blacklist and non-self scope, and removes the upstream sorted-index truncation bias. Missing GO annotations remain explicit; selected negatives are not experimentally confirmed noninteractions.

The positive split retained 433,253 of 721,082 eligible positive pairs and reached its original time limit with a solver-reported gap of 49.41%; optimal retention was not established.

See [recovery details](recovery-v1/README.md) for solver termination, checkpointing and the unchanged-data contract.
