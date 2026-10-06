# Prepared v3 benchmark — deferred

The user deferred benchmarking until a later task. This folder is prepared but inactive. The [active training protocol](../retrain-v3/FINAL_PROTOCOL.md) ends after retraining, with no automatic handoff. The prospective comparison below is retained for later use.

Train one residue-MLP, seed 2, LR 2e-5 for exactly 12,745 updates. After full completion, select the maximum official-validation pooled AP among 13 planned validations; earlier wins an exact tie. Freeze checkpoint and prediction hashes before new test inference. A forward check verifies the selected checkpoint against saved validation logits. The comparison cannot run on an unfinished training run or a fold checkpoint.

Reuse checksummed native, two v1 and four v2 prediction arrays. Infer only the one v3 checkpoint on all 52,048 official test pairs. Retain full sequences, both orientations, mean raw-logit scoring, FP32 weights, BF16 autocast, efficient attention and TF32 disabled. No model fitting or calibration on test. Decision thresholds use official validation only and remain immutable across restarts.

Primary: pooled test AP. Also report AUROC, Brier, orientation/length diagnostics and 1,000 paired protein-bootstrap replicates, seed 20260929, matching benchmark-v2. Native and v2 clean BCE are the main comparators. Baseline metric and bootstrap outputs must reproduce v2 to 1e-12. Report descriptive intervals; the historical test and one seed cannot establish independent generalization.

Inference writes atomic checksummed 512-row chunks across four fixed shards. Resumption verifies and reuses committed chunks. Completion requires exactly one finite prediction per test row, label agreement, distinct GPUs, strict checkpoint loading, fixed-source integrity and completed statistical outputs. The report and `completed.json` are generated only after these checks.

When benchmarking is separately started later, the prepared pipeline can write `REPORT.md`, `results/benchmark-summary.json`, and `../retrain-v3/FINAL_RESULTS.md` and `FINALIZED.json`. **V3 ends after this comparison, whether performance improves or not.** No additional model, seed, hyperparameter or data search follows automatically.
