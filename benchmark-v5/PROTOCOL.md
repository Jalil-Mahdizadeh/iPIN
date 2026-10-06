# v5 frozen-checkpoint benchmark

Protocol fixed on 2026-10-05 before new test inference and scoring. See the hashed selection and data manifests in `provenance/` for executable identities.

## Models and selection

All models are frozen; no fitting or test-driven checkpoint selection occurs here. The two iPIN checkpoints were selected independently by maximum pooled AP on all 165,742 v5 ILP DEV pairs: ESM2 update 27,374 (AP 0.6153270385), ESMC update 21,899 (AP 0.5924616586). ESM2 is the DEV-preferred iPIN model. Both results will be reported regardless of their test ranking. The training folder was renamed from `retraining-v5` to `retrain-v5`; checkpoint bytes are verified against the frozen training manifests.

The eight entries are iPIN ESM2, iPIN ESMC, released native PLM-interact Bernett/Leakage-Free, released TUnA Bernett, X-PAIR interaction_bernett, X-PAIR multitask_xfair (default), released RAPPPID red-dreamy, and native SPRINT. Both X-PAIR releases are included separately at the user's request. SPRINT receives exactly the 350,382 v5 TRAIN-positive edges, also explicitly chosen by the user. DEV/test labels never enter its graph. The other competitors use released checkpoints, not newly fitted versions on v5 data.

## Tests and prediction reuse

Each frozen test contains 52,048 pairs (26,024 positives and 26,024 negatives). Original Bernett and custom ILP tests share all positives and 1,154 negatives; their union contains 76,918 unique pairs over 3,022 unique sequences. The ILP test is a custom negative-distribution sensitivity analysis, not a separate independent replication or the published Bernett-2026 test. No test pairs are omitted because of sequence length or scoring difficulty.

Score each unique pair once per predictor and map the score to each test. Reuse native PLM-interact's existing 52,048 original-test predictions only after checking sequence, pair, label, checkpoint, precision and code provenance. Compute its 24,870 missing ILP-negative scores. Reuse competitor sequence features only with verified sequence hashes, encoder/checkpoint identities and a fresh numerical check against native inference. Model comparisons use exactly the same rows within each test.

## Inference conventions

iPIN and native PLM-interact use full sequences, FP32 weights, BF16 inference, TF32 disabled, and the arithmetic mean of raw AB/BA logits; sigmoid supplies probabilities. Also report original-order diagnostics. This full-sequence symmetric native comparator is consistent with prior iPIN benchmarks and differs from paper settings with a length cap. No masking, model updates, or probability recalibration occurs.

TUnA uses the released Bernett classifier with full-length ESM2-150M representations, preserving its native uncertainty-adjusted score. X-PAIR uses full-length Ankh-Large representations and the native interaction head for both checkpoints. RAPPPID uses its released tokenizer and native leading-1,500-residue cap; quantify affected proteins and pairs, retain those pairs, and label this limitation. Optimized feature reuse and independent scoring must agree with native singleton inference. SPRINT uses native defaults (PAM120, Thit=15, Tsim=35, Thc=40), full TRAIN+TEST sequences for HSP construction, 64-thread HSP generation and serial scoring. SPRINT's sequence preprocessing is transductive, and its nonnegative unbounded scores are not probabilities.

## Analysis and limits

AP and AUROC are the primary metrics, reported separately on each test. Report Brier scores only for probability outputs, plus class, length, zero-score/tie and coverage diagnostics. Operating thresholds, if shown, come from frozen DEV or a labeled fixed 0.5 diagnostic, never from test optimization. Report protein-macro AP for proteins with at least ten incident test pairs and both labels, with eligibility counts and mean local positive prevalence. Count a self-pair only once in its protein's incident set.

Secondary subsets (defined before computing test metrics) are combined length at most/above 2,193 residues, pairs unaffected by RAPPPID's per-protein length cap, pairs absent from known X-PAIR default TRAIN/DEV, and pairs where neither endpoint appears in those data. Apply each subset identically to all eight models; include prevalence and counts. These are descriptive sensitivity analyses, not alternative test sets for choosing a winner.

Use 1,000 paired protein-endpoint bootstrap replicates (seed 20260929) for uncertainty and differences versus native PLM-interact. Draw proteins with replacement within the test; weight each non-self pair by the product of endpoint multiplicities and self-pairs once. These intervals describe this fixed test and trained instance, not training-seed or family-level generalization. Original and ILP test intervals are not independent. A native AP gain of at least 0.010 with a paired 95% interval above zero is a descriptive practical target inherited from the proposal, not a claim of preregistered confirmatory significance.

Audit known released training/validation exposure by exact sequence and pair where source data are available. Report unavailable provenance explicitly. The new iPIN models and SPRINT share v5 positive training evidence; released competitors may have different training data, exposure, objectives and resources. Do not attribute a score difference solely to architecture or call the whole comparison uniformly leakage-free. Preserve failed attempts, hashes, exact scripts and per-pair predictions.
