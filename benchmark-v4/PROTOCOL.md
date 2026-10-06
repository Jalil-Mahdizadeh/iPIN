# V4 ESMC benchmark protocol

Authorized on 2026-10-02: create benchmark-v4, evaluate ESMC test performance and compare it with native PLM-interact, v1, v2 and v3, reusing existing data where possible. This benchmark evaluates the completed **C0 ESMC 600M standard-attention** run only. S1 and C1 remain stopped. The originally nominated v4 primary native comparison was C1; this requested C0 comparison is descriptive and does not replace C1 or complete the factorial experiment.

## Selection before testing

C0 completed 12,745 updates, 815,680 physical-pair exposures and all 13 planned validations. Independently verify each saved 59,258-row validation array and recompute pooled AP, then select its strict maximum with the earlier checkpoint winning exact ties. This selects **update 7,000**, validation AP **0.6576301574729071**. Freeze its committed checkpoint hash and copied validation predictions before test inference. No checkpoint, threshold or numerical setting is selected using C0 test performance.

Use the original C0 SIF and byte-identical model and training-data code from `retrain-v4/releases/20261001-production`. Strictly load FP32 checkpoint parameters and use BF16 autocast, efficient attention, deterministic settings and TF32 disabled. Verify SDK/Torch/package versions against the training contract, all cached residue IDs against the ESMC tokenizer, and saved validation predictions against fresh forwards before test inference. Include the longest validation pair as an execution check. The training configuration and artifacts remain unchanged.

## Common inputs and reuse

All nine models cover exactly **52,048 test pairs**, with **26,024 positives**. Full sequences, both orientations and the mean of AB/BA raw logits are used throughout. No test sequence truncation, new negatives, model fitting or probability recalibration is performed. The same 59,258 validation pairs determine each model's maximum-F1 threshold, choosing the highest threshold on exact ties, before new C0 test inference.

Reuse all eight earlier models' test and validation predictions from benchmark-v3. Verify their hashes against its completed artifact manifest and check row IDs, labels and finite scores. Preserve the baseline checkpoint identities, old selection provenance, source summary and bootstrap replicates. Recompute statistics from these arrays and require every baseline AP/AUROC/Brier value and paired bootstrap replicate to reproduce benchmark-v3 within 1e-12. Native/ESM2 inference is not rerun in the ESMC container.

## Metrics and interpretation

Primary metric: pooled test AP. Also report AUROC, Brier, original-order scores, validation-derived operating points, orientation diagnostics and length strata at combined residue length 2,193. Reuse the 1,000-replicate paired protein bootstrap with seed 20260929: endpoint multiplicities multiply edge weights, with one multiplicity for self-pairs. Report all C0-minus-comparator differences and descriptive percentile 95% intervals.

Protein-macro AP is an additional diagnostic already requested in the v4 proposal: compute within-protein AP on incident test pairs, retain proteins with at least one positive and one negative pair, count self-pairs once, then average equally over the common eligible proteins. Report eligibility counts. It is not a selection metric or a substitute for pooled AP.

The proposal's practical numerical target was +0.010 absolute AP over native with a paired interval above zero. Report whether C0 satisfies those numbers descriptively, without relabeling it as the original C1 primary comparison. Historical test reuse, one seed, multiple comparisons and unmodeled homology/selection uncertainty limit inference. C0 has 13 validation opportunities and selected update 7,000; v3/S0 was stopped at 8,446 and selected update 4,000 from eight validations. Training-budget and selected-exposure differences must be disclosed. No further training or model search follows from this benchmark.

## Execution and completion

Preserve the four fixed logical shards, their stable length sorting, 16,384-token budget, maximum eight physical pairs per microbatch and atomic 512-row commits. Use the available interactive GH200 GPU sequentially because a four-GPU node has an estimated wait of over one hour. Record nonoverlapping shard times and worker identity. Reuse committed chunks after integrity checks on restart. Stop markers belong to benchmark-v4.

Completion requires unique full test-row coverage, matching labels, finite logits, unchanged checkpoint/source fingerprints, all nine model statistics, successful baseline reproduction, figures and a written report. `completed.json` certifies this C0 benchmark only; both stopped chain-aware arms remain recorded in the report. Predictions, analysis and reports stay in benchmark-v4.
