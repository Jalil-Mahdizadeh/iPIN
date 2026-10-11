# Vx-128 full TEST benchmark

Authorized 11 October 2026: evaluate the frozen `retrain-vx` model on all **52,048 original Bernett TEST pairs**, using **one exclusive four-GPU node**. The user explicitly requested this evaluation after the TRAIN/VAL run missed its predefined advancement criteria. That earlier finding is preserved; TEST is not used to repair or select the model. This TEST set has historical iPIN benchmark results, so it is not a newly blind dataset.

## Frozen comparison

- Native PLM-interact versus native + true Vx-128 and native + quality/profile. Reuse the two verified TRAIN-only heads from `retrain-vx/results/heads.json`, including every scaler/PCA/classifier parameter and both VAL-selected fusion weights (**alpha = 1**). No fitting, calibration, threshold selection, model selection, or hyperparameter search on TEST. No shuffled arm.
- Preserve original shared-accession matching, monomer cap 4,096, at most 127 homologs + query, joint redundancy/taxonomy rules, masks, minimum eight homologs, minimum 50% quality coverage per chain, and combined length limit 1,536. Same frozen Pairformer layer 15, fixed projection, 128 global summaries, AB/BA averaging and reliability gate. Quality/profile uses the original 68 features and requires no additional GPU forwards.
- Build TEST monomer caches from the same checksum-pinned omicMSA archive, using the unchanged parser. Exclude exact ungapped TRAIN/VAL query sequences from TEST homolog context, following the existing opposite-split exclusion convention. No new homology filter or alignment rescue. Preserve every original TEST row, including unavailable pairs.
- Reuse all 52,048 checksum-verified native TEST predictions from `benchmark-v1`: full sequences, FP32 parameters, BF16 autocast, TF32 off, mean AB/BA logits. Any missing/corrupt baseline row blocks the run. Data preparation verifies original CSV sequences, labels, row IDs, token arrays and prediction provenance without calculating new performance results.
- Feature workers receive label-free row manifests. An unavailable biological input retains the exact native score. Corrupt files, missing chunks, CUDA errors or incomplete worker coverage abort evaluation; they never become fallback or cause rows to be dropped.

## Reporting and interpretation

Primary population: **all 52,048 TEST pairs**, including native fallback. Report AP and AUROC for all three methods and paired AP differences with the same 1,000-replicate protein Poisson bootstrap (endpoint-product weights, seed 20261007, fixed fitted predictions). These intervals do not include model-training or research-selection uncertainty.

Keep the practical effect-size criteria from TRAIN/VAL for interpretation: true − native AP at least +0.010 with a positive 95% lower bound, AUROC loss no worse than 0.005, and a positive AP interval over quality/profile. Report every result regardless of whether these criteria pass. No pairing-specific or co-evolution claim is supported by this unshuffled benchmark.

Secondary descriptive strata are fixed before extraction: eligible/fallback, combined lengths ≤512/513–1024/1025–1536/1537–2048/>2048, depth 0/1–32/33–64/65–127/128, and eligible minimum quality-mask fractions [0.5,0.75)/[0.75,0.9)/[0.9,1]. Report availability reasons and depth/Neff summaries. Strata never select a new model or a preferred TEST population.

## Execution and safeguards

Preparation and GPU qualification use the existing one-GPU interactive allocation, bounded at four hours; it is not altered or cancelled. The main batch requests one exclusive node, four GPUs, four independent workers and 288 CPUs. Fixed length-balanced shards cover every row exactly once. Only compact features and provenance are stored; full pair tensors are discarded.

Main ceiling: **24 hours / 96 allocated GPU-hours / 6,912 allocated CPU-core-hours**, including idle analysis phases; reserve two hours for scoring, independent verification and shutdown. Additional storage ceiling: 40 GB. Queue time is not charged. A label-free conservative forecast must fit before submission. No automatic retry, requeue, partial-cohort evaluation or second submission.

Before submission, check the completed TRAIN/VAL artifact hashes, CPU/integrity tests, original feature reproduction on fixed TRAIN/VAL examples, and new TEST input repeat/symmetry checks without outcome evaluation. Freeze code, data, source heads, qualification and input hashes. At job start verify the SIF and exactly four distinct GH200 GPUs on one node. After complete extraction, apply the frozen heads once and independently reconstruct predictions and protein-bootstrap intervals, check native fallback and all native source predictions, and replay four deterministic source inputs per reason/length/depth bin. Source replay is a prespecified sample, not a claim that every new alignment is independently rebuilt.

The run ends after the verified TEST report. No automatic retraining or follow-up benchmark. See [STATUS.md](STATUS.md).
