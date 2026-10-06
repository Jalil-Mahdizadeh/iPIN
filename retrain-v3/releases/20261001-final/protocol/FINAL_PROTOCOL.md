# V3: one final full-data experiment and closeout

The user authorized "agreed, finalize v3 now" on 2026-10-01, after rejecting further screening, controls and research branches. This protocol replaces the prospective multi-seed confirmation plan with **one** exploratory full-data residue-MLP run, its final benchmark and closure of v3 regardless of outcome.

## Decision and fixed training budget

Select residue-MLP at LR **2e-5**, seed **2**, based on the completed three-fold development comparison. Mean selected AP was 0.620374 versus 0.616488 for CLS-linear and 0.614789 for CLS-MLP. Its +0.003886 gain over linear **missed the original +0.005 gate**. The unchanged `decisions/readout-window5000.json` continues to record non-promotion. This single user-authorized closeout is an explicit exploratory override, not a claim that the old gate passed.

- Run: `clean-residue-mean-official-seed2`.
- Start from the pinned pretrained ESM2 650M, not a fold checkpoint or native supervised weights. Full backbone fine-tuning; residue-mean residual MLP, width 128; existing model implementation unchanged.
- Official train: **163,085** pairs (81,550 positive); validation: **59,258** (29,628 positive). No length cap; no added data or split changes. Trainer has no test loader.
- Clean, equally weighted BCE, classification scale 10, no masking or MLM. Both orientations, 64 physical pairs per optimizer update, four GPUs, FP32 parameters and AdamW state, BF16 autocast, efficient attention and TF32 off.
- Exactly **12,745 updates**, **2,000 warmup updates**, LR 2e-5, weight decay 0.01, gradient clip 1. This matches the existing full-data v2 clean-BCE training budget and sampling configuration, approximately five passes (815,680 pair exposures). The development folds' 6,000-update/1,000-warmup schedule is not resumed.
- Validate at 1,000 through 12,000 and 12,745: **13 opportunities**, identical to v2. Maximum full official-validation mean-AB/BA-logit AP selects the checkpoint; earliest wins an exact tie. All planned validations must complete before benchmarking. No test-based checkpoint choice or budget extension.

The exact qualified v3 readout model, data, optimizer and checkpoint core are reused. V3's `gradient_as_bucket_view=False` fix preserves bitwise four-GPU interrupted/resumed execution for the actual residue head; the separate v2 run used the older storage behavior. This is disclosed rather than treated as a new scientific control. Prior full-size resume checks remain valid for the identical core. A short forward-only check verifies the benchmark adapter and longest official validation pair (39,391 tokens); it trains no additional model.

## Fixed benchmark and stopping point

One SLURM allocation runs full training, freezes the validation-selected checkpoint, checks its saved validation logits, then performs **only its** official-test inference. Native plus all two v1 and four v2 predictions are reused after integrity and coverage checks. Eight models are compared on the same **52,048** pairs, all full length, same SIF, BF16 forward and mean AB/BA raw-logit scoring. The already verified v2 clean-BCE model is the main architectural comparator; native is the target baseline.

Primary endpoint: pooled test AP. Report AUROC and Brier alongside it, and paired protein-resampling AP differences versus native and v2 clean BCE, using the existing 1,000-replicate seed-20260929 bootstrap. Any thresholded metric uses validation-derived thresholds fixed before new test inference. Preserve all baselines and outcomes; no selection among test checkpoints. Baseline metric and bootstrap equality with v2 is checked numerically.

This historical test has informed previous research; the single-seed result is exploratory, not independent confirmation. A positive AP difference alone does not establish reliable superiority; descriptive intervals and limitations accompany it. **After this report, v3 ends whether it wins or loses.** No new LR/head screen, extra seed, external-data branch, native adaptation or rescue run is authorized by this protocol.

## Runtime and recovery

One node with four GPUs, 72 CPU cores, 400 GB RAM, 48-hour allocation. Atomic hashed complete checkpoints every 100 updates, 15 minutes, validation, and safe stop retain optimizer/RNG/sampler/selection/pending-validation state. Resume requires the same immutable release and four GPUs; existing exact-resume qualification is retained.

The final wrapper skips completed training during resumption. Inference commits row-verified 512-pair chunks and skips verified completed chunks. A time-limit request stops at a safe boundary and requeues this one job, up to eight restarts. `runs/clean-residue-mean-official-seed2/REQUEST_STOP` stops the pipeline safely; it never triggers benchmarking an unfinished training run. Remove that marker only for intentional continuation, then use `scripts/launch_final.py --submit`. Failures retain artifacts and cannot become a successful result.

The complete pipeline and its input hashes are frozen before submission. Output: `benchmark-v3/REPORT.md`, detailed machine-readable results and `retrain-v3/FINAL_RESULTS.md`. `retrain-v3/FINALIZED.json` is written only after the completed benchmark passes coverage, provenance and numerical checks.
