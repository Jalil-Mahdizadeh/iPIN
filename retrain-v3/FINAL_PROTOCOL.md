# V3: one final full-data retraining run

The user authorized one final residue-MLP run on 2026-10-01 and then clarified that this folder is for **retraining only**. Benchmarking will be conducted later in a separate `benchmark-v3` folder. This stage ends after training and validation-based checkpoint selection. No test inference or automatic benchmark is launched here.

## Decision and fixed training budget

Select residue-MLP at LR **2e-5**, seed **2**, based on the completed three-fold development comparison. Mean selected AP was 0.620374 versus 0.616488 for CLS-linear and 0.614789 for CLS-MLP. Its +0.003886 gain over linear **missed the original +0.005 gate**. The unchanged `decisions/readout-window5000.json` continues to record non-promotion. This single user-authorized closeout is an explicit exploratory override, not a claim that the old gate passed.

- Run: `clean-residue-mean-official-seed2`.
- Start from the pinned pretrained ESM2 650M, not a fold checkpoint or native supervised weights. Full backbone fine-tuning; residue-mean residual MLP, width 128; existing model implementation unchanged.
- Official train: **163,085** pairs (81,550 positive); validation: **59,258** (29,628 positive). No length cap; no added data or split changes. Trainer has no test loader.
- Clean, equally weighted BCE, classification scale 10, no masking or MLM. Both orientations, 64 physical pairs per optimizer update, four GPUs, FP32 parameters and AdamW state, BF16 autocast, efficient attention and TF32 off.
- Exactly **12,745 updates**, **2,000 warmup updates**, LR 2e-5, weight decay 0.01, gradient clip 1. This matches the existing full-data v2 clean-BCE training budget and sampling configuration, approximately five passes (815,680 pair exposures). The development folds' 6,000-update/1,000-warmup schedule is not resumed.
- Validate at 1,000 through 12,000 and 12,745: **13 opportunities**, identical to v2. Maximum full official-validation mean-AB/BA-logit AP selects the checkpoint; earliest wins an exact tie. All planned validations must complete before benchmarking. No test-based checkpoint choice or budget extension.

The exact qualified v3 readout model, data, optimizer and checkpoint core are reused. V3's `gradient_as_bucket_view=False` fix preserves bitwise four-GPU interrupted/resumed execution for the actual residue head; the separate v2 run used the older storage behavior. This is disclosed rather than treated as a new scientific control. Prior full-size resume checks remain valid for the identical core. A short forward-only check verifies the benchmark adapter and longest official validation pair (39,391 tokens); it trains no additional model.

## Training completion and recovery

One node with four GPUs, 72 CPU cores, 400 GB RAM and a 48-hour limit trains the single model. At the fixed horizon, `runs/clean-residue-mean-official-seed2/completed.json` records training completion; `best.json` identifies the best official-validation checkpoint. The job then exits. No benchmark dependencies or test-reading commands are present in the active job wrapper.

Atomic hashed checkpoints every 100 updates, 15 minutes, validation and safe stop retain model, optimizer, per-rank RNG, sampler, selection and pending-validation state. Resume requires the unchanged configuration/core and four GPUs. Time-limit signals save and requeue this one training job, up to eight restarts. `REQUEST_STOP` in the run directory stops safely. For intentional continuation, remove that marker and run `python retrain-v3/scripts/launch_final.py --submit` from the workspace.

## Correction of the initial handoff

The initial job 3206387 included an automatic post-training benchmark. Following the user's clarification, it was stopped safely at update 26 before any test inference. Its full training checkpoint is retained. The replacement `20261001-final-training` release has byte-identical training configuration and all five core modules, so continuation preserves the existing optimizer and sampling trajectory. Only the job wrapper and launch/protocol metadata change. The old frozen release and submission remain historical provenance; `releases/CURRENT_FINAL` selects the training-only replacement.

The separate `benchmark-v3` directory contains prepared code and verified cached baseline predictions, but is inactive. Its preparation does not mean that the final model has been tested. No v3 test results or improvement claim exist yet. Benchmarking will be a later task.
