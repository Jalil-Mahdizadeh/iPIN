# V3 prospective training and decision protocol

Frozen before production on 2026-09-30. Source: [proposal snapshot](provenance/improvment-proposal-v3.md), SHA-256 `3e26b374d2f3d870f047fe19f472e01f55461e4f94d91827ed4fe6510bda8e86`.

## Hypotheses and scope

Clean BCE was the most promising v2 objective, but late training deteriorated and aggregate historical AP did not establish superiority to native. We will test (1) gentler adaptation under an otherwise matched clean objective and (2) added residue information beyond increased head capacity. V3 does not repeat masking/MLM or class-weight experiments in the initial array.

Stage 1 has six independently initialized fold runs: two learning rates, three fixed folds, seed 2. Stage 2, conditional on Stage 1 review, has six runs: two readout heads, three folds, at the selected learning rate. These are development trials, not independent seed replicates. The three folds have overlapping training populations, so a fold mean is not three independent experiments.

## Fixed training contract

| Item | Stage 1 / Stage 2 contract |
| --- | --- |
| Initialization | Pinned ESM2 t33 650M UR50D, revision `08e4846e537177426273712802403f7ba8261b6c`; never native/v1/v2 supervised weights |
| Pairs | Existing v2 internal folds; both validation endpoints and detected close-homology components excluded from the corresponding training fold |
| Objective | `10 × mean BCE(logit_AB, label; logit_BA, label)`; positive weight 1; no input masking, no MLM |
| Pair length | Full sequences; no training cap and no validation truncation |
| Exposure | 6,000 optimizer updates × 64 physical pairs = 384,000 pair exposures, two orientations each |
| Effective epochs | Fold 0: 5.210948; fold 1: 5.355798; fold 2: 5.334371 |
| Optimizer | AdamW, matrix weight decay .01, vector decay 0, gradient-norm cap 1 |
| Learning rate | Stage 1: 2e-5 versus 5e-6 applied to all trainable parameters; 1,000-update linear warmup, linear decay through the fixed horizon |
| Precision | FP32 parameters/optimizer; BF16 autocast; TF32 off; deterministic rank-ordered FP32 reduction |
| Compute | One node, four GPUs, fixed global batch; length-bucket microbatches, no dropped remainders |
| Validation | Clean full sequences, mean of AB/BA raw logits; every 1,000 updates and final update |
| Checkpoint selection | Maximum pooled validation AP; strict greater-than retains earliest update on an exact tie |
| Recovery | Existing byte-identical v2 trainer/model/data/state/release engine; independently qualified v3 invocation and SLURM wrapper |

The factor 10 is retained from v2 to preserve the matched optimization/gradient-clipping contract. Lowering the single learning rate also lowers the classifier rate; results cannot identify an encoder-only effect. The shorter internal-fold horizon still represents roughly five pair-sampling cycles, with fold-specific counts disclosed above.

## Readout controls

Stage 1 uses the native-shaped `Linear(ReLU(CLS))` classifier. Stage 2 compares zero-initialized residual additions:

- CLS-only: 512 nonlinear hidden units reshaped into four 128-unit blocks, averaged, then projected to one residual logit.
- Residue: per-chain masked means `mu_A`, `mu_B`, features `[CLS, mu_A + mu_B, abs(mu_A - mu_B), mu_A * mu_B]`, 128 nonlinear units, one residual logit.

Each adds exactly **655,489 parameters** at ESM2 hidden width 1,280. Output weights and bias start at zero, preserving baseline initialization logits. Means exclude special tokens and padding. Both models retain the same joint standard attention and AB/BA pooling. Symmetric residue features alone do not make contextual encodings exactly swap invariant. Residue access is not a validated interface/contact explanation.

## Promotion and selection

`scripts/compare_development.py` requires completed runs and all six scheduled validation outputs per run, verifies their hashes and label/row order, reconstructs checkpoint selection and refuses a comparison changing both learning rate and readout.

For every matched three-fold comparison, promotion requires all of:

1. Mean paired fold AP gain at least **0.005**.
2. Strictly positive AP gains on at least **two folds**.
3. No single-fold AP drop greater than **0.01**.
4. Mean AUROC drop no greater than **0.005**.
5. Mean macro AP drop no greater than **0.005**.

Protein macro AP averages incident-pair AP equally across proteins having **at least two positive and two negative pairs** in that validation population. Self-pairs count once. Shared edges and sampled partner lists make it neither independent protein evidence nor full-proteome partner-retrieval performance. Eligible proteins are identical between models on a fixed fold.

Stage 1 selects 5e-6 only if every gate passes against 2e-5. Otherwise 2e-5 remains the administrative baseline for the readout screen; that does not prove the lower rate is ineffective. Best-at-horizon and the final validation change are reported. A still-improving candidate is inconclusive at this horizon. No extension is automatic: changing the decay horizon changes the optimization trajectory and is rejected as exact continuation. Any further-horizon experiment needs a separately specified matched schedule, trial entry and budget; never edit a committed contract in place.

Stage 2 compares CLS-MLP versus CLS-linear, residue-MLP versus CLS-linear, and residue-MLP versus capacity-matched CLS-MLP. A residue candidate must pass both its linear and capacity controls to be selected by this implementation. If multiple candidates pass, select the largest mean AP gain versus CLS-linear; exact ties prefer CLS-MLP. If no readout passes but lower LR passed Stage 1, clean CLS at the lower LR remains eligible for confirmation. If neither mechanism passes, stop confirmation spending and revisit evidence/partner supervision.

The comparison also reports AP/AUROC/Brier, macro AP, length strata and a paired protein-endpoint multinomial bootstrap (1,000 replicates, seed 20260930). The bootstrap is descriptive and conditional on these graphs and selected checkpoints; it excludes seed and selection uncertainty. Require at least 95% usable replicates; a graph that fails this check needs investigation, not relaxed criteria. These gates are prospective resource-allocation rules, not significance tests. Retain all failed trials and selection opportunities.

## Evaluation access and claims

Internal folds are development data. Native and supervised v1/v2 checkpoints have seen the parent training labels, so they cannot initialize, teach or act as clean internal-fold baselines. The permitted initial state is ESM2's unsupervised pretrained checkpoint. Unsupervised sequence exposure, supervised interaction-label exposure and homology overlap are separate limitations.

Historical official validation is available for the optional practical/native branch and eventual full-data checkpoint selection, but is not the selector for Stage 1. The historical 52,048-pair test was already inspected during prior research; further results on it remain exploratory. **No historical test arrays, labels, predictions or inactive test-only sequence tokens are copied into this package.** The trainer accepts only train/validation splits, and the stage selector accepts only internal-fold runs.

Confirmation needs one frozen candidate and a matched control at seeds **2, 17 and 42**, alongside the fixed released native baseline on an exposure-audited panel. Report all seeds and their mean; never choose a seed using test AP. The aspirational AP gain over native is .01, with a positive conditional interval and stable seed behavior, not a performance prediction. New panels require new native inference; old scores are reusable only for an identical pair population and inference contract.

A separately reserved external development/test panel is **not yet curated**. [Its specification](external/README.md) is prepared, but this package does not claim independent confirmation is ready. Native continuation, teachers and native-preserving adapters require a different exposure contract and are disabled here. No automatic downloading of “new” database rows is treated as a clean external holdout.

## Execution boundaries

Only Stage 1 enters the initial immutable release. Stage 2 templates, the optional clean/capped full-data arm and later confirmation are not enabled. All production runs must use the pinned native SIF. The launcher's default operation is a hash-verifying dry run, and this preparation stops before its `--submit` step.

The optional clean/capped configuration retains v2's 12,745 updates, 2,000 warmup, seed 2, LR 2e-5 and global batch 64; its only changes relative to v2 capped are removal of corruption and MLM. Validation stays uncapped. Any result completes the historical factorial comparison; gains from capping and clean BCE must not be added to forecast its result.
