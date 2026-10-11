# V7 v1: pretrained Cβ-contact features

Implements [proposal V7](../improvment-proposal-v7.md) as a bounded, exploratory TRAIN/VAL pilot. No TEST inputs, new contact predictor, encoder training, parameter sweep, or automatic production continuation.

## Frozen scientific contract

- Reuse exactly the original 4,000 TRAIN / 4,000 VAL samples, with 1,825 / 2,628 eligible pairs. Preserve original depth 128, accession selection, taxonomic shuffle, masks, length cap, gate and baseline fallback.
- Use the existing pinned image and frozen layer-15 Cβ head. Apply the released head to the directed pair tensor before symmetrization/compression; align and average the two chain-order probability maps. LM/ConFind heads and layers after layer 15 are forbidden by runtime hooks.
- Every encoded pair must reproduce both original Vx true/shuffled 128-feature vectors exactly. Verify paired/null token bytes against Vx-v2. Numerical/computational failure stops the run; it is never converted into biological unavailability.
- Six features: top-20 mean, fraction ≥0.5, minimum and maximum chain participation at ≥0.5, mean normalized marginal entropy, and maximum supported 5×5 mean. A window requires at least 20/25 valid cells. Use original coordinates and valid cells only. For chains shorter than five or no supported windows, window score is zero with an explicit diagnostic. Empty diagnostic maps produce six zeros without changing eligibility. Entropy is zero for zero mass or a single eligible residue.
- Coordinate controls permute only valid positions within each chain using fixed UID/protein seeds. The first five features must be invariant. Boundary controls exclude the union of the A-end/B-start and A-start/B-end 5×5 corners, accounting for both encoding orientations. The pretrained model itself is unchanged.

## TRAIN-only fitting and calibration

Fit StandardScaler + logistic regression, fixed C=0.1 and 1,000 iterations, on eligible TRAIN only. Contact heads have six inputs and no PCA. The quality/length control uses the original 68 profile features plus eight symmetric nuisance statistics: log minimum/maximum length, log homolog depth, log Neff80, log valid map size, minimum query quality, effective joint support and log length asymmetry; it retains the original PCA32 readout convention.

Five-fold StratifiedGroupKFold keeps identical unordered query-sequence pairs in one fold. All trainable-head fusion inputs are out-of-fold. These folds are **not protein/family-disjoint** and provide no independent generalization claim. Native PLM-interact remains frozen; its TRAIN logits are in-sample for that pretrained model. Generate native scores only for the 1,825 eligible TRAIN calibration pairs, after reproducing three original native VAL microbatches for numerical qualification.

For each contact/control branch, select alpha from `[0, 0.1, 0.25, 0.5, 1]` by eligible TRAIN OOF AP, taking the smallest tie. Fit the final small head on all eligible TRAIN. Preserve historical Vx predictions/alpha as an immutable incumbent. Refit Vx only inside TRAIN folds to obtain honest auxiliary-head OOF inputs for selecting the new incremental alpha; final VAL predictions use the unchanged historical Vx head. Historical Vx alpha was selected on VAL calibration; no new alpha uses VAL.

Scores are `native + alpha * original_gate * evidence` and `frozen_Vx + alpha * original_gate * evidence`. Fit matched true, shuffled, coordinate-true/shuffled and boundary-true/shuffled heads as declared controls. Also apply the true-trained head unchanged to shuffled, coordinate-permuted and boundary-excluded features. These variants are diagnostics, not candidates selected on VAL.

Primary: **native + Vx + true contacts versus native + Vx** on the original 958 assessment rows. Contact-only fusion is a secondary replacement comparison. Compare the primary also against **native + Vx + quality/length**, so Vx itself cannot explain apparent separation from a weaker nuisance-only baseline.

## Evaluation and safeguards

Preserve the original VAL component assignment: calibration and crossing rows are descriptive only in V7. All new calibration is TRAIN-only. Report all fixed VAL rows and eligible subsets, length bands, TRAIN-derived quartiles of depth/Neff80/effective coverage/map size/quality/length asymmetry/null change, partial family witnesses, and removal of the ten most represented assessment proteins/family witnesses.

Use 1,000 matched protein Poisson-multiplier bootstraps with endpoint-product edge weights and 95% percentile intervals. Primary gain must be ≥0.005 AP with a positive interval, separation from the matched Vx+quality control, and AUROC loss ≤0.005 against Vx. Boundary exclusion and dominant-group removals must retain a positive primary point estimate for a robust positive decision. Subgroup intervals and mechanism controls are exploratory and unadjusted. Missing interval support or insufficient classes cannot pass a success criterion.

The contact-head membership audit is recorded in `provenance/exposure-audit.json`. The inspected release does not establish its training membership or TRAIN/VAL overlap. Any positive outcome remains exploratory, with unresolved exposure, incomplete family annotations, and a repeatedly inspected assessment set. Contact probabilities are not calibrated PPI probabilities; sequence-grid neighborhoods are not validated 3D interface patches.

## Execution

Use the existing idle one-GH200 allocation if qualification shows the complete run fits. Separate ceiling: 14 GPU-hours, 112 CPU-core-hours, and 30 GB of owned artifacts. The initial 12-hour planning cap was revised before freezing/launch after qualification projected a 12.4-hour conservative bound; preserve that evidence in the resource-plan revision record. Runtime is charged conservatively from the original recorded resource start, including qualification and preparation within that interval. Fixed safety factor 1.5 and 30-minute analysis/verification reserve. A deadline or corrupt/missing artifact stops the run. No automatic restart, queue submission or post-launch resource extension.

The existing SIF contains both frozen models. Store averaged true/shuffled FP32 A×B maps, validity masks and source hashes, with six-feature records for all 8,000 pairs. Do not cache the full 256-channel tensor. No old study or running benchmark is modified.

Entry points: `prepare.py`, `audit_exposure.py`, CPU unit tests, `native.py --qualify`, `qualify.py`, `freeze.py`, then host `launch.py --start`. The detached controller runs native TRAIN scoring, contact extraction, TRAIN fitting/VAL reporting and independent reconstruction of all cached summaries and VAL scores. `STATUS.md`, `results/phase.json` and `results/worker.json` track progress; completion requires `results/verification.json` and `results/COMPLETE.json`.
