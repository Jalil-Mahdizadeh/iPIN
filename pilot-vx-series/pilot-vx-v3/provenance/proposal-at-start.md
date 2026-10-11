# Original Bernett Vx v3: column-shuffled homologs

Authorized 9 October 2026. This study implements **C only**; repeated-query **R is deferred**. [STATUS.md](STATUS.md) tracks execution. The [family-context hypothesis](../pilot-vx-v2/review-20261008/REVIEW.md) and [completed VZ study](../pilot-vz/results/REPORT.md) motivate this next ablation. This protocol is frozen before any new C outcome fitting; the previously inspected DEV makes its conclusions exploratory.

## Intervention and question

Starting with the exact original Vx masked paired tokens, keep row zero unchanged. Independently permute the non-query entries within every alignment column, including gap tokens. Do not reorder alignment columns. The original PAD columns stay fixed. C preserves depth, query coordinates, and the complete unweighted residue/gap histogram at every position, while disrupting the original arrangement of residues across homolog sequences. C contains synthetic rows assembled from real homolog residues, not intact homolog sequences. Finite samples and conserved positions retain some dependence; zero covariance is not guaranteed.

The primary question is whether this preserved positional-profile information, processed jointly through the original encoder, can recover the useful Vx augmentation. C does not isolate covariance alone: it also alters row-wise identity, phylogenetic structure, coherent gap patterns, and the sequences used by learned weighting. C success supports sufficiency of retained information within this pipeline; C loss does not uniquely prove a biological covariance mechanism. R is not run, so nominal-depth explanations remain incompletely tested.

Use a single fixed realization per pair. Seed NumPy PCG64 from the first 16 hex digits of SHA256 of `vx-v3:C:v1:{seed}:{uid}:{a}:{b}:{replicate}`, with canonical original A/B IDs, seed 20261007, and primary replicate 0. Draw an independent row permutation for each column in coordinate order. Build C once in canonical coordinates; the unchanged encoder swaps those same tokens for BA. There is no seed search, rejection sampling for a stronger perturbation, or selection by outcomes. Replicates 1 and 2 are used only on the three original label-free qualification examples to describe input/feature sensitivity; they are never fitted or selected.

## Matched pipeline and controls

Keep original 4,000 TRAIN and 4,000 DEV rows, labels, UIDs, and internal memberships. There are 1,825 eligible TRAIN and 2,628 eligible DEV pairs; 4,453 symmetric C encodings and 8,906 main forwards. DEV has 1,068 calibration, 958 assessment, and 1,974 crossing rows, with 659 eligible assessment pairs. Every originally unavailable pair retains the native PLM-interact logit exactly. All 8,000 status records are required before fitting. Computational failure is never converted to biological fallback.

Reuse the pinned Vx SIF and unchanged `pilot-vx/scripts/encoder.py`: frozen evaluation, FP32 parameters/BF16 autocast, TF32 off, zero-based layer 15, directed inter-chain symmetrization, original 256-to-32 seeded projection, mean/std/max/q95 over original valid cells, and AB/BA averaging into 128 scalars. No contact or language-model output head executes. No new pooling, architecture, cropping, masking rule, or sequence search is introduced.

Fit one C head on the original eligible TRAIN rows: StandardScaler → PCA(32, full solver) → logistic regression C=0.1, lbfgs, max_iter=1000, seed 20261007. Select fusion alpha from `{0, .1, .25, .5, 1}` using calibration AP only, smallest-alpha tie. Copy original eligibility and reliability gates; do not deduplicate synthetic rows, re-filter them, or recompute gates from C. Native TRAIN logits remain unused.

Reuse verified Vx **true**, **shuffled**, and **quality/profile**, VZ **Q**, and the native baseline. Reconstruct their predictions from saved heads and archived features, check against verified DEV arrays, and preserve original alphas (Q=0.5; true/shuffled/quality=1). No reference refitting, recalibration, or GPU re-extraction beyond qualification. Q and C use different selected alphas if calibration requires it; common-alpha and unfused-head diagnostics distinguish this from the primary fused-pipeline comparison.

## Prespecified evidence and decisions

Primary estimand: **AP(C) − AP(shuffled)** on all 958 assessment rows, including fallback. Also report C−true, C−Q, C−native, C−quality and the existing Q/true/shuffled contrasts. Preserve original population-specific matched protein Poisson multipliers: 1,000 draws, seed 20261007, endpoint-product edge weights, 95% percentile intervals. Original Vx and VZ contrasts must reproduce. Intervals condition on fitted heads, selected alphas, and this one C realization; they do not cover development-selection or permutation uncertainty.

- **Useful C gain:** C−native ≥0.010 AP, its lower interval bound >0, and AUROC no more than 0.005 below native.
- **Recovery within tolerance:** useful C gain and lower bounds for C−shuffled and C−true both >−0.010 AP. Call this noninferiority, not equality. This margin permits loss of roughly 45% of the original shuffled gain.
- **Profiles recover the gain beyond Q:** recovery within tolerance plus C−Q ≥0.010 AP with lower interval bound >0. This supports sufficiency of retained profiles plus the fixed encoder/masks/gate in this classifier, not a unique biological mechanism.
- **Meaningful loss under column shuffling:** shuffled−C ≥0.010 AP with lower interval bound >0. A true−C advantage is reported separately. Loss can reflect destroyed dependencies or the synthetic-input distribution shift.
- **Practical equivalence:** the whole interval is inside ±0.010 AP. A nonsignificant contrast does not establish equivalence. The already fixed true−shuffled interval does not meet this criterion; adding C cannot change it.

If C gains over native but fails recovery, report partial recovery only when supported; otherwise report inconclusive attribution. A verified alpha-zero, negative, or inconclusive outcome is scientifically valid. Technical failures invalidate completion. No adjustment of thresholds, subgroup rescue, extra fit, automatic continuation, or TEST access follows any outcome.

## Manipulation checks and diagnostics

Before freezing, prepare only allowlisted TRAIN/DEV inputs and compact original/C token arrays. Verify original paired-token hashes against Vx v2 and original query hashes/metadata against VZ. For each eligible pair, check exact column histograms, row-zero identity, PAD mask, depth, breakpoint, alphabet, deterministic seed, and no mutation of the source. Bind every prepared input checksum and an input-audit manifest. Preserve unperturbable conserved columns; do not reject them based on weak change.

Record actual and permutation-expected changed-token fractions, variable-column counts, row-wise query-identity and gap-fraction dispersion. On up to 128 deterministic variable-column pairs in each of within-A, within-B, and cross-chain categories, compare categorical cross-covariance energy for original true, original row-shuffled, and C inputs. These label-free diagnostics include gaps and exclude the query; finite-depth residual dependence is expected. They diagnose the intervention rather than assert zero covariance or select examples/seeds. Report population aggregates and original length/Neff/effective-coverage strata, using the existing TRAIN-derived cuts and partial family witnesses.

GPU qualification reproduces old true/shuffled features exactly on `val-057993`, `val-041603`, and `train-027500`, checks deterministic C repeats and AB/BA symmetry, and exercises the maximum length 1,536 with `train-047928`. Add fixed label-free timing anchors near 256/512/768/1024/1280/1536 at maximum retained depth (128). Observe both orientations' original depth and execution of blocks 0–15; forbid output heads. Qualification-only alternate seeds never enter the primary feature set. These few inputs may be reconstructed independently during bulk CPU preparation; freezing requires their C hashes and depths to match the completed prepared inputs exactly.

Tests cover independent column permutations, histograms/query/PAD preservation, corruption and metadata tampering, source immutability, missing-output rejection before fitting, TRAIN-only transforms, calibration-only selection, exact fallback, finite evidence even at alpha zero, reference prediction reuse, and independent sklearn bootstrap reconstruction. The final CPU audit rechecks prepared-input invariants and feature checksums, reconstructs all six models' predictions and decision rules, and verifies the original intervals without refitting.

## Execution and resources

Use the already idle GH200 in allocation 3565552; do not submit a new scheduler job or disturb other work. Reuse `images/msa-pairformer/msa-pairformer-arm64-v1.sif`. Separate hard ceilings are **10 allocated GPU-hours, 720 allocated CPU-core-hours, and 5 GB additional storage**, charged from this task's preparation start, including idle time, qualification, failures, fitting, and audit. Earlier pilots have separate ledgers. Keep the user's interactive allocation open.

Launch only after synchronized qualification timings, monotonically bounded by length, give a full-run estimate fitting the remaining budget/allocation with a factor of 1.5, 600 seconds I/O reserve, and 1,800 seconds analysis/audit reserve. A detached preparatory supervisor pins code/protocol, waits for all audited inputs, runs the complete test suite, freezes provenance, and launches once. It stops if preparation fails, code changes, or the remaining budget becomes insufficient. The GPU launch independently checks that the existing allocation is active and idle. Process longest eligible pairs first, then original fallbacks. A detached controller runs extraction → one fit/report → independent audit → resource accounting. Only a successful audit within budget writes `results/COMPLETE.json`. No automatic retry, requeue, budget extension, new container, or incomplete-cohort analysis.

```bash
python -B pilot-vx-v3/scripts/status.py
```
