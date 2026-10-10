# Original Bernett Vx v4: independently sampled homologs

Authorized 10 October 2026 (Stockholm). One new **I** arm tests whether jointly encoding intact A/B homologs can retain the Vx gain without requiring a shared-accession intersection. [STATUS.md](STATUS.md) tracks execution. This is a prospective protocol on previously inspected DEV, so all conclusions remain exploratory. R remains deferred; no TEST access.

**Primary comparison:** I versus original Vx shuffled, with true, Q, C, quality/profile and native PLM-interact reused unchanged. The primary population is the original 958-row assessment, including the original native fallbacks. Do not expand eligibility or select subgroups based on results.

## Independent sampling intervention

Use the original verified monomer caches (up to 4,096 accession keys per monomer); no new sequence search or raw-archive preparation. Each cache already carries the original alphabet, coverage and human-context exclusions. This tests removing the intersection within these retained pools, not unrestricted sampling from every archived homolog.

For each originally eligible pair, retain the exact original homolog count (at most 127), query row, residue order, PAD mask, length, eligibility and reliability gate. Filter each monomer's intact rows independently using the original minimum coverage on valid query columns. Deterministic SHA256 priorities, separately salted for A and B, order accession keys and taxonomic families. Round-robin taxonomic families independently within each chain, taking the required number without replacement. Independently randomize each selected chain's row order, then concatenate A/B rows by index. No shared-accession/taxonomy matching, cross-chain rejection, or preference for the old intersection is allowed. Coincidentally matched keys or identical A/B sequences are retained and counted; rejecting them would couple the draws.

There is no additional per-chain sequence-redundancy filter: imposing one can make the required original depth unattainable. Original Vx's joint redundancy filter cannot be retained while selecting chains independently. Consequently this intervention changes homolog selection/diversity as well as matching; measure these changes explicitly. Every synthetic pair row consists of two intact masked homolog sequences, unlike C's within-sequence column scrambling. Fail if any originally eligible pair lacks enough independently valid keys; never replace this with duplication, lower depth, new fallback, or a smaller analysis cohort.

Seeds use `vx-v4:I:v1:{seed}:{uid}:{a}:{b}:{replicate}:{chain}:{purpose}` with seed 20261007, canonical original A/B IDs, independent chain salts, and primary replicate zero. Use SHA256 priorities for selection and NumPy PCG64 seeded by the first 16 hex SHA256 digits for final row ordering. No outcome-based seed selection. Replicates 1/2 are qualification-only checks on three old label-free examples, never fitted.

## Unchanged pipeline and governance

Keep the exact 4,000 TRAIN / 4,000 DEV samples, internal memberships, 1,825/2,628 eligible pairs, and original gates. Encode 4,453 pairs symmetrically; require all 8,000 records before fitting. The same original Pairformer encoder, SIF, zero-based layer 15, inter-chain representation, seeded 256-to-32 projection, mean/std/max/q95 summaries, and AB/BA average produce 128 features. Frozen evaluation, FP32 weights/BF16 autocast, TF32 off; no pretrained output heads, cropping changes, new pooling or architecture.

Fit exactly one I head: TRAIN-only StandardScaler, full-solver PCA(32), logistic regression C=0.1, lbfgs, max_iter=1000. Select alpha from {0, .1, .25, .5, 1} on the existing calibration set, smallest-alpha tie. No native TRAIN logits. Retain all historic heads, alphas, predictions and bootstrap populations; independently reproduce their results. All unavailable pairs keep the exact native prediction. Computational failures invalidate completion, not biological availability.

## Prespecified decisions

Use the original matched protein bootstrap: 1,000 Poisson resamples, seed 20261007, endpoint-product weights, 95% percentile intervals. These are conditional on this sample, fitted heads, selected alphas and one sampling realization; they exclude seed and development-selection uncertainty.

- **Useful I gain:** I minus native at least +0.010 AP with lower interval above zero, and AUROC no more than 0.005 below native.
- **Recovery within tolerance:** useful gain plus lower bounds for I minus shuffled and I minus true both above -0.010 AP. This is noninferiority; the margin permits loss of about 45% of the original shuffled gain.
- **Improvement over shuffled:** I minus shuffled at least +0.010 AP with lower interval above zero. Report the true comparison separately.
- **Meaningful loss:** shuffled minus I at least +0.010 AP with lower interval above zero. Otherwise unresolved recovery is inconclusive, even if point estimates differ.
- Report I minus Q, C and quality/profile; a material contrast requires +0.010 AP and a positive lower bound. Equivalence requires the entire interval inside +/-0.010 AP; nonsignificance is not equivalence. Report eligible-only noninferiority as a sensitivity check.

Recovery would support independently sampled intact families as practically sufficient within this pipeline. It would not establish absence of cross-protein evolutionary information or isolate pairing from changed homolog composition. Failure would not prove that matching accessions is biologically necessary. No subgroup rescue, additional head/seed search, automatic production continuation or TEST evaluation.

## Diagnostics and qualification

Record per-chain source/filtered/selected depth, accession overlap with original selected rows and the other chain's eligible pool, coincidental row matches, identical cross-chain sequences, taxonomic breadth, occupancy/effective coverage, query identity, Neff80, within-chain and cross-chain sampled covariance, and changes in column profiles. Preserve the old TRAIN-derived length/Neff/coverage strata and partial family witnesses. Diagnostic counts are never used to select seeds or fit a new quality control.

For original fallback pairs, count potential independent availability using the same length/quality/coverage limits and at least eight rows per chain, capped at 127. Keep this label-free coverage audit separate: no encoding, new head, rescoring, or expansion of the primary cohort. Count insufficiency/missing monomers explicitly; corrupt existing artifacts raise errors.

Tests must verify separate chain sampling, no replacement, exact depth/query/PAD, intact source-row membership, invariance to cache ordering and labels, deterministic repeats, partner-pool independence at fixed depth, corruption/insufficient-pool failures, no source mutation, complete-cohort enforcement, TRAIN/calibration boundaries, exact fallback, finite features even at alpha zero, and independent AP/decision reconstruction. GPU qualification replays original true/shuffled features exactly, checks I repeats and AB/BA symmetry, and covers maximum length/depth plus fixed length timing anchors. Freeze code, inputs and protocol before outcome fitting. Final CPU audit independently verifies selected keys/rows, replays sampling, reconstructs predictions and intervals without refitting.

## Resources and execution

Reuse the idle GH200 in existing allocation 3565552 and `images/msa-pairformer/msa-pairformer-arm64-v1.sif`. Separate ceilings: 10 allocated GPU-hours, 720 allocated CPU-core-hours, 5 GB added storage, charged from task inspection at 2026-10-09 23:02:29 UTC. Also obey the earlier allocation expiry. Launch only if label-free timing with 1.5 safety factor plus 600 seconds I/O and 1,800 seconds analysis reserve fits both limits. No new scheduler job, container, automatic retry, budget extension or cancellation of the user's allocation.

A detached controller runs extraction, one fit/report, independent audit and resource accounting. Only verified complete results within budget receive `results/COMPLETE.json`. Run `python -B pilot-vx-v4/scripts/status.py` for read-only status.
