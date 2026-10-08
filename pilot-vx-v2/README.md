# Original Bernett Vx v2: local-pooling ablation

Authorized 8 October 2026 following the [completed-pilot review](../pilot-vx/review-20261008/REVIEW.md). This is one bounded, exploratory ablation of spatial aggregation. Current progress is in [STATUS.md](STATUS.md); the result will be written to `results/REPORT.md` and `results/decision.json`.

Hypothesis: coherent local inter-chain patches contain useful pairing-dependent information lost by global summaries. The separate short-pair assessment did not establish a length-specific benefit; short pairs are a reporting stratum, not a selected training/evaluation population.

## Frozen design

Reuse the exact 4,000 original Bernett TRAIN / 4,000 DEV pairs, native PLM-interact logits, eligibility, gate, paired rows, taxonomic pairing-null permutations, encoder/weights/precision/layer, 256→32 projection, and AB/BA averaging from `pilot-vx`. No TEST file, TEST exclusion-hash file, or mixed-cohort annotation input is accessed. Cached monomers already embody the original exposure exclusions and remain immutable. Original global features must reproduce **exactly for every eligible pair**, or extraction fails.

Each 128-dimensional local vector contains 32 global means, 32 standard deviations, 32 means of the four highest block means and 32 means of the four lowest block means. Split each original chain into ceil(length/8) balanced contiguous blocks. Normalize sums over valid cells; require ≥50% valid positions in each constituent chain block. If fewer than four blocks qualify, use all. Do not join masked-away coordinates or crop sequences for encoding.

The spatial null permutes valid residue positions independently within each chain after encoding, jointly across channels. It preserves masks and the multiset used by the original global summaries. Per-pair/per-chain seeds are deterministic and label-independent. Identical permutations apply to true and pairing-shuffled tensors, with chain identities respected under AB/BA reversal. Both spatial views come from the same forward passes.

Fit exactly four heads: local true, local pairing-shuffled, spatially scrambled local true, and spatially scrambled local pairing-shuffled. Each uses eligible TRAIN only: StandardScaler → PCA(32) → logistic regression C=0.1, max_iter=1000. Choose fusion alpha from {0, .1, .25, .5, 1} on the original calibration subset only, ties to the smallest alpha. Reuse the original quality/profile control and original global heads without refitting. Also evaluate each true-trained head on its corresponding pairing-shuffled features without refitting. Native TRAIN logits are never fitting inputs or residual targets.

Assessment remains the original 958 pairs; calibration 1,068; crossing 1,974. Report the complete fixed DEV sample as well. This assessment has been inspected previously, so all follow-up evidence is exploratory. No hyperparameter search, depth expansion, length-specific optimization, alternative fusion grid, or automatic continuation is allowed.

## Diagnostics and decision

Record retained/candidate/cached paired depth, cap saturation, declared 80%-identity Neff with ≥80% comparison coverage separately for both chains and their concatenation, comparison-coverage distributions, taxonomic diversity, per-column occupancy, jointly nongap residue-pair support, valid cell area, length asymmetry, and actual sequence/token changes under the pairing null. These are diagnostics, not new features or gate inputs. Neff remains a threshold-defined proxy rather than a ground-truth count of independent evolutionary observations.

The existing DEV-only FADI table supplies TRAIN-shared protein-family witnesses and annotation coverage. It is matched by sequence SHA256 and frozen. Complete family annotations are not available through an exclusively TRAIN/DEV input, so report this limitation and do not equate missing witnesses with novel families. Taxonomic families of homolog rows are distinct from protein families of human queries.

Primary contrast on assessment:

`(AP_local_true − AP_local_shuffled) − (AP_global_true − AP_global_shuffled)`.

Require its 95% matched protein-bootstrap interval above zero, positive local-true versus baseline/shuffled/quality intervals, ≥.010 AP over native baseline, improvement of the true model over the original true model, AUROC decline no greater than .005, and adequate actual null changes (≥80% of covered assessment pairs have ≥50% of homolog sequences changed). Report the local-versus-original true interval as well. A gap increased solely by harming the shuffled model does not pass. Family breadth must remain unresolved if the available annotations cannot establish it; no result authorizes production.

For a stronger locality interpretation, also require positive intervals for the pairing-gap advantage of real versus scrambled spatial order and for the local true head's true-versus-shuffled-input contrast. Otherwise a pairing gain cannot be attributed to local coherence. Even a positive result would not establish direct compensatory coevolution, since biological partner assignment and finer phylogeny remain imperfectly controlled.

Use 1,000 joint protein Poisson-multiplier bootstrap replicates across DEV, multiplying endpoint weights for each pair. Report AP, AUROC, and AP standardized to 50% prevalence. Fixed length bins, coverage/diversity strata with cutpoints derived from eligible TRAIN, and protein/family-witness removal sensitivity are secondary, unadjusted diagnostics. Skip subgroup intervals with fewer than ten observations in either class; record counts. Do not replace the primary endpoint with a favorable subgroup. Equal true/shuffled gains fail the pairing hypothesis; wide intervals are inconclusive and trigger no automatic larger run.

## Integrity, resources and execution

The new input snapshot verifies all 8,000 original feature checksums and identities. A separate freeze binds source snapshots, reused code, new code/tests, qualification, worker assignment, and the protocol before any new outcome fitting. Every computed feature record has its own checksum and fingerprint. Missing/corrupt/nonfinite computational results cause an explicit error or inconclusive report, never biological fallback. Original ineligible examples retain the native logit exactly. There is one submission and no automatic requeue or retry.

Resource accounting starts from 12.893333 charged GPU-hours in the original pilot/audit. Reserve one additional GPU-hour for v2 preparation/qualification on the existing one-GPU allocation, beginning 09:24:23 UTC, and at most ten GPU-hours for one four-GPU, 2h30m batch. Maximum cumulative charge is 23.893333 of the original 24 GPU-hours. Qualification must finish within its reservation. Conservatively reserve 700 prior CPU-core-hours and 260 v2 CPU-core-hours, within 2,000; retain the cumulative 150-GB additional-storage ceiling. Failed work counts. A budget stop is inconclusive; no automatic extension is enabled.

The existing ARM64 SIF is reused after checksum verification. Dense tensors are never persisted. A deterministic scheduling assignment balances the source's measured per-pair costs without labels. Allocation time, not summed forward time alone, determines GPU charging. The batch performs extraction, complete-coverage verification, CPU head fitting, analysis and reporting automatically.

```bash
python -B pilot-vx-v2/scripts/status.py
```

The underlying scripts, input identities, qualification results, scheduler identifiers and final resource records remain in this directory. The original `pilot-vx` study is not modified.
