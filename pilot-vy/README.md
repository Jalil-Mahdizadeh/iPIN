# Original Bernett VY pilot

Authorized 8 October 2026 from [the VY proposal](../improvment-proposal-vy.md). This pilot tests independent monomer evolutionary context added to the frozen native PLM-interact logit. It does not test cross-protein co-evolution. Current execution is in [STATUS.md](STATUS.md).

Use the exact original Vx 4,000 TRAIN / 4,000 DEV sample and original labels. Internal DEV membership remains 1,068 calibration, 958 assessment and 1,974 crossing pairs. Assessment was previously inspected; all VY evidence is exploratory. No TEST data, TEST exclusion-hash file, new search, new negative sampling, or V5/V11 model enters this study. Existing monomer caches already contain the original context-exposure exclusions and remain immutable.

## Frozen information and models

Select homologs per complete query, independently of its candidate partner or label: query-seeded accession-hash order, round robin over reported taxonomic families, 90%-identity redundancy with 80% good-position comparison coverage. Preserve original coordinate masks. Require at least eight retained homologs, 50% good positions and 50% good-column coverage of each selected homolog. Maximum 127 homologs plus query; maximum 1,536 residues per chain. No windows, crops, dynamic depth reduction or rescue searches. Unknown taxonomy is an explicit group; taxonomic families are not human protein families.

Use the pinned existing Pairformer SIF, FP32 parameters and BF16 autocast, TF32 disabled, frozen weights. The trunk runs all 22 blocks plus its final MSA update with one chain. Only the final 464-channel query-residue representation is read. Runtime hooks reject any call to the released contact heads or language-model output head. A query-only control uses exactly the same encoder and quality mask with depth one.

Per monomer, cache masked global mean and population standard deviation plus means in four balanced contiguous original-coordinate blocks (`numpy.array_split`): 2,784 numbers. Empty blocks are zero; their support is recorded. This retains coarse sequence organization, not an interface map. No dense tensor is persisted. Each required protein is encoded once for MSA and once for query-only context; the same cache is shared across all partners. Available monomers that never occur in a pair with another available monomer do not require neural extraction.

The profile control has exactly 192 entries, with names frozen in `data/profile-schema.json`. Twelve global metadata entries describe log1p length, raw/cached/filtered/retained depth, Neff80, comparison adequacy, log1p taxonomic family count/effective number, unknown taxonomy fraction, depth saturation and quality. Each of five coordinate regions (whole query plus four blocks) contributes 36 entries: frequencies of all 26 amino-acid/ambiguity/gap tokens, nongap-conditional entropy mean/std/q25/q75, occupancy mean/q25/min, query agreement, ambiguous-token fraction and good-position fraction. Homolog rows exclude the query. Frequencies average equally over good columns; gaps remain in frequency denominators, entropy conditions on the 25 nongap categories. Empty regions are zero. X/B/Z/U/O are retained explicit encoder categories and flagged separately; this is not a canonical-residue-only Neff definition.

Compute Neff80 on retained homologs, excluding the query: a pair is comparable if both rows are nongap at ≥80% of good query positions, and neighbors require ≥80% identical residues over their joint nongap positions. Sum reciprocal neighbor counts including self. Report comparison coverage as well; incomplete comparability can inflate this operational diversity proxy. The selection threshold is 90%, so Neff is not definitionally equal to retained depth.

## Five small heads and controls

For M (MSA), S (query-only) and P (profile), fit StandardScaler → PCA(16, full solver, no whitening) on distinct TRAIN proteins appearing in eligible TRAIN pairs, counting each once. Apply shared A/B transforms. Construct `(z_A+z_B)/2`, `abs(z_A-z_B)` and `z_A*z_B` (48 entries), then pair-level TRAIN-only StandardScaler → logistic regression (`C=0.1`, lbfgs, max_iter=2000). Convergence warnings fail rather than accepting an under-converged fit.

SP concatenates both 48-entry S/P pair descriptors without further PCA: 97 supervised coefficients including intercept, versus 49 each for M/S/P. It is intentionally a stronger control. U uses only M's 16-entry mean term: 17 supervised coefficients. No native TRAIN logits or residual targets enter fitting.

All new heads use identical evidence eligibility and reliability:

`g = available_A * available_B * min(1, min(Neff80_A, Neff80_B)/32) * min(quality_A, quality_B)`.

Fusion is `native_logit + alpha*g*head_logit`. Select alpha only on calibration from `{0, .1, .25, .5, 1}`, smallest-alpha tie. Native inference remains full-sequence, mean AB/BA logits, with the original checkpoint and precision. Every unavailable pair retains its native logit exactly, including for S/P controls. Missing computational results, corruption, nonfinite values, OOM and budget stops are errors or incomplete studies, never biological fallback.

Also evaluate the M-trained transform/head on S inputs without refitting, plus common-alpha M/S/SP tables. These diagnostics never select another weight. Report unfused head metrics only on eligible pairs. Original Vx true/shuffled/quality heads, predictions and gate are reused unchanged and must reproduce saved metrics exactly. Their differing eligibility and head dimension make them historical references. There is no biologically meaningful cross-chain pairing shuffle in two independent monomer encoders.

## Decisions and diagnostics

Primary endpoint: **AP(native+M) − AP(native+SP)** over all 958 assessment pairs, including fallback. Require a positive 95% interval; positive M-minus-native/S/P intervals; ≥0.010 AP over native; AUROC loss ≤0.005; and a positive primary point after separately removing each of the ten most frequent assessment proteins and recorded family witnesses. These are conjunctive exploratory continuation checks, not a multiple-comparison discovery search. M versus U tests the declared pair terms. Even a statistical pass leaves complete family breadth unresolved and authorizes no production or TEST.

Use 1,000 joint all-DEV protein Poisson-multiplier replicates, edge weight equal to endpoint-product weight, shared across every comparison. Report percentile intervals conditional on fitted models. Subgroups are descriptive, unadjusted and cannot replace the primary endpoint; intervals need ≥10 observations per class. Fit requires ≥100 examples per class; calibration and assessment require ≥50 examples per class.

Report full fixed DEV (4,000, not complete released DEV), assessment, calibration, crossing, common coverage/VY-only/Vx-only/neither, and fixed combined-length bins ≤512/513–1024/1025–1536/>1536. TRAIN-derived quartiles stratify monomer Neff, depth, quality, effective occupancy, comparable fraction, taxonomic diversity, ambiguity and length asymmetry. Report prevalence, AP standardized to 50% prevalence, AUROC, correction magnitude and Brier score. Family information is the existing DEV-only TRAIN-shared witness table, with annotation adequacy and explicit missingness; lack of a witness is not family novelty.

Scientific failure or uncertainty stops this configuration without pooling/layer/PCA/alpha expansion. An independent CPU audit reconstructs every saved prediction, checks TRAIN-only transforms and calibration selection, and recomputes all primary-assessment contrast intervals using sklearn AP, independently of the fast production AP implementation.

## Execution and budget

VY has a separate ceiling: **8 allocated GPU-hours, 600 allocated CPU-core-hours, 10 GB additional working storage**, including preparation, qualification, failures, analysis and verification. The dedicated interval begins in `provenance/resource-start.json` within the user's existing one-GPU / 72-CPU allocation, job 3511678. Charge full allocated resources during this interval, including idle time; earlier unrelated allocation time belongs to the allocation's scheduler accounting, not to VY. The Vx/V2 remaining budget is not reused. No new SIF or SLURM allocation is necessary if label-independent timings fit the remaining interval and allocation expiry.

Qualification uses deep monomers nearest fixed length targets 256/768/1280/1536, plus the longest and shallowest required monomers. Exact repeatability, depth-one execution, full-stack hooks and label-independent row-order sensitivity are checked. Conservative scheduling uses the upper-length timing buckets, a factor of two, ten minutes I/O overhead and thirty minutes for analysis/audit. The full pipeline starts only if it fits the existing budget. It is detached within the existing allocation, with an execution lock, one launch intent and deadline enforcement. No automatic retry/requeue or second run is enabled.

Freeze snapshots and hashes bind all executable code, tests, config, selected monomer tokens, required IDs, profiles, source references and qualification before any new outcome fitting. A completed run needs every required neural feature and every fixed pair. `run.py` performs extraction, complete-coverage verification, fitting/reporting and independent audit automatically; only then writes `results/COMPLETE.json` and final resources.

```bash
python -B pilot-vy/scripts/status.py
```

The original Vx experiments and unrelated workspace files remain independent of this pilot.
