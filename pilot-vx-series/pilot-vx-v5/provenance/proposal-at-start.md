# Original Bernett Vx v5: matched MSA depth 256 versus 128

Authorized 10 October 2026. Test whether additional accession-matched homologs improve the original Vx pilot, before deciding about a separate full benchmark. **256 means at most 255 homologs plus the query; 128 means at most 127 homologs plus the query.** This remains exploratory on previously inspected DEV. No TEST access, full benchmark, R, architecture change or hyperparameter search is authorized by this pilot.

Vx-v4 did not establish recovery under independent sampling. That result does not prove accession matching is necessary because composition and diversity also changed. Here accession matching is retained as the existing working method.

## Intervention and controls

Three new arms: **T256** (matched homologs), **S256** (original taxonomy-aware row shuffle at the deeper depth), and **P256** (original quality/profile control with the deeper pairing metadata). Reuse the original depth-128 true, shuffled, quality/profile and native PLM-interact predictions unchanged. Primary comparison: **T256 minus original true** on the original 958-pair assessment, including native fallbacks.

Use the original verified monomer caches and unmodified `pilot-vx/scripts/msa.py::pair`, changing only `maximum_paired_rows_including_query` from 128 to 256. Retain the original common-accession intersection, taxonomy compatibility, coverage checks, identical-cross-chain rejection, family round-robin and joint 90% redundancy filter. Require every original matched token row and selected key/taxonomy to be an exact prefix of the deeper input. No duplicated rows, extra searches or depth padding. Naturally shallow pairs retain their available depth; report counts unchanged, partly extended, and reaching 256.

Apply the original shuffle with the original per-pair seed to the complete deeper alignment. Its old-prefix permutation may change as depth grows; the shuffled arm is a control using the same algorithm, not a prefix-preserving intervention. Verify query/PAD, each chain's row multiset and column frequencies. Reuse original true/shuffled features when depth is unchanged, only after exact token and metadata reproduction. New classifiers are fitted on the whole original eligible TRAIN population, so reused input features need not have unchanged fitted predictions.

## Unchanged pipeline

Keep 4,000 TRAIN / 4,000 DEV records, the existing 1,068 calibration / 958 assessment / 1,974 crossing memberships, 1,825/2,628 eligible pairs and original reliability gates. Require all 8,000 feature records and all 4,453 eligible inputs. A compute failure cannot become a native fallback. Increasing the cap cannot remove previously eligible pairs. Verify the newly computed natural gate equals the original gate; it already saturates above 32 effective homologs, while originally shallow inputs cannot gain rows under an otherwise identical selector.

Same frozen Pairformer/SIF, zero-based layer 15, inter-chain pair representation, fixed 256-to-32 projection, mean/std/max/q95 summaries (128 features), AB/BA average, FP32 weights with BF16 autocast and TF32 disabled. No structural output heads or new pooling. P256 uses the original 68 quality/profile features and includes updated paired depth/Neff; monomer profiles are unchanged.

Fit one classifier per new arm: original TRAIN-only StandardScaler, full PCA(32), logistic regression C=0.1, lbfgs, max_iter=1000, seed 20261007. Choose fusion alpha only on the old calibration split from {0, .1, .25, .5, 1}, with smallest-alpha ties. No native TRAIN logits. Preserve exact native fallback and all historical reference predictions/intervals. Do not select a depth, seed, subset or head after inspecting assessment outcomes.

## Decisions and interpretation

Use the original matched protein Poisson bootstrap: 1,000 draws, endpoint-product weights, seed 20261007, 95% percentile intervals. The primary interval uses the original assessment protein universe. Intervals condition on these samples, heads and calibration selections; they do not account for repeated DEV inspection or training uncertainty.

- **Meaningful depth benefit:** T256 minus true128 at least +0.010 AP with lower interval above zero, and T256 AUROC no more than 0.005 below true128. Also report its gain over native and measured compute cost.
- **Evidence against the extra compute:** the primary upper interval is below +0.010, ruling out the prespecified material benefit within this analysis. A smaller positive gain can exist. A material loss requires a point estimate at most -0.010 with the upper interval below zero.
- Otherwise the depth comparison is inconclusive. Equivalence requires the entire interval inside +/-0.010; nonsignificance alone is not equivalence.
- Report S256 minus shuffled128, P256 minus quality128, both new learned arms minus P256, and T256 minus S256. Increased pairing sensitivity requires T256 minus S256 at least +0.010 with a positive lower interval, plus a positive lower interval for `(T256 - S256) - (true128 - shuffled128)`. These secondary comparisons are exploratory and unadjusted.

A shared gain in matched and shuffled arms supports additional homolog context, not a demonstrated pairing-specific effect. No automatic full benchmark or TEST evaluation follows any outcome.

## Diagnostics and validation

Record actual paired depth, Neff80 for joint and individual chains, effective coverage/occupancy, query identity, taxonomic breadth, added keys, shuffle token/row changes and timing/peak memory. Retain original TRAIN-derived strata and partial family witnesses; add fixed depth-unchanged, partially-extended and full-256 groups. These are descriptive, not subgroup rescue or model selection.

CPU checks cover exact prefix, masks/query, original gate, source membership, matching, deterministic cache-order/label independence, natural shallow depth, corrupted inputs/features, class/split governance, all-arm complete coverage, finite fusion, baseline fallback, and independent statistics. Label-free GPU qualification must exactly reproduce original true/shuffled features, show repeatability and AB/BA symmetry at depth 256, and cover maximum length/depth. Freeze protocol, code, inputs and qualification before any outcome fitting. The final CPU audit reconstructs predictions and intervals without refitting, and verifies every deeper source row and all depth/prefix/shuffle invariants.

## Resources and execution

Use the idle GH200 in allocation **3615626**, the original SIF, and no new scheduler job. Ceilings: **23 allocated GPU-hours, 1,656 allocated CPU-core-hours, 6 GB added storage**, charged from 2026-10-10 06:04 UTC including preparation and idle time; also obey allocation expiry. Launch only if label-free timing for both learned arms, a 1.5 safety factor, 600 seconds I/O and 3,600 seconds analysis/audit reserve fit. No automatic retry, resource extension, cohort reduction, silent depth truncation or cancellation of the user's allocation.

The detached controller performs extraction, fitting, independent verification and resource accounting. Only a fully verified run within budget receives `results/COMPLETE.json`. See [STATUS.md](STATUS.md); `python -B pilot-vx-v5/scripts/status.py` is read-only.
