# VZ proposal: does the Vx gain require homolog rows?

9 October 2026. **Proposal only: no `pilot-vz` implementation, feature extraction, fitting, or job launch has started.**

I recommend one query-only arm. It is a more direct ablation of Vx than VY was: change the input depth while preserving Vx's joint two-chain representation, readout, training population, and fusion procedure. The result can establish whether query-only joint encoding is sufficient to recover the observed gain under this fixed pipeline. It cannot, by itself, prove that cross-chain computation is the unique cause of that gain.

## Evidence and the question actually tested

The existing 958-pair assessment gives:

| Frozen model, including native-logit fusion | AP | AUROC |
|---|---:|---:|
| Native PLM-interact | 0.640930 | 0.648144 |
| Vx true | 0.664361 | 0.658658 |
| Vx shuffled | 0.663248 | 0.661071 |
| Vx quality/profile | 0.650942 | 0.652725 |

True minus native is +0.023431 AP, with the original 95% matched protein interval [+0.003601, +0.037539]. True minus shuffled is +0.001113 [-0.009639, +0.013333]. Thus Vx improved the baseline, but did not establish pairing specificity. Even true minus quality/profile remains uncertain: its interval is [-0.005171, +0.031090]. See the [original metrics](pilot-vx/results/metrics.json) and [Vx review](pilot-vx/review-20261008/REVIEW.md).

[VY](pilot-vy/results/REPORT.md) selected zero fusion weight for independent monomer-MSA and query-only embeddings. However, VY used the final query-residue state, monomer pooling, distinct-protein PCA, and a different eligible population. Its negative result does not isolate joint processing. [Vx v2](pilot-vx-v2/review-20261008/REVIEW.md) also failed to establish a benefit from its proposed local pooling. VZ therefore returns to the original successful global readout and changes only whether non-query rows enter the encoder.

**Hypothesis:** jointly encoding the two query sequences already supplies most of the useful Vx augmentation; correctly paired homologs may not be required, and even shuffled homolog rows may add little practical benefit.

Two limits must accompany any answer:

- Vx's residue masks, evidence eligibility, and reliability gate are MSA-derived. Keeping them fixed is essential for this ablation. A successful Q arm establishes that *homolog rows inside the encoder* are dispensable within this matched setting, not that the entire workflow can dispense with MSA data.
- Q versus shuffled changes both homolog content and input depth. If Q loses, the additional rows matter operationally, but this experiment cannot separate evolutionary information from depth-dependent behavior, gap/occupancy patterns, or a depth-one distribution shift. Nor does a positive Q result distinguish genuine interaction information from sequence, family, or ensemble effects.

## One new arm, with the original contract

For an originally eligible pair, reconstruct its original masked paired tokens from the existing monomer caches using the unchanged Vx pairing function. Then perform the sole encoder intervention:

```python
tokens, meta = original_msa.pair(monomer_a, monomer_b, original_config)
# Assert original availability, pairing metadata, and gate still match the cache.
q_tokens = tokens[:1].copy()   # shape: (1, length_a + length_b)
q_features = original_encoder.symmetric(q_tokens, length_a)
```

Use the same canonical A/B ordering as Vx. The breakpoint remains the original full A length; BA reverses the two complete coordinate blocks and uses the B length. Keep the original PAD=26 positions. Do not concatenate independently encoded monomers, unmask residues, crop sequences, introduce a separator residue, or replace the depth-one input with duplicated queries. The original `core_stack` arguments remain unchanged; do not copy VY's `query_only=True` argument or final-layer extraction.

| Component | Fixed Vx behavior |
|---|---|
| Samples | Original 4,000 TRAIN / 4,000 DEV pairs, labels, UIDs, order, and internal DEV memberships |
| Eligibility | Original 1,825 eligible TRAIN and 2,628 eligible DEV pairs; 4,453 new Q feature vectors |
| Internal DEV | 1,068 calibration, 958 assessment, 1,974 crossing; respectively 692, 659, and 1,277 eligible |
| Encoder | Existing pinned Pairformer SIF, source, weights, FP32 parameters/BF16 autocast, TF32 off, frozen evaluation |
| Representation | Inter-chain `final_pairwise_repr` returned after zero-based layer 15, with the same chain-break encoding |
| Readout | Average directed inter-chain blocks; same seeded 256-to-32 Gaussian projection; mean/std/max/q95 over identical valid cells; average AB and BA; 128 scalars |
| Q head | TRAIN-only StandardScaler → PCA(32, full solver) → logistic regression, C=0.1, lbfgs, max_iter=1000, seed 20261007 |
| Fusion | `native_logit + alpha * original_gate * Q_head_logit`; alpha in `{0, .1, .25, .5, 1}`, calibration AP only, smallest-alpha tie |
| Fallback | Exact native logits for every originally unavailable pair, including combined length >1,536 |

Fit Q's own scaler, PCA and classifier on exactly the original 1,825 eligible TRAIN rows. There are only 33 supervised classifier parameters including the intercept. Reusing the true/shuffled fitted transforms for Q's primary head would instead test distribution transfer and would not match the original separate-head comparison. Native TRAIN logits remain unused.

**Do not recompute the gate from the depth-one input.** That would either reject Q for insufficient homologs or shrink its weight for a reason unrelated to predictive information. Copy the original per-pair gate unchanged. Q could technically run on additional pairs, but expanding its availability would answer a different question and is excluded here.

## Smallest implementation after review

Use a thin, isolated `pilot-vz` wrapper around [Vx's encoder](pilot-vx/scripts/encoder.py), [pair construction](pilot-vx/scripts/msa.py), and [analysis contract](pilot-vx/scripts/analyze.py). Reuse VY's audit patterns for completeness, atomic writes, and independent prediction reconstruction; do not import VY's biological selection or modeling choices. Existing experiment directories remain immutable.

1. **Snapshot allowlisted TRAIN/DEV inputs and references.** Bind the original configuration/code, sample, native logits, monomer catalog/cache hashes, original feature metadata, heads, and metrics. The already verified Vx feature snapshot in `pilot-vx-v2/data/source.npz` is sufficient for old evidence vectors. Reconstruct true/shuffled/quality logits from their saved heads without refitting or recalibration; verify them against the saved Vx columns in `pilot-vy/results/dev_predictions.npz`, matching UIDs and labels. Original true/shuffled/quality alpha values are all 1. Reproduce original AP/AUROC before permitting Q fitting.
2. **Qualify the exact depth-one joint call without labels.** Reuse the original qualification UIDs `val-057993`, `val-041603`, and `train-027500` (lengths 478, 913, 1,535), plus `train-047928` at the actual maximum length 1,536. Check repeatability, AB/BA symmetry, depth exactly one, masks, breakpoint, finite 128-dimensional output, frozen layer, and forbidden output-head hooks. On the three old cases, reproduce cached original true/shuffled features exactly using the unchanged encoder. No complete true/shuffled re-extraction is needed. If runtime compatibility fails, stop rather than changing the precision, layer, or readout.
3. **Freeze, extract, and fit once.** Freeze VZ code/configuration, inputs, qualification, resource limit, and the criteria below before new outcome fitting. Record all 8,000 pair statuses, including the original biological fallback reasons. Generate Q only for the 4,453 eligible pairs, with two forwards each: 8,906 main encoder calls. Only then fit the one Q head, select its alpha on calibration, and evaluate the fixed assessment.
4. **Audit before completion.** Independently reconstruct Q and reused reference predictions, validate TRAIN-only transforms and calibration selection, recompute metrics and intervals, and check checksums and exact fallback. Write a completion marker only after this audit passes. Preserve a negative or inconclusive result without another fit or automatic continuation.

Do not run the original full `verify_freeze()` blindly: its manifest covers preparation inputs beyond the needed TRAIN/DEV path. Verify the manifest identity and explicitly allowlisted dependencies, as Vx v2/VY did. Do not open TEST sequences, labels, predictions, or TEST exclusion-hash files. Existing monomer caches already contain their original exposure exclusions; do not rebuild them from the archive.

## Primary comparison and decision rules

All primary comparisons concern **fused predictions** on the same 958 assessment rows, including fallback. Let B denote native PLM-interact, T true, S shuffled, and Q the new arm. The primary new estimand is **AP(Q) − AP(S)**. Report Q−T, Q−B, T−S, T−B, and S−B alongside it, with AP and AUROC for every model.

Preserve Vx's 1,000 matched protein Poisson-multiplier replicates, edge weights equal to the endpoint-count product, seed 20261007, and 95% percentile intervals. Use the same assessment protein multipliers across all model contrasts. The old T−S interval must reproduce. These intervals condition on fitted heads and selected alphas; they do not account for repeated development decisions or establish complete family independence.

Freeze these practical thresholds before Q outcomes:

- **Useful Q gain:** Q−B point estimate ≥0.010 AP, its 95% lower bound >0, and Q AUROC no more than 0.005 below B.
- **Q recovers the homolog-arm performance within a declared tolerance:** useful Q gain plus lower bounds for Q−S and Q−T both above −0.010 AP. This is a noninferiority statement, not proof of equal performance. A 0.010 tolerance is the original practical continuation scale; it allows loss of roughly 45% of the observed shuffled gain, which must be made explicit rather than called identical performance.
- **Practical equivalence (`≈`):** the entire 95% interval for the relevant AP difference lies within [−0.010, +0.010]. A nonsignificant difference does not qualify. Use the same rule for Q versus native when interpreting absence of a practical Q gain.
- **Material advantage (`>`):** point difference ≥0.010 AP and 95% lower bound >0. Smaller positive separations can be reported but are not a practical explanation for the original gain.

| Observed pattern, supported by the rules above | Permitted interpretation |
|---|---|
| Q recovers S/T performance and improves on B | Query-only joint Pairformer augmentation is sufficient within the declared tolerance and original MSA-derived masks/gate. Homolog rows are not needed to recover that much gain. |
| Q ≈ S ≈ T, all usefully above B | Stronger practical-equivalence version of query-only sufficiency; requires equivalence evidence for all three pairwise comparisons. |
| S and T materially exceed Q; T ≈ S | Non-query rows supply useful context in this pipeline, without a demonstrated need for correct pairing. Marginal evolutionary context is a plausible explanation, not uniquely identified. |
| T materially exceeds S, and S materially exceeds Q | Evidence for both non-query context and pairing sensitivity under the frozen null; still not proof of direct co-evolution. |
| Q ≈ B, while S/T materially exceed both | This Q configuration fails to recover the gain. Homolog-bearing inputs are needed for the tested pipeline; necessity of biological evolutionary signal remains unresolved. |
| Q improves B but cannot meet noninferiority, or intervals straddle the margins | Partial recovery or inconclusive attribution; retain the uncertainty and stop this configuration. |

There is an important known limit: **T−S is fixed already, and its interval extends above +0.010 and includes zero.** Adding Q cannot establish either T/S equivalence at this margin or a new pairing-specific advantage on this unchanged assessment. Consequently, the realistic target is the Q sufficiency/noninferiority question or evidence of an S−Q loss, not a definitive three-way mechanistic verdict. Do not widen margins after seeing Q.

Technical completion and scientific support are separate. Failed provenance, incomplete extraction, nonfinite features, nonconvergence, or a failed audit invalidate the experiment. A verified Q with alpha zero or insufficient gain is a valid negative/inconclusive result for this configuration. It is not evidence that all possible query-only representations lack useful information. This assessment has already been inspected repeatedly; every VZ conclusion remains exploratory and authorizes neither production nor TEST.

## Essential diagnostics without another model

Keep these small and prespecified:

- Report the 659 eligible assessment pairs separately, plus all fixed DEV, calibration, and crossing populations. All-row noninferiority can be diluted by the 299 shared fallback rows. If eligible-only uncertainty cannot exclude a meaningful loss, limit the conclusion to the full fixed population rather than claiming equivalence of the encoder signals.
- Show the Q calibration grid, selected alpha, unfused head AP/AUROC on eligible rows, and Q/T/S fused scores at every **existing** common alpha. These are descriptive diagnostics, never another selection procedure. In particular, alpha zero makes Q's fused result identically B and makes input-effect comparisons at that alpha uninformative.
- Retain the already fitted quality/profile prediction as a secondary reference. Before attributing gain specifically to neural pair information beyond these cheap features, require a positive Q−quality interval; otherwise state that the quality/profile explanation remains unresolved. It is not a new trained arm.
- Reuse existing length bins, Vx v2 paired Neff80/effective-coverage summaries and TRAIN-derived cutpoints, plus protein/family-witness influence summaries where available. No new searches, annotations, subgroup-specific models, or subgroup rescue decisions. Above 1,536 residues every arm is fallback, so agreement there says nothing about the encoder. Partial family witnesses cannot establish family independence.

Add only tests that guard the intervention: exactly one input row; unchanged query coordinates and chain break; unchanged eligibility/gate; symmetry; TRAIN-only fitting and calibration-only selection; missing/corrupt required Q outputs fail before fitting; finite evidence is required even when alpha is zero; original predictions and exact fallback reproduce. Output-head hooks must forbid contact and language-model output heads. These safeguards do not change the representation.

## Effort, resources, and recommendation

This is a small feature-extraction ablation and one CPU classifier, not a model-training project. Allow roughly **one to two focused implementation/review days**, largely for provenance and matched-comparison checks. Reuse `images/msa-pairformer/msa-pairformer-arm64-v1.sif`; no new SIF is justified.

Plan for approximately **3–6 allocated GH200 GPU-hours**, subject to real joint depth-one qualification. This is an estimate, not a benchmark: Vx extracted true and shuffled with four orientations per pair in about 8.39 allocated batch GPU-hours; VZ has two orientations. VY's depth-one timings also show that long-query pair computation remains expensive despite removing homolog rows. Depth one should not be advertised as a 128-fold speedup.

Propose a separate ceiling of **8 allocated GPU-hours, 600 CPU-core-hours, and 5 GB additional storage**, including preparation, qualification, extraction, fitting, audit, and failures. Do not silently spend Vx's remaining budget. Measure synchronized qualification timings, extrapolate by length with a safety reserve, and launch only if the full run and audit fit. If not, stop for resource review without shrinking the sample or changing the experiment. Store compact features and provenance, not dense pair tensors. No automatic retry, requeue, or budget extension.

**Recommendation: proceed with this one ablation after review.** It fills a specific missing control while retaining the successful Vx representation. A positive result would make query-only joint encoding a credible sufficient explanation within this fixed setup; a negative result would show that depth-one joint encoding alone does not recover the gain. Either outcome is useful, provided the report preserves the limits above. This proposal is the only new artifact at this stage; `pilot-vz` has not been started.
