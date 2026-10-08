# Vx review and one bounded follow-up

8 October 2026. Recommendation: one frozen, capacity-matched pooling ablation is scientifically reasonable. The evidence does not justify a short-protein-only study, larger training, or production continuation. No follow-up model has been implemented or launched.

This review reverified all 8,000 cached TRAIN/DEV records and reconstructed the saved heads without refitting. Original metrics and exact baseline fallback reproduced. The new [diagnostics](diagnostics.json) and [reproducible CPU script](diagnostics.py) are separate from the frozen pilot; no TEST artifact was opened, no GPU was used, and original results/configuration/code were not changed. The CPU review took 230 seconds. Its subgroup analyses are exploratory, with unadjusted intervals, not a new blind evaluation.

## What the length result actually supports

The original primary assessment has AP 0.64093 for native PLM-interact, 0.66436 for baseline + true MSA, 0.66325 for baseline + shuffled MSA, and 0.65094 for baseline + quality/profile. The true-minus-shuffled difference is only +0.00111, with the original 95% interval [-0.00964, +0.01333].

The published length table used all 4,000 sampled DEV pairs, including calibration and crossing pairs. Separate assessment results do not show the same monotonic length trend:

| Combined query length | All DEV pairs | True-minus-shuffled AP | Assessment pairs | True-minus-shuffled AP |
|---|---:|---:|---:|---:|
| ≤512 | 416 | +0.013995 | 103 | +0.007667 |
| 513–1024 | 1,759 | +0.002592 | 450 | -0.003560 |
| 1025–1536 | 1,051 | +0.000026 | 254 | +0.009716 |

New joint protein-multiplier bootstraps give a short-pair interval of [-0.00454, +0.03392] on all DEV and [-0.01811, +0.03847] on assessment. The assessment difference between the pairing gaps for ≤512 and 513–1536 is +0.00869, interval [-0.02066, +0.04048]. The eligible-only comparison is also inconclusive. Pairs above 1,536 cannot inform this comparison because their MSA contributions are identically zero.

On the 103 short assessment pairs, the **same true-trained head** gives AP 0.78694 on true features and 0.78937 on shuffled features. Baseline + quality/profile gives 0.78409. Thus the separate-head short-pair gap is not persuasive evidence of a causal pairing benefit. Short assessment prevalence is 63/103, versus 221/416 on all short DEV; absolute AP and AP differences across populations also depend on prevalence.

Among eligible pairs in the three length bins, median valid inter-chain cell counts are **25,056 / 84,640 / 212,250**, while median depth stays 127. Median depth per valid query residue falls **0.366 / 0.195 / 0.120**. These support two competing hypotheses: spatial dilution and limited alignment information per residue. Short pairs do not have better median minimum quality (0.755 / 0.773 / 0.775), more retained depth, or a much stronger null (median moved-row fractions 0.921 / 0.913 / 0.921). Their monomer gap fractions and conservation differ, so composition remains a possible explanation.

Removing each of the ten most frequent short-pair proteins or recorded family witnesses does not eliminate the descriptive all-DEV gap. This is limited reassurance: the DEV-only FADI table matches all 3,711 proteins by exact sequence hash but contains just one TRAIN-shared Pfam witness per protein, not complete family annotations. Unknown witnesses are not evidence of family novelty. Complete domain/family breadth remains unresolved.

## Compression and missing diagnostics

The current 256→32 random projection is followed by mean, standard deviation, maximum and 95th percentile over every valid inter-chain cell, then TRAIN-fitted PCA to 32 dimensions. **Maximum pooling is already present.** All four statistics discard location and spatial coherence. A sparse interface may be diluted in the mean or fail to shift the 95th percentile; a maximum is sensitive to isolated extremes and the number of cells searched. Top-k cells or log-sum-exp alone would still discard geometry.

This is a plausible bottleneck, not an established cause. Standardized true–shuffle feature perturbations remain substantial at longer lengths; they do not disappear monotonically. The earlier audit's 81% preservation of DEV perturbation energy by PCA concerns already-pooled features, not information lost before pooling or task-relevant information. Its 1B70 contact positive control establishes runtime pairing sensitivity, not Bernett classification validity. The released [contact head](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/regression.py) reads each residue pair before spatial aggregation; its supervision must remain outside this follow-up.

The Neff audit is especially relevant: **all 4,453 eligible pairs have Neff equal to retained depth**, because selection and reweighting use the same 90%-identity predicate. This is consistent with the implemented definition, but is not an independent diversity measurement. Depth hits the 127 cap for 94.1% of eligible TRAIN and 89.7% of eligible DEV; the gate's depth factor is saturated for 98.8% of eligible DEV.

Add diagnostic logging, without changing eligibility, the gate, or quality-head inputs:

- Cached common-accession depth, candidate depth after coverage/conflict filters, retained depth, and cap saturation; distinguish cached depth from unrestricted archive depth.
- Separately declared 80%-identity, gap-aware reweighting on retained rows for each chain and the concatenation, with comparison-coverage distributions and taxonomic diversity. These remain operational Neff measures, not ground-truth evolutionary independence.
- Valid residues per chain, valid cell area, length asymmetry, per-column occupancy, jointly nongap paired support, and Neff per valid query residue. A quality-mask fraction alone is not effective paired coverage.
- Actual sequence/token changes under the null, not only moved row indices; within-family/order/class counts and conservation. The previous actual-token audit covered only three pairs.
- TRAIN-overlap and complete query protein family/domain annotations where available; distinguish these from the homologs' taxonomic families. Report group sizes, label prevalence, protein degree, family influence and uncertainty. Keep missing annotations explicit.

## Smallest useful next experiment

**Hypothesis:** coherent local patches in the frozen inter-chain representation contain useful pairing-dependent information that global distribution summaries lose.

Keep the same 4,000 TRAIN / 4,000 DEV sample, eligibility, native baseline logits, selected MSA rows, hierarchical null/permutation seeds, precision, layer, projection, AB/BA averaging, gate, 68-dimensional quality control, and TRAIN-only StandardScaler → PCA(32) → logistic regression C=0.1. Keep the same calibration-only alpha grid and assessment membership. Do not optimize a ≤512 subset, increase depth, tune fusion beyond its current upper boundary, or change multiple axes together.

Test **one** replacement 128-dimensional readout: retain the 32 global means and 32 standard deviations; replace max/q95 with the means of the highest four and lowest four local block means in each projected channel. Partition each chain's original coordinates into ceil(length/8) balanced contiguous blocks, giving approximately 8×8 cells per block pair. Average only valid cells; retain block pairs with at least 50% valid positions in each chain block. Use all qualifying blocks if fewer than four exist. Freeze these choices before new outcomes. Do not collapse masked coordinates or run the encoder on sequence crops. Current per-chain ≥50% quality guarantees at least one qualifying block in each chain; an unexpected empty/nonfinite result is an execution error.

Compute an additional deterministic spatial control from the same forward tensors: permute valid residue indices independently within each chain after encoding, identically across channels, before local pooling. This preserves the multiset underlying the old global summaries while destroying sequence adjacency. Use the same permutation for true and shuffled views, respecting reversed orientation. Fit matched local heads for true/shuffled with and without this spatial scramble: at most four small classifiers, no attention, CNN, trunk gradients, or hyperparameter search. Also evaluate each true-trained head on its corresponding shuffled input without refitting.

Record the added MSA diagnostics during this extraction. They are reporting variables, not new predictive features. Freeze a separate follow-up manifest and retain the original artifacts. Missing computational outputs make the run inconclusive; only already-defined biological unavailability permits exact native-baseline fallback. Corrupt caches, changed inputs, nonfinite values, and budget exhaustion must retain fail-closed behavior.

## Decision and cost

Predeclare a primary paired bootstrap contrast on the existing assessment: the local true–shuffle AP gap minus the original global true–shuffle AP gap. A convincing useful-readout result requires its 95% interval above zero, a positive local true–shuffle gap, and an improvement of local true over original true; report an interval for each. Retain the original practical continuation checks: ≥0.010 AP over native baseline, positive true-minus-baseline/shuffled/quality intervals, AUROC decline no greater than 0.005, adequate null change and credible family breadth. A larger gap obtained only by degrading the shuffled model is not success.

For the stronger **locality** explanation, the pairing gap should also be larger with real spatial order than with scrambled order, with a matched interval, and true-head counterfactual predictions should favor true pairing. If spatial scrambling preserves the gain, any pairing benefit cannot be attributed specifically to local coherence. Report assessment-wide results first, then length, depth/coverage and family strata; do not promote the most favorable subgroup to the primary endpoint. Common-prevalence AP and AUROC can accompany subgroup AP to expose prevalence effects. Use matched protein bootstraps and complete-family sensitivity analyses where annotations support them.

Equal improvement of true and shuffled models fails the pairing hypothesis. A precise nonpositive contrast fails this readout hypothesis; a wide interval is inconclusive, not proof that coevolution is absent. Either outcome stops this proposed ablation without an automatic model search or larger run. Reusing an already-inspected assessment remains exploratory; even a positive result does not create a new blind validation or authorize TEST. A pairing-dependent gain would still not prove direct compensatory coevolution: the null controls only coarse phylogeny, and biological partner assignments remain incompletely verified.

Implementation is modest, but dense tensors were not retained: the 4,453 eligible examples need true/shuffled AB/BA encoder extraction again. The original batch used 8.39 allocated GPU-hours; roughly **8–10 additional allocated GPU-hours** is a planning estimate for the same extraction plus cheap pooling controls, subject to measured qualification and the original cumulative 24-hour ceiling. CPU head fitting is small; Neff/coverage logging needs profiling and the existing CPU/storage ceilings still apply. Reuse the existing SIF, write only compact features/diagnostics, account for failures, and do not automatically extend the budget. This single ablation is justified by an identifiable, testable information bottleneck—not by a demonstrated short-length effect.
