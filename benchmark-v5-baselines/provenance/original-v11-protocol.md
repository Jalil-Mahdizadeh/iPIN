# Fixed, inexpensive baselines for the released STRING V11 benchmark

This protocol is recorded before fitting these baselines or calculating their five-species results. The five-species neural-model results have already been inspected in the parent project; this is an exploratory audit, not a newly blinded benchmark. No baseline settings will be selected using species test labels. There is one configuration per method, with no validation or test parameter search.

## Data and comparisons

- Use the exact released human TRAIN (421,792 rows) and human validation (52,725 rows), already independently shown to be identical for native PLM-interact humanV11 and TUnA human seed 47. Validation labels never enter fitting or TRAIN graph construction.
- Preserve the original rows, sequence strings, labels, ordering, duplicates, and conflicts of the five released nonhuman tests (242,000 rows total). Sequence identity means exact amino-acid string equality.
- Reuse the existing native humanV11 and TUnA human seed-47 nonhuman predictions, verifying hashes and pair alignment. Do not run new neural inference. Neural human-validation metrics will be left unavailable unless exact existing predictions are found.
- Include the original exact-sequence TRAIN-degree lookup as a reference control, in addition to the three requested baselines.

## TRAIN information

For each distinct human TRAIN sequence, count its occurrences in positive and negative TRAIN rows; a self pair contributes once to its endpoint. Define `r(p) = log((d_positive(p)+1)/(d_negative(p)+1))`. Preserve TRAIN row multiplicity for counts. Build the interolog graph from distinct unordered TRAIN-positive sequence pairs; if TRAIN contains conflicting labels, a known positive is still in this graph. No validation or test labels, external interaction network, annotation, or protein language model is used for fitting.

## Three fixed methods

1. **Homology-transferred degree.** Search query sequences against human TRAIN sequences with MMseqs2 on CPU: sensitivity 7.5, E-value <= 1e-5, identity >= 0.25, query and target coverage >= 0.5, at most 1,000 prefilter candidates, one search iteration. Retain up to five hits ordered by descending bit score, descending similarity weight, then sequence identifier. A hit has weight `w = identity * sqrt(query_coverage * target_coverage)`. If a query has an exact TRAIN sequence match, use that protein's `r` directly. Otherwise use the weighted mean `r` of its retained hits. If there is no hit, use the mean `r` over distinct human TRAIN proteins. Add the two endpoint values for the pair score. The exact lookup reference instead uses `r=0` for an unseen sequence, reproducing the earlier +1 rule.
2. **Sequence-only protein propensity.** Predict `r(p)` from 422 features: natural log sequence length, 20 amino-acid fractions, 400 ordered dipeptide fractions, and the fraction of noncanonical residues. Fractions use full sequence length (dipeptides: length minus one); noncanonical-containing dipeptides contribute to no canonical bin. Fit one scikit-learn HistGradientBoostingRegressor on distinct TRAIN proteins, uniformly weighted: squared error, 300 iterations, learning rate 0.05, 15 leaves, minimum 20 samples per leaf, L2=10, 255 bins, early stopping disabled, seed 47. Pair score is `g(A)+g(B)`; there is no partner-specific term or exact-identity override.
3. **Interolog lookup.** Use the same up-to-five human homologs per query, including an exact match with weight 1 as the first hit where available. Score a query pair by the maximum `w(A,u)*w(B,v)` over retained human homologs whose unordered pair is a TRAIN-positive edge. Score zero if no supporting edge exists. Both orientations must yield identical scores. This uses conserved pair evidence, unlike the two partner-independent methods above.

Exact hits are explicitly inserted even if low-complexity masking prevents their retrieval. Nonexact search hits are not orthology assertions. Top-five truncation and the fixed similarity thresholds are limitations of these deliberately small baselines, not assertions about the best possible alignment model.

## Evaluation and safeguards

- Compute AP and AUROC for all methods on all source rows; do not discard pairs lacking homologs. Report protein and pair homolog coverage, supporting-edge coverage, score ties, and positive prevalence. Scores are ranking scores, not calibrated probabilities.
- Verify row/prediction mapping, finite scores, exact-lookup reproduction of the prior human validation result, invariance to endpoint reversal, TRAIN-only graph/target construction, and independent ranking-metric calculations.
- Secondary, identically applied subsets: remove exact TRAIN-pair overlaps; remove all pairs containing an exact human TRAIN sequence (nonhuman tests only); retain one row per unordered pair after excluding contradictory-label pairs. Report their different sizes/prevalences and do not substitute them for the primary results.
- Use 500 paired protein-bootstrap replicates with seed 20261006 for full-cohort 95% intervals and differences versus the two neural comparators. Resample distinct endpoints; a pair has the product of endpoint multiplicities, with one multiplicity for self pairs. This addresses shared endpoints but is not a phylogenetic uncertainty model. Species-macro point estimates average the five species equally; do not pool different species into one ranking.
- Archive scripts, configuration, provenance, source hashes, per-row scores, learned regressor, homolog lists, metrics, plots, and measured CPU resource use. Use at most 16 CPU threads. No GPU computation or new neural training.

## Interpretation fixed before results

Strong protein-propensity results would show that partner-independent information suffices for much of this benchmark. Strong interolog results would establish a cheap alternative, potentially exploiting real conservation. Failure of these specific baselines would not establish that the dataset is unbiased, nor exclude stronger inexpensive methods. Human validation and cross-species findings will be stated separately. Do not conclude that STRING V12.5 inherits V11's difficulty without constructing and assessing its actual benchmark.

Implementation references: [MMseqs2 user guide](https://github.com/soedinglab/MMseqs2/wiki); [scikit-learn HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html); [Bernett et al. 2024](https://doi.org/10.1093/bib/bbae076).
