# The three fixed V11 baseline methods, refitted using v5 TRAIN

This protocol is frozen before fitting or evaluating these methods on v5. The existing neural v5 results are already known, so this is an exploratory baseline audit, not a newly blinded confirmation. Only v5 TRAIN labels are used for counts, graph construction and regressor fitting. No baseline settings, signs, thresholds, or checkpoints are selected using validation or either test.

## Data and scope

- Use the exact final `data-preparation-v5` data consumed by both iPIN models: 700,764 TRAIN rows (350,382 positives), 165,742 validation rows (82,871 positives), original Bernett test and custom ILP-negative test (52,048 rows / 26,024 positives each).
- Preserve all full sequences and source rows. No length cap, resampling, new negative construction, or filtering is applied. Verify exported pairs against both backbone training arrays and the existing benchmark arrays.
- The tests share all 26,024 positives and 1,154 negatives. They are two negative-distribution evaluations, not independent replications. Check that all shared pairs receive identical scores.
- Copy the previous scientific implementation and configuration from `literature/string-v11-baselines` (the former `literature/V11-baselines`). Verify its original manifest and archive source hashes. Only paths, dataset names, preparation and evaluation adapters change.
- Include the original exact-degree lookup as a reference control. It is not a fourth newly proposed method.

## Identical baseline definitions

For each distinct TRAIN sequence, compute `r = log((positive TRAIN row occurrences + 1)/(negative TRAIN row occurrences + 1))`. A self pair would contribute once to its endpoint. TRAIN row multiplicity is retained. The interolog graph contains distinct unordered TRAIN-positive sequence pairs. Validation and test labels never enter these quantities.

1. **Homology-transferred degree:** exact TRAIN matches use their `r`; otherwise transfer the weighted mean `r` from up to five human TRAIN matches. MMseqs2 runs on CPU with sensitivity 7.5, E-value <= 1e-5, identity >= 25%, coverage >= 50% of both proteins, 1,000 prefilter candidates, and one iteration. Rank matches by bit score, similarity weight, then exact sequence hash. The weight is `identity * sqrt(query_coverage * target_coverage)`. No match receives mean TRAIN protein `r`. Pair score adds the two endpoint scores.
2. **Sequence-only propensity:** predict `r` using the same 422 log-length, amino-acid, dipeptide and unknown-residue features. Use the same HistGradientBoostingRegressor: squared error; 300 iterations; learning rate 0.05; 15 leaves; minimum 20 proteins per leaf; L2=10; 255 bins; no early stopping; seed 47; uniform protein weighting. Pair score adds the two predictions. No PLM embedding or exact-identity override.
3. **Interolog lookup:** maximum product of weights for the retained homolog pairs connected by a positive v5 TRAIN edge. Exact matches are included with weight 1. No supporting edge receives zero. Retain every query pair regardless of homolog coverage.

For the exact-degree reference, an unseen sequence has `r=0`, as in V11. Exact matches are explicitly inserted if masking prevents their alignment retrieval. Do not replace, perturb, or expand a target that becomes nearly constant under degree-balanced v5 negatives. Report its measured variation instead. The fixed 25%/50% search can retrieve weaker or partial homologs remaining after v5's 40%-identity/80%-both-coverage exclusion; this is intentional reuse of the same baseline definition, not a claim that all homology was eliminated.

## Evaluation

- Evaluate all four score functions on full v5 validation and both tests. AP and AUROC have reference value 0.5 because all three datasets are balanced. Reuse the selected v5 ESM2 and ESMC full validation predictions and both test predictions; reuse native PLM-interact Bernett test predictions. Do not substitute its different historical validation set for v5 validation.
- Treat iPIN validation metrics as checkpoint-selection performance, not a fresh unbiased test estimate. No neural inference or training is run. Other historical benchmark reports remain unchanged.
- Verify protocol/configuration identity, data hashes, TRAIN-only fitting, full coverage, pair-order symmetry, independent metric implementations, reused prediction mapping, and identical scores on shared test pairs. Save per-row scores, fitted model, search output, retained hits and computational costs.
- Report TRAIN degree residuals and target variation, all split exact-sequence overlaps, homolog coverage, positive/negative interolog support, and score ties. Missing-hit pairs remain in every primary metric denominator.
- Use the same 500 paired protein-endpoint bootstrap replicates and seed 20261006. Report 95% percentile intervals and baseline-minus-iPIN/native differences on identical rows. Self pairs receive one multiplicity; other pairs receive the product of endpoint multiplicities. These descriptive intervals do not measure training-seed, homolog-family or dataset-construction uncertainty, and are not corrected for multiple comparisons. Do not treat the two tests as independent evidence or select a winner between their negatives.
- Because v5 should have disjoint sequences and unique pairs, verify these invariants directly. If any fails, stop and investigate rather than silently remove rows. No new data audit or model sweep is added.

## Interpretation

Near-chance baseline performance would show that these particular methods do not recover V11-like signal from this v5 preparation. It would not establish that v5 is free of bias or prove that a neural model learned molecular binding mechanisms. Any cross-study comparison must note that V11 and v5 differ in training graph, sequences, splits, negatives and class balance. Raw AP is not comparable across their 1:10 and 1:1 prevalences.

All new computation uses at most 16 CPU threads in the current interactive allocation. There is no GPU computation and no production retraining submission. Preserve all results in this folder.
