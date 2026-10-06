# The three fixed V11 baseline methods on the v5 data

Completed: 2026-10-06T08:12:28.477318+00:00

The three requested baseline methods were fitted afresh on **v5 TRAIN only**, with the exact configuration used in the previous V11 study. Full v5 validation, original Bernett test and custom ILP-negative test were scored. The previous V11 fitted regressor and graph were not reused. Existing v5 iPIN and native Bernett predictions were reused; no neural model was retrained or rerun.

## Findings

**All three fixed baselines are near the 0.5 reference on v5 validation and both tests, while the reused iPIN models score substantially higher. This contrasts with their V11 results and is consistent with the different degree balance and protein partitioning. It is evidence about these tested mechanisms, not proof that v5 contains no shortcuts.**

TRAIN positive and negative degrees are equal for 13,087/13,110 proteins (99.82%). Only 23 proteins retain a degree difference. The log-degree-ratio target has standard deviation 0.015776518 and range [-0.45685651, 0.69314718]. These measured targets were retained without modification. Both propensity methods predict this nearly constant target; they are not general-purpose sequence pair classifiers.

On Original Bernett test, the highest AP among the three requested methods is 0.5034 and the highest AUROC is 0.5057; these may belong to different methods and are descriptive, not a selected ensemble. iPIN ESM2 scores 0.6917/0.6964 (AP/AUROC), and ESMC scores 0.6871/0.6981.
On V5 ILP test, the highest AP among the three requested methods is 0.5022 and the highest AUROC is 0.5023; these may belong to different methods and are descriptive, not a selected ensemble. iPIN ESM2 scores 0.6570/0.6669 (AP/AUROC), and ESMC scores 0.6452/0.6637.

The main metric tables report every method separately. The exact TRAIN-degree lookup is the same reference control used previously and is constant on all three evaluations because their proteins are absent from TRAIN.

## Data

| Split | Pairs | Positives | Negatives | Distinct sequences |
|---|---:|---:|---:|---:|
| train | 700,764 | 350,382 | 350,382 | 13,110 |
| validation | 165,742 | 82,871 | 82,871 | 6,023 |
| original | 52,048 | 26,024 | 26,024 | 3,022 |
| ilp | 52,048 | 26,024 | 26,024 | 2,948 |

The three protein groups TRAIN, validation and test are exactly sequence-disjoint. All splits contain unique, non-self unordered sequence pairs. No sequences were truncated and no rows were dropped or resampled. Both tests share all 26,024 positives and 1,154 negatives. All 27,178 shared pairs were verified to have identical scores under every compared predictor.

## Average precision

| Method | V5 validation | Original Bernett test | V5 ILP test |
|---|---:|---:|---:|
| Exact TRAIN degree (reference) | 0.5000 | 0.5000 | 0.5000 |
| Homology-transferred degree | 0.5003 | 0.5033 | 0.5007 |
| Sequence-only propensity | 0.4996 | 0.5016 | 0.5006 |
| Interolog lookup | 0.5005 | 0.5034 | 0.5022 |
| iPIN v5 ESM2 | 0.6153 | 0.6917 | 0.6570 |
| iPIN v5 ESMC | 0.5925 | 0.6871 | 0.6452 |
| PLM-interact (Bernett) | — | 0.6903 | 0.6377 |

## AUROC

| Method | V5 validation | Original Bernett test | V5 ILP test |
|---|---:|---:|---:|
| Exact TRAIN degree (reference) | 0.5000 | 0.5000 | 0.5000 |
| Homology-transferred degree | 0.5006 | 0.5057 | 0.5009 |
| Sequence-only propensity | 0.4995 | 0.5016 | 0.5004 |
| Interolog lookup | 0.5004 | 0.5035 | 0.5023 |
| iPIN v5 ESM2 | 0.6133 | 0.6964 | 0.6669 |
| iPIN v5 ESMC | 0.5951 | 0.6981 | 0.6637 |
| PLM-interact (Bernett) | — | 0.6995 | 0.6504 |

All datasets are balanced, so reference AP and AUROC are 0.5. Raw AP cannot be compared directly with V11 or its five-species tests, which use about 1:10 positives/negatives. Native Bernett has no cached predictions on this v5 validation set; its historical validation set was not substituted. iPIN validation metrics were used to select their checkpoints and are not an independent estimate of test performance.

## Methods held fixed

See the [frozen protocol](PROTOCOL.md) and [inherited V11 protocol](provenance/original-v11-protocol.md). The source implementation was checked against the completed V11 artifact manifest. The full configuration, degree-counting routine, sequence features, gradient boosting model, homology transfer rule, interolog rule and bootstrap implementation are unchanged. Paths and data adapters were updated for this study.

- **Homology-transferred degree:** add the two proteins' positive/negative TRAIN log-degree propensities, transferred through up to five TRAIN sequence matches; exact matches use stored propensity and no-hit queries use the mean TRAIN propensity.
- **Sequence-only propensity:** the same 422 log-length, amino-acid, ordered-dipeptide and unknown-residue features; 300 iterations of a 15-leaf histogram gradient boosting regressor; predict the same TRAIN log-degree propensity and add the endpoint predictions.
- **Interolog lookup:** maximum product of sequence-match weights over retained homologs connected by a v5 TRAIN-positive edge. No supporting edge scores zero; all such pairs remain in evaluation.

MMseqs2 thresholds remain 25% identity, 50% query and target coverage, E-value 1e-5, sensitivity 7.5, and up to five retained hits. This search can find weaker or partial similarities that survived the v5 exclusion criterion of 40% identity over 80% of both sequences. No claim of complete absence of all homology is made. No validation/test label or test-specific annotation was used to fit these methods.

## TRAIN degree diagnostic

There are 13,110 TRAIN proteins, of which 13,087 have equal positive/negative row-occurrence counts. The sum of absolute degree differences is 7,178; maximum absolute difference is 1,294. The smoothed log ratio uses +1 in both counts. Its mean is 0.00010136396, standard deviation 0.015776518, and it has 22 unique values. These counts come exclusively from the final actual TRAIN pairs, not the solver objective or requested tolerances.

[training-degrees.csv.gz](results/training-degrees.csv.gz) contains every TRAIN protein's counts and target. Uniform protein weighting, seed 47 and all other training parameters were retained even if the target carries little variation.

## Homolog and interolog coverage

| Dataset | Proteins with retained TRAIN match | Pairs with both endpoints matched | Positive pairs with interolog support | Negative pairs with interolog support |
|---|---:|---:|---:|---:|
| V5 validation | 209/6,023 | 529/165,742 | 115/82,871 | 41/82,871 |
| Original Bernett test | 773/3,022 | 3,565/52,048 | 200/26,024 | 20/26,024 |
| V5 ILP test | 747/2,948 | 3,722/52,048 | 200/26,024 | 81/26,024 |

Coverage is descriptive and computed after scores were frozen. Low interolog coverage can make many scores tie; this limitation is visible rather than removed by evaluating only covered pairs.

Interolog lookup retains a small detectable residual: its AUROC intervals are just above 0.5, although the absolute improvement is tiny. Only 200 of the 26,024 shared test positives have supporting retained TRAIN-homolog edges (about 0.77%). Thus near chance does not mean exactly zero signal. The degree and sequence-propensity AUROC intervals include 0.5 on every split.

## Uncertainty and interpretation

The [95% intervals](results/confidence-intervals.csv) and [paired differences](results/paired-differences.csv) use 500 protein-endpoint bootstrap replicates and seed 20261006, identical to the V11 baseline protocol. Every model receives the same sampled endpoint multiplicities within each dataset. Self pairs would receive one multiplicity; other pairs receive their product. Intervals are descriptive and unadjusted for multiple comparisons. They do not cover training-seed, phylogenetic-family or data-construction uncertainty. No across-test significance calculation treats the shared positives as independent.

For all three requested baselines versus each iPIN model, the paired 95% intervals for baseline-minus-iPIN performance lie below zero in both AP and AUROC on every evaluated split. Validation comparisons concern the already selected iPIN checkpoints; the two test comparisons share positives.

A low score here shows that these particular baseline mechanisms do not recover the strong V11 signal under this preparation. It does not establish that v5 is unbiased or that iPIN learned binding-site compatibility. In particular, degree-target regression is not a general test of every sequence-only classifier. The V11/v5 comparison changes the training graph, proteins, partitioning, negative construction and prevalence simultaneously, so it cannot isolate the causal contribution of any one change.

## Computation and verification

All new computation was CPU-only inside the current interactive allocation, using at most 16 threads. Search wall time: 108.13 s. Regressor fitting: 1.17 s; feature construction: 0.36 s; prediction for all proteins: 0.03 s. Both homology methods share one search. No production SLURM job or new neural inference was submitted.

Evaluation was performed twice after eliminating repeated decompression during CSV export. The first run completed normally; neither the models nor frozen predictions changed. All first-run point metrics, retained in its log, were exactly reproduced. Both runs' resource records and the export-optimization provenance are retained; this was an output-speed fix, not another scientific configuration.

Source bytes and both backbone TRAIN/validation arrays were verified. Cached neural predictions were checked against exact sequence-pair identity, labels and source row order, then their existing metrics were reproduced. Baseline checks include independent TRAIN counts, finite complete predictions, exact AB/BA symmetry, a scalar interolog reference, independent AP/AUROC calculations including weighted bootstrap checks, and shared-test-pair concordance. The container and MMseqs2 binary are pinned by SHA256.

## Files and reproduction

- [Comparison figure](results/comparison.png), with PDF/SVG copies; [metrics CSV](results/metrics.csv); [summary JSON](results/summary.json).
- results/*-predictions.csv.gz contains source row numbers, sequence hashes, labels, all scores and support diagnostics. Machine-readable NPZ scores are also saved.
- models/ contains the fitted regressor, the v5 TRAIN-positive graph, degree/propensity arrays and retained homologs. work/alignments.tsv stores the filtered search output.
- inputs/ archives the exact source files; provenance/ records source checksums, inherited methods, configuration and the pre-evaluation prediction freeze.
- logs/*.resources.txt records elapsed time, CPU time and peak memory. results/COMPLETE.json is the final artifact manifest.

From the project root:

```bash
bash benchmark-v5-baselines/scripts/run.sh
```

Completed preparation, search and fitting stages verify and reuse their artifacts. Evaluation reads the frozen predictions. The earlier V11 study and existing v5 benchmarks are not modified.
