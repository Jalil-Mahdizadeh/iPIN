# Three fixed CPU baselines on the exact V11 human and five-species data

Completed: 2026-10-05T22:01:35.634220+00:00

**The inexpensive baselines recover substantial cross-species signal, but all three fall below both native PLM-interact humanV11 and TUnA human seed 47 on every species in both AP and AUROC. These results do not support claiming that these baselines match the expensive models across the five species.**

The strongest fixed baseline by five-species mean is homology-transferred degree: AP 0.5975, AUROC 0.8591, compared with native PLM-interact AP 0.8327, AUROC 0.9525 and TUnA AP 0.7784, AUROC 0.9371. This is a descriptive comparison of the three prespecified methods, not selection of a new predictor or a per-species ensemble.

The defensible message is narrower: a CPU method that scores the proteins separately reaches AUROC above 0.90 on mouse, fly and worm, so learning partner-specific compatibility is not required for substantial performance on these tests. This does not measure which mechanisms the neural models use, or establish that all predictive protein-level signal is a sampling artifact. The much stronger human-validation lookup result and the remaining cross-species gap should both be reported.

Human TRAIN contains 421,792 pairs (38,344 positive; 383,448 negative) and 15,631 distinct sequences. Human validation has 52,725 rows. The five nonhuman tests have 242,000 rows in total. The release called human_test is the human validation split. Every released row is preserved in the primary results.

One fixed configuration was evaluated for each requested method. No validation or species-test labels were used for fitting or configuration selection. The original exact-sequence degree lookup is included as a reference control. Existing native PLM-interact humanV11 and TUnA human seed-47 predictions were reused and their source-row mapping and hashes verified. No neural model was retrained or rerun.

## Average precision

| Method | Human validation | Mouse | Fly | Worm | Yeast | E. coli | Five-species mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact TRAIN degree (reference) | 0.8360 | 0.1590 | 0.0911 | 0.0909 | 0.0909 | 0.0909 | 0.1046 |
| Homology-transferred degree | 0.8360 | 0.6675 | 0.7223 | 0.6818 | 0.4521 | 0.4638 | 0.5975 |
| Sequence-only propensity | 0.6518 | 0.4854 | 0.4262 | 0.4821 | 0.3599 | 0.5342 | 0.4576 |
| Interolog lookup | 0.4013 | 0.7224 | 0.6319 | 0.5419 | 0.3387 | 0.2141 | 0.4898 |
| PLM-interact humanV11 | — | 0.9084 | 0.9168 | 0.8909 | 0.7147 | 0.7329 | 0.8327 |
| TUnA human seed 47 | — | 0.8832 | 0.8550 | 0.8368 | 0.6441 | 0.6730 | 0.7784 |

## AUROC

| Method | Human validation | Mouse | Fly | Worm | Yeast | E. coli | Five-species mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Exact TRAIN degree (reference) | 0.9635 | 0.5367 | 0.5011 | 0.5000 | 0.5000 | 0.5000 | 0.5076 |
| Homology-transferred degree | 0.9635 | 0.9203 | 0.9360 | 0.9033 | 0.7914 | 0.7443 | 0.8591 |
| Sequence-only propensity | 0.9175 | 0.8370 | 0.8203 | 0.8429 | 0.7960 | 0.7879 | 0.8168 |
| Interolog lookup | 0.6735 | 0.8484 | 0.8009 | 0.7502 | 0.6386 | 0.5677 | 0.7212 |
| PLM-interact humanV11 | — | 0.9830 | 0.9834 | 0.9753 | 0.9144 | 0.9066 | 0.9525 |
| TUnA human seed 47 | — | 0.9774 | 0.9710 | 0.9602 | 0.8992 | 0.8775 | 0.9371 |

Chance AP is the positive prevalence: 0.090925 for human validation and 0.090909 for each species. Chance AUROC is 0.5. The five-species mean is an unweighted mean of the five separate metrics, not a pooled ranking. Dashes indicate unavailable exact neural human-validation predictions; values from a different split or paper were not substituted.

The equal human-validation scores for homology-transferred degree and exact TRAIN degree are expected: every validation sequence is already in TRAIN, and exact matches use their stored TRAIN propensity. Those two human columns are not independent evidence of transfer. The sequence-only model uses no identity lookup and still reaches human-validation AUROC 0.9175; its validation proteins are nevertheless also present among its fitting examples.

## Methods and interpretation

The [frozen protocol](PROTOCOL.md) contains all parameters and limitations. TRAIN-degree propensity is log((positive occurrences + 1)/(negative occurrences + 1)), with a self pair counted once toward its endpoint. Both requested protein-propensity models add two separately computed protein scores, so neither learns partner-specific compatibility.

- **Homology-transferred degree:** use the exact TRAIN propensity if available; otherwise average propensities from up to five human TRAIN sequence matches, weighted by identity and alignment coverage. No match uses the mean TRAIN protein propensity.
- **Sequence-only propensity:** a 300-iteration, 15-leaf histogram gradient boosting regressor predicts the TRAIN propensity from 422 length/composition/dipeptide features. The target is protein sampling propensity, not a calibrated pair interaction probability. It uses no PLM embeddings.
- **Interolog lookup:** maximum product of two sequence-similarity weights over human TRAIN-positive edges connecting the retained homologs. No supporting edge gives zero. Conservation is legitimate biological information; this method is not a pure shortcut diagnostic.

The alignment thresholds were fixed at E-value 1e-5, identity 25%, and 50% coverage of both proteins, with MMseqs2 sensitivity 7.5 and up to five retained hits. All missing-hit pairs remain in the denominator. Alignment hits are not assertions of orthology. Top-five truncation and fixed thresholds limit this particular implementation.

Strong performance of the first two methods would show that independent protein information can explain much of this benchmark. Failure of these fixed implementations does not establish that the data are unbiased or that stronger cheap baselines cannot work. The neural five-species results were already known before this study, so this is an exploratory assessment; baseline settings were nonetheless frozen before their scores were evaluated.

## Coverage

| Dataset | Exact human TRAIN endpoint in pair | Both endpoints have a retained human hit | Interolog score > 0: positives | Interolog score > 0: negatives |
|---|---:|---:|---:|---:|
| Human validation | 52,725 / 52,725 | 52,725 / 52,725 | 1,667 / 4,794 | 41 / 47,931 |
| Mouse | 2,689 / 55,000 | 48,476 / 55,000 | 3,487 / 5,000 | 51 / 50,000 |
| Fly | 142 / 55,000 | 19,773 / 55,000 | 3,013 / 5,000 | 50 / 50,000 |
| Worm | 0 / 55,000 | 10,071 / 55,000 | 2,505 / 5,000 | 39 / 50,000 |
| Yeast | 0 / 55,000 | 8,111 / 55,000 | 1,391 / 5,000 | 50 / 50,000 |
| E. coli | 0 / 22,000 | 888 / 22,000 | 271 / 2,000 | 0 / 20,000 |

## Sensitivity analyses and uncertainty

Removing every pair containing an exact human TRAIN sequence leaves 52,311 mouse pairs (4,001 positives). Homology-transferred degree still achieves AP 0.6085 and AUROC 0.9116, versus native PLM-interact 0.8886/0.9811 and TUnA 0.8564/0.9749. Fly changes negligibly (homology degree AP 0.7228, AUROC 0.9360 on 54,858 remaining pairs). Worm, yeast and E. coli already have no exact human TRAIN endpoint matches. Thus the cross-species signal is not explained solely by exact sequence reuse.

All paired 95% protein-bootstrap intervals for each of the three requested baselines minus either neural comparator remain below zero for both metrics in every species. This supports the observed performance gap under this resampling scheme, with the dependence and multiple-comparison limitations stated below.

After deduplicating E. coli and excluding contradictory-label pairs, 18,228 pairs remain, including 1,999 positives. The sequence-only model has AP 0.5628 and AUROC 0.7878, versus native PLM-interact 0.7545/0.9072 and TUnA 0.6963/0.8770. The higher positive prevalence partly changes AP; the ranking conclusion persists.

[subsets.csv](results/subsets.csv) applies the same subsets to every compared method: exclude exact TRAIN-pair overlaps; exclude every nonhuman pair containing an exact human TRAIN sequence; and retain unique unordered pairs after removing label conflicts. These subsets have different sizes and sometimes different prevalence, so their AP values should not be compared across cohorts as though the tasks were identical. The primary tables preserve the original duplicate rows, including the substantial E. coli duplication.

[confidence-intervals.csv](results/confidence-intervals.csv) contains 95% percentile intervals from 500 paired protein-bootstrap samples. [paired-differences.csv](results/paired-differences.csv) compares each baseline to each neural reference on the same resampled endpoints. Endpoint resampling addresses shared proteins, but not all homolog-family or phylogenetic dependence. Intervals are exploratory, unadjusted for multiple comparisons; they do not establish equivalence. Protein-unseen performance cannot be inferred from human validation, whose proteins all appear in TRAIN.

## Computation and verification

All new computations used CPU cores within the existing interactive GH200 allocation, with at most 16 threads and no GPU computation. Measured sequence-search wall time: 183.7 s. Regressor fitting: 1.4 s; prediction for all proteins: 0.1 s; sequence-feature construction: 1.1 s. Full fitting, scoring, and symmetry audits: 8.0 s; preparation: 11.8 s. These are timings on this node, not portable cost guarantees or a complete cost comparison with the historical neural training runs.

Detailed elapsed time, CPU time and peak resident memory are in logs/*.resources.txt. Two homology methods share one search. The installed module is named mmseqs2-gpu but was explicitly run with --gpu 0; its binary SHA256 is recorded. Python used the existing ARM64 SIF with GPU passthrough disabled. The environment, fitted model, protocol hash, scripts, TRAIN-only inputs, and pre-evaluation prediction hashes are recorded in provenance/.

Checks passed: released-source hashes; all neural/source pair mappings; independent TRAIN count implementation; finite full-coverage scores; exact endpoint-reversal symmetry; independent scalar interolog checks; independent AP/AUROC including weighted checks; reproduction of earlier human degree metrics and existing neural species metrics. Worm, yeast and E. coli exact lookup scores are constant and give AUROC 0.5 and AP 1/11.

## Artifacts and reproduction

- [Metric table](results/metrics.csv), [summary JSON](results/summary.json), [comparison figure](results/comparison.png), and vector PDF/SVG versions.
- results/*-predictions.csv.gz preserve source row numbers, exact sequence hashes, labels, scores and homolog diagnostics; NPZ files retain machine-readable score arrays.
- models/ contains the fitted regressor, TRAIN graph, protein scores/counts and retained homologs. work/alignments.tsv contains the full filtered search output.
- inputs/ contains lossless archives of all seven original data files; provenance/prepared.json records input hashes, URLs and reused neural outputs.
- results/COMPLETE.json is the final artifact manifest, created only after evaluation and report generation finish.

From the project root:

```bash
bash literature/V11-baselines/scripts/run.sh
```

Completed preparation, search and predictions are verified and reused. Evaluation can be regenerated from frozen scores. GPU inference is never invoked.

Method references: [MMseqs2 documentation](https://github.com/soedinglab/MMseqs2/wiki); [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html); [Bernett et al. 2024](https://doi.org/10.1093/bib/bbae076).
