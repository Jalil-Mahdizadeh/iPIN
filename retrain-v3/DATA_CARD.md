# V3 data card

Prepared inputs are independent **byte-identical copies of the qualified v2 train/validation inputs**. No rows have been added, relabeled, resampled, capped or re-partitioned for the initial six runs. The [manifest](data/prepared/manifest.json) remains SHA-256 `1f5adc6e656390a298ec93173134914b90896c2198bcaf672fe42ea7c81e8f12`.

| Population | Train pairs | Train positives | Validation pairs | Validation positives | Crossing pairs excluded |
| --- | ---: | ---: | ---: | ---: | ---: |
| Official historical | 163,085 | 81,550 | 59,258 | 29,628 | N/A |
| Internal fold 0 | 73,691 | 36,897 | 17,490 | 8,732 | 71,904 |
| Internal fold 1 | 71,698 | 35,923 | 18,285 | 9,048 | 73,102 |
| Internal fold 2 | 71,986 | 35,765 | 18,515 | 9,255 | 72,584 |

Each fold partitions the official training population into training, validation and excluded crossing edges. Keeping both endpoints outside the held-out group defines training; keeping both inside defines validation. Moving crossing pairs into training would invalidate both-endpoints-unseen development.

The existing label-blind assignment uses seed 20260929 and connected components of MMseqs hits at at least 40% identity and at least 80% coverage of **each** sequence, E-value <= .001, sensitivity 7.5. There are 4,154 components, largest size 3. The source alignment table and grouping hash are rechecked in v3. No exact endpoint or detected qualifying homology edge crosses a fold boundary. This does not exclude remote homology, shared domains or all pretraining exposure.

The source dataset is `danliu1226/Bernett_benchmarking`, pinned revision `5d2ad03baa165c27df32a2eadf066462a2a83073`. Original source row IDs and global protein indices are preserved. Tokens exist for 7,995 active official train/validation proteins only; unused historical test-only entries have zero token length. File and download hashes are verified before launch and by the trainer. No model-dependent resplitting is performed.

Negative labels mean degree-controlled sampled unreported interactions, **not experimentally established noninteractions**. V3 adds no new evidence curation, hard negatives or negative resampling. Dataset provenance and assay biases remain limitations.

## Length and sampling

Full official training has 32,624 pairs above 2,193 combined residues. The optional capped population would have 130,461 pairs, matching v2 capped. The six enabled configurations are uncapped.

| Fold | Maximum training tokens | Maximum validation tokens | Training exposures / rows |
| --- | ---: | ---: | ---: |
| 0 | 12,219 | 16,322 | 5.210948 |
| 1 | 16,322 | 10,416 | 5.355798 |
| 2 | 16,322 | 11,599 | 5.334371 |

Tokens include CLS and two EOS tokens. Validation coverage is unchanged. A length bucket is a computational batching strategy, not a filter. The sampler visits every retained row once per cycle and carries the final remainder into a fixed-size 64-pair update; v3 rechecks full-cycle coverage and cursor reconstruction for every fold.

Official validation can reach 39,391 tokens. It is not a Stage-1 validation population. The unchanged engine's prior v2 profile covers that length; this turn's new memory measurements cover the longest **internal-fold** pairs only.

## Fixed Stage-0 controls

[Cheap baseline diagnostics](diagnostics/cheap-baselines.json) use only these folds, with train-only standardization and fixed L2 logistic regression C=1 (no validation search). Amino-acid composition is calculated from full sequences. The additive control has score `f(A)+f(B)+bias`; the richer pair control includes sum, absolute difference and product features. Mean fold AP is:

| Baseline | Mean AP |
| --- | ---: |
| Constant training prevalence | 0.497985 |
| Length only | 0.523559 |
| Additive unary composition | 0.504290 |
| Pair composition | 0.555325 |

The pair-composition result is a useful shortcut control, not a ceiling on PLM performance or evidence of an interaction mechanism. No frozen-embedding probe or ensemble has been fitted during this preparation; those remain optional Stage-0 diagnostics.
