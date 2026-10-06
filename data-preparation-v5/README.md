# V5 data preparation

The agreed [proposal](../improvment-proposal-v5.md) uses HIPPIE v3 supervision and two Bernett test variants with the same 26,024 positives. Data preparation is separate from model retraining and benchmark inference.

**Status: complete, 2026-10-02.** The independent final audit passed, `completed.json` records `complete: true`, and all 23 exported-file hashes were verified after completion. See [the final report](REPORT.md), [final audit](reports/final-audit.json), and [completion manifest](completed.json). No model retraining or benchmark inference was started.

The single additional positive-partition attempt finished at its 30-minute CPU solve limit. It found **no additional retained positives**; the same partition remains selected. The mathematical gap narrowed from 49.41% to **47.18%**, without a dataset change. Independent review confirmed the original protein assignments and all 23 export hashes are unchanged. No further attempt is running or planned. See [the attempt report](partition-attempt-v1/REPORT.md).

The final training set contains **700,764 pairs** (350,382 positive and 350,382 negative) across 13,110 distinct sequences. Validation contains **165,742 pairs** (82,871 positive and 82,871 negative) across 6,023 distinct sequences. Both test sets contain **52,048 pairs**. The train/dev negative samplers reached gaps of **0.3362% / 0.0438%** in **95.9 / 17.4 seconds**, respectively. The previously completed positive split and ILP test retain their original solver gaps of **49.41% / 2.98%**; those stages were not rerun or claimed to meet the 1% target.

The original 52,048 test rows, labels and sequences have been verified against the existing benchmark arrays. All 3,022 test proteins remain protected; ILP negatives are drawn only among the 2,948 positive-endpoint proteins. The original test remains unchanged, including any labels conflicting with newer evidence.

The frozen HIPPIE snapshot contains 1,179,347 unique interaction pairs and 29,080 accessions. Its pairs and scores exactly match the authors' released source. UniProt sequences, aliases and GO-BP annotations are pinned to release `2026_03`. Original test sequences are preserved independently of current UniProt sequences. See [source/test audit](reports/source-and-test-audit.json).

At launch on 2026-10-02, initial exclusions left 808,015 positive pairs. Bidirectional test-homology filtering then left **721,082 eligible positive pairs across 20,812 distinct sequences**, before train/dev splitting and removal of crossing interactions. These are intermediate counts. Normalization also identified 358 original negative labels conflicting with the HIPPIE snapshot; these labels remain unchanged. See [normalization](reports/normalization-audit.json), [homology exclusions](reports/test-homology-filter.json) and [historical exposure](reports/historical-exposure.json).

Preparation proceeds through identifier/isoform resolution, exact-sequence deduplication, test-family and homology exclusion, grouping, a two-way positive-splitting ILP, and separate TRAIN/DEV/test-negative ILPs. The final checks verify pair integrity, source membership, protein/family separation, directed homology boundaries, known-positive exclusions and both backbone tokenizations.

Implementation choices are fixed in [configuration.json](configuration.json). Qualifying homology means detected identity ≥40% over an alignment with ≥80% coverage of both sequences, using MMseqs2 with sensitivity 7.5. Aliases and isoforms of a protected UniProt entry are excluded independently of alignment coverage. Development entry families and qualifying homology components remain together before KaHIP grouping and the two-way retained-interaction ILP.

The negative ILP uses the pinned author objective and hard degree cap. Its wrapper adds a complete known-positive family blacklist (HIPPIE plus historical train/validation/test positives), a human, non-self scope for both positives and new negatives, and seeded random selection of excess candidates within GO strata. The last change corrects upstream truncation of sorted pair keys. Candidate pools, solver settings, actual solver gaps and residuals are saved. A feasible time-limited solution is not claimed as a proven optimum; sampled negatives are not experimentally established noninteractors.

The CPU runtime uses the existing ESMC SIF with an isolated Python environment, CVXPY 1.9.2, HiGHS 1.15.1 and locally built ARM64 KaHIP 3.25. Both ILP formulations passed comparisons with exhaustive solutions on small instances; see [qualification](reports/qualification.json). Model weights are not loaded for this workflow.

The original TRAIN/DEV negative-sampling attempt reached its time limit without a feasible validation incumbent. Recovery completed successfully using the finished stages and frozen candidate pools, verified feasible starts, an equivalent scaled formulation and saved improving solutions. Its runtime was capped at 30 minutes per sampler and five minutes without an improved incumbent; both samplers finished below those limits. See [recovery details](recovery-v1/README.md).

Launch/resume recovery on a compute allocation with at least 16 allocated CPUs:

```bash
python data-preparation-v5/recovery-v1/launch.py
```

The drivers allow one active preparation process, validate completed-stage hashes, and refuse a silent resume after code/configuration changes. Completed stages and fixed candidate pools are reused. Recovery restarts an interrupted solver from its saved feasible incumbent, without a saved branch-and-bound tree, and accounts for the remaining runtime budget. Two eight-thread samplers may overlap. The completed original positive-splitting and test-negative ILPs had two-hour limits; recovery's shorter limits apply to the unfinished TRAIN/DEV samplers. Existing interactive allocation resources are used, with no additional GPU or model-training job.

Final artifacts are `prepared/train.csv`, `prepared/val.csv`, full development sequences and metadata, separate `prepared/esm2` and `prepared/esmc` arrays, the unchanged `frozen-tests/original-test.csv`, the custom `frozen-tests/test-ilp.csv`, a final report and the hashed completion manifest. `retrain-v5` and `benchmark-v5` are future, separate tasks. Full sequences are retained, including 585 training pairs and 446 validation pairs longer than 8,192 combined residues; the model-training memory qualification remains to be completed before production.
