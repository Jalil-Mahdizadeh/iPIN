# V8: frozen partner diagnostics and external-assay feasibility

Partner research screen: **no_go_or_inconclusive**.
External assay gate: **no_go_insufficient_panel_support**.

No models were trained, no new encoder predictions were generated, and no original Bernett TEST data or scores were opened. The pre-existing TEST sequence-hash exclusion list was reused.

## Primary validation diagnostics

The unchanged assessment excluding pilot rows contains 13,055 pairs, 6,544 positives, 1,746 proteins and 1,709 inherited sequence components. This is retrospective analysis of previously examined validation data.

Within-anchor support: 1445 anchors / 1419 components. Retrieval support: 151 anchors / 151 components, requiring at least 20 negative-labelled candidates per anchor. Swaps: 2000 selected from 34401 admissible quartets, spanning 903 components.

| Frozen score / fixed control | Within-anchor concordance | Recall@10 | Swap preference |
|---|---:|---:|---:|
| native | 0.644579 | 0.238203 | 0.744000 |
| vx | 0.647601 | 0.237507 | 0.746500 |
| quality | 0.645110 | 0.240727 | 0.748000 |
| aac_cosine | 0.562449 | 0.211489 | 0.614500 |
| kmer3_cosine | 0.541517 | 0.190238 | 0.591000 |
| length_ratio | 0.520949 | 0.190422 | 0.510750 |

| Primary contrast | Difference | Paired 95% component interval |
|---|---:|---|
| within_anchor: Vx − native | 0.003022 | [-0.002613, +0.009953] |
| within_anchor: Vx − quality | 0.002491 | [-0.003584, +0.009242] |
| recall@10: Vx − native | -0.000697 | [-0.004187, +0.002450] |
| recall@10: Vx − quality | -0.003221 | [-0.006691, +0.000273] |
| swap_preference: Vx − native | 0.002500 | [-0.023174, +0.030710] |
| swap_preference: Vx − quality | -0.001500 | [-0.032213, +0.029766] |

The research screen requires both within-anchor and swap improvements of at least 0.010 over native, positive paired lower bounds over native and quality on both, and positive swap lower bounds over composition and 3-mer controls. Every fixed control and the secondary all-assessment view remain in the JSON. A failed screen does not establish equivalence or negate the completed modest Bernett AP gain.

A nonconstant additive endpoint score cancels exactly on every selected quartet. All alternatives were existing negative-labelled candidates; none was manufactured or newly scored. Sequence similarity can also favor observed pairings, so swap preference is not evidence of physical interfaces or co-evolution.

## External experimental panel

| Prespecified population | Pairs | Positives | Confirmed assay non-detections |
|---|---:|---:|---:|
| current_train_val_pair_disjoint | 1770 | 69 | 1701 |
| eligible_final_source | 2674 | 376 | 2298 |
| historical_primary | 1940 | 77 | 1863 |
| primary | 1116 | 40 | 1076 |
| primary_both_cached_and_length_eligible | 240 | 7 | 233 |
| primary_length_eligible | 1024 | 38 | 986 |

The primary starts from the historical exposure-excluded Y2H panel, removes exact original Bernett TRAIN/VAL sequence pairs, then conservatively excludes every pair touching a previously evaluated TEST protein hash. That final exclusion is stricter than removing exact TEST pairs and changes the population. Earlier views are transparent attrition counts, not alternative evaluation populations selected after the audit.

Required earlier support floors failed: even_all_length_eligible_positives_below_40_MSA_support_floor.

External model inference was therefore not run. Current-training homology, new component construction and exact paired-MSA reconstruction were not pursued after the decisive earlier gate. Cache absence is unknown availability; the sequence-length limit is a genuine maximum possible Vx coverage. This is a panel-feasibility result, not a failed external Vx benchmark.

Assay non-detection does not establish universal noninteraction. The source used selected candidates and reference sequences rather than verified exact constructs. Even a panel meeting the minimum floors would only support a bounded exploratory transfer study; it need not resolve a small improvement.

## Verification and interpretation

All reused logits and their sequence/pair identities match the frozen parent records. Independent checks rebuild every quartet selection and cap, all anchor/retrieval points, all reported interval quantiles and paired contrasts, every external exposure exclusion, and 12 full raw bootstrap draws per validation view. Synthetic qualification checks ties, component multiplicities, deterministic sampling and fail-closed duplicate handling.

Intervals use 2,000 paired component draws, condition on the fitted models and candidate design, and omit model-selection uncertainty. Retrieval intervals resample anchor components conditional on fixed lists and leave shared-partner dependence. The inherited 40%-identity/80%-coverage graph is heuristic and does not certify unseen biological families.

See [validation results](validation_diagnostics.json), [external feasibility](external_feasibility.json), [verification](verification.json), and [the frozen proposal](../../improvment-proposal-v8.md). No outcome-dependent model changes or new experiment follow automatically.
