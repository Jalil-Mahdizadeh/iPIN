# Completed Vx v2: local pooling did not establish a pairing benefit

The batch completed successfully. The scientific decision remains **no-go or inconclusive**: this fixed local-pooling readout did not improve the true model over the original global readout, outperform the shuffled-pairing control, or establish a benefit of spatial coherence. This is a result about the tested readout and development sample, not proof that useful co-evolutionary information is absent from the encoder.

## Primary assessment

All entries below retain the same native PLM-interact baseline and exact fallback. Assessment contains 958 pairs, including 659 eligible for MSA evidence.

| Model | AP | AUROC |
|---|---:|---:|
| Native baseline | 0.640930 | 0.648144 |
| Original global true MSA | 0.664361 | 0.658658 |
| Original global shuffled MSA | 0.663248 | 0.661071 |
| Quality/profile control | 0.650942 | 0.652725 |
| Local true MSA | 0.655735 | 0.655391 |
| Local shuffled MSA | 0.662517 | 0.659400 |
| Spatially scrambled local true | 0.651501 | 0.653388 |
| Spatially scrambled local shuffled | 0.657550 | 0.656363 |

The local true model improves over native baseline by +0.014805 AP, with a 95% matched protein-bootstrap interval [+0.000271, +0.026339]. But it trails its shuffled control by -0.006783 [-0.017395, +0.003384], and trails the original true model by -0.008626 [-0.021953, +0.004153]. Its improvement over the quality/profile control is also uncertain.

The primary contrast—the increase in the true–shuffled gap relative to global pooling—is **-0.007896**, interval **[-0.023482, +0.007892]**. The pairing-gap advantage of real over scrambled spatial order is -0.000734 [-0.011045, +0.009565]. Neither supports the hypothesis that this local aggregation recovers useful pairing-specific spatial information.

The same local true-trained head performs slightly better on true than shuffled input: +0.004356 AP [-0.001152, +0.010382]. That remains inconclusive. Calibration selected alpha 0.5 for both true arms and 1 for both shuffled arms, following the frozen rule. The separate-head comparison therefore includes calibration choices as designed; it should not be interpreted as proof that shuffling intrinsically improves biological information. The verification artifact also provides a posthoc table at every common alpha from the original grid, without changing any selected coefficient or primary result.

At a common alpha of 0.5, assessment AP is 0.655735 for local true and 0.654643 for local shuffled (gap +0.001092). At alpha 1, it is 0.661227 and 0.662517 (gap -0.001290). Thus unequal calibration weights explain much of the selected models' negative gap, but equal-weight comparisons still do not show a substantial pairing separation or recover the original true model's AP. These descriptive counterfactuals do not replace the frozen primary comparison.

## Length, diversity and coverage

The short-pair hypothesis was not rescued. On the 103 assessment pairs of combined length ≤512, local true AP is **0.778537** and local shuffled AP **0.798786**. On all 416 short DEV pairs, the corresponding values are 0.690284 and 0.705258. These are exploratory subgroups, not alternative primary endpoints.

Across all 4,000 fixed DEV pairs, local true AP is 0.654342, local shuffled 0.656710, and original global true 0.657453. The same overall conclusion holds.

The added diagnostic Neff is informative rather than identical to retained depth: median paired Neff at 80% identity is 86.40 on eligible TRAIN, 82.09 on eligible DEV and 83.03 on eligible assessment. This is still a declared identity/coverage proxy, not a ground-truth count of independent evolutionary observations. Median actual shuffled-token change on assessment is 27.66% of unmasked B-chain homolog cells. Median effective-support fraction is 89.65%: that fraction of valid residue pairs has jointly nongap observations in at least half the retained homolog rows.

None of the reported assessment strata has a positive lower interval bound for local true versus shuffled, improvement of the pairing gap, or the spatial-order pairing-gap advantage. This includes the highest Neff and coverage groups. Removing each of the ten most frequent proteins or recorded family witnesses leaves the primary contrast negative. These observations do not establish universal family independence: only partial TRAIN-shared family witnesses were available, and subgroup intervals are unadjusted.

## Hypothesis: useful family context survives pairing shuffle

**User-proposed interpretation, added 9 October 2026 after VZ completed.** This is a post hoc explanatory hypothesis, not an established mechanism or a change to either frozen experiment.

Shuffling the correspondence between homolog rows leaves the encoder jointly processing members of protein family A and members of protein family B. Schematically, `(A_i, B_i)` becomes `(A_i, B_permutation(i))`; each chain still contains the same homolog sequences. If compatibility between the two families is broadly conserved, the remaining information could support prediction through conserved interface residues, domain-related sequence patterns, tolerated substitutions, or structural constraints. Correct species-by-species matching may add little to this particular classifier even when homolog context is useful.

The [original shuffle implementation](../../pilot-vx/scripts/msa.py) preserves more than family identity:

| Input property | Effect of the implemented shuffle |
|---|---|
| Query A–B row | Remains correctly joined and unchanged |
| Each chain's homolog sequence collection | Preserved exactly |
| Per-position residue/gap frequencies and within-chain covariance | Preserved exactly |
| Original A–B homolog row assignments | Disrupted within permitted taxonomic groups |
| Depth, masks, eligibility, and reliability gate | Unchanged |

Permutations occur within **taxonomic** family, then order/class where needed; these groups are distinct from protein families. Residual singletons stay unchanged. Shared phylogenetic structure can therefore survive, and the intervention does not guarantee elimination of all cross-chain statistical association. It tests dependence on the original row assignments under this restricted null. It also does not establish that every original same-genome homolog pair is a biologically interacting pair.

The observed median **27.66% change in unmasked B-chain homolog tokens** is consistent with substantial conservation surviving the permutation. That percentage measures changed input tokens, not the fraction of predictive information destroyed or retained. The tensor can still change through joint processing even where individual input residues remain identical.

The subsequent [VZ query-only ablation](../../pilot-vz/results/REPORT.md) is consistent with this hypothesis: assessment AP was **0.643861 for Q**, **0.663248 for shuffled**, and **0.664361 for true**. Shuffled exceeded Q by +0.019387 AP, with a 95% matched protein interval [+0.000528, +0.035510]. True minus shuffled remained +0.001113 [-0.009639, +0.013333]. Thus homolog-bearing inputs provide useful information that survives this shuffle; the experiments do not demonstrate true/shuffled equivalence or an additional benefit from the original pairing.

Several explanations remain possible: site conservation, within-chain correlations, compatibility between families, residual cross-chain association, input-depth effects, and family/quality characteristics correlated with dataset labels. Family membership alone does not guarantee binding or preserve specificity among paralogs. Q changes both homolog content and depth, so its loss cannot identify which explanation is responsible. [VY's negative result](../../pilot-vy/results/REPORT.md) does not exclude useful family context during joint encoding: its independently pooled monomer features used a different representation and eligible population.

This hypothesis also leaves open whether the original global summaries discard localized pairing-sensitive information. The tested Vx v2 pooling did not establish a recovery of that information. The interpretation therefore supports retaining the distinction between **useful homolog context** and **demonstrated pairing-specific benefit**, while preserving the existing negative/inconclusive pooling decision. No new extraction, fitting, threshold change, or TEST evaluation follows from this documentation update.

## Completion and accounting

SLURM job **3516311** finished with `COMPLETED`, exit `0:0`. All four workers completed their assignments: **8,000/8,000 records**, including **4,453 eligible pairs**. Runtime was **2:01:01**, using **8.0678 allocated GPU-hours** and **145.22 allocated CPU-core-hours**. Cumulative GPU charge, including the existing qualification/audit reservations and the one-hour v2 reservation, is **21.9611 of 24 GPU-hours**. Final scheduler accounting is in [resources.json](../results/resources.json).

The independent CPU check in [verify_results.py](verify_results.py) verifies the freeze and per-record checksums, reconstructs predictions and metrics, checks TRAIN-only scaling and calibration selection, and recomputes the primary interval using sklearn AP instead of the production bootstrap's custom AP routine. Its machine-readable findings are in [verification.json](verification.json). It does not refit heads, run GPU inference, access TEST, or change the original results.

All checks passed: all 8,000 records verified, zero reported original-global feature error on all 4,453 eligible pairs, exact preservation of the retained global statistics, prediction reconstruction error at most 8.89e-16, and independently reproduced primary/control intervals. Spatial scrambling changed every eligible assessment local vector. Actual null changes met the declared minimum for 99.54% of eligible assessment pairs. No execution or artifact-integrity defect was identified by these checks.

The operational recommendation is to **stop this pooling ablation as planned**. The result does not justify scaling this configuration or automatically searching pooling settings on the same already-inspected assessment. The frozen experiment, diagnostics and negative/inconclusive result should be retained together.
