# Original Bernett VZ pilot

Authorized 9 October 2026 from the [VZ proposal](../improvment-proposal-vz.md). The user's subsequent “pilot-vs” instruction is interpreted as this proposed `pilot-vz` ablation. [STATUS.md](STATUS.md) records execution. The proposal is retained verbatim in `provenance/proposal-at-start.md`.

One new arm, **Q**, encodes the two query sequences jointly through the unchanged original Vx encoder at MSA depth one. The original masked paired tokens are reconstructed with Vx's pairing code, checked against their Vx-v2 hash and original metadata, and sliced to `tokens[:1]`. Q retains the complete original query coordinates, PAD masks, chain breakpoint, zero-based layer 15, directed inter-chain symmetrization, seeded 256→32 projection, global mean/std/max/q95, and AB/BA average. No structural or language-model output head executes. No VY encoder/readout or Vx-v2 pooling enters Q.

Use the same 4,000 TRAIN / 4,000 DEV rows, 1,825 eligible TRAIN / 2,628 eligible DEV pairs, and original per-pair gate. DEV has 1,068 calibration, 958 assessment and 1,974 crossing rows; 659 assessment rows are eligible. Q needs 4,453 symmetric pair encodings (8,906 forwards). Every previously unavailable pair retains its native PLM-interact logit exactly. Recomputing depth or availability from Q is forbidden. Computational errors never become biological fallback.

Fit exactly one Q head: eligible TRAIN StandardScaler → PCA(32, full solver) → logistic regression C=0.1, lbfgs, max_iter=1000, seed 20261007. Only calibration AP chooses alpha from `{0, .1, .25, .5, 1}`, smallest-alpha tie. Native TRAIN logits are not used. Reconstruct original true/shuffled/quality predictions using saved heads and original gates/alphas; compare against independently verified VY reference columns. No historical head is refitted or recalibrated.

The primary estimand is fused Q-minus-shuffled AP on all 958 assessment pairs. Retain original Vx population-specific protein Poisson multipliers (1,000, seed 20261007), endpoint-product edge weights and 95% percentile intervals; the original true-minus-shuffled interval must reproduce. Reuse matched assessment draws for assessment strata; full DEV has its own original Vx protein universe. Intervals condition on fitted models/alpha and are exploratory on previously inspected DEV.

Useful Q gain requires ≥0.010 AP over native, a positive 95% interval, and AUROC decline ≤0.005. Recovery within tolerance additionally requires Q-minus-true/shuffled lower bounds >−0.010. Equivalence requires the entire interval inside ±0.010. The tolerance permits approximately 45% of the original shuffled gain to be lost; it is not a claim of identical performance. Report eligible-only noninferiority, original quality/profile, common-alpha and unfused-head diagnostics, unchanged TRAIN-derived Neff/coverage bins, and partial family/protein influence. None selects a new model or rescues a failed primary result. The fixed true-minus-shuffled evidence cannot become stronger by adding Q.

All successful interpretations are conditional on MSA-derived masks, eligibility and gate. A Q loss cannot isolate evolutionary information from input-depth effects; a Q gain cannot uniquely establish biological interaction information or a fully MSA-free pipeline. No production or TEST continuation is authorized.

Preparation snapshots only allowlisted TRAIN/DEV artifacts and verifies source manifests without traversing their unrelated TEST exclusion inputs. Monomer caches are immutable, checked at preparation and every cache load; no archive rebuild/search. Freeze binds code, tests, snapshots, qualification and source checksums before outcome fitting. Output checksums and fingerprints reject partial/corrupt caches. All 8,000 status records are required before fitting; finite evidence is checked even when alpha is zero.

GPU qualification uses the three original Vx real cases plus the actual maximum combined length 1,536. Additional timing anchors nearest fixed lengths 256/512/768/1024/1280/1536 are selected without labels. Hooks verify both orientations use depth one and blocks 0–15; old true/shuffled vectors must reproduce exactly on the original cases. Q repeats and reversed orientation must agree exactly. Timing uses monotone upper-length buckets including reconstruction, a factor of 1.5, 600 seconds I/O and 1,800 seconds analysis reserve. These timing anchors do not introduce outcome experiments.

Separate ceilings: **8 allocated GPU-hours, 600 allocated CPU-core-hours, 5 GB additional storage**. The dedicated interval starts in `provenance/resource-start.json`, including preparation/qualification, idle time, failures, analysis and audit, charging one GPU and the full 72 allocated CPUs in existing allocation 3565552. The launcher requires an idle GPU, unchanged SIF, enough remaining allocation time and budget for the conservative full workload. The worker preserves an analysis reserve; the controller enforces the deadline. No automatic retry/requeue, new SIF, or cancellation of the user's interactive allocation.

The detached controller runs extraction → one fit/report → independent CPU verification. The audit reconstructs predictions, checks TRAIN transforms and calibration, and recomputes all primary and eligible-only assessment intervals with sklearn AP. Only a successful audit and resource ledger produce `results/COMPLETE.json`. An execution error takes precedence over any provisional scientific report.

```bash
python -B pilot-vz/scripts/status.py
```

Do not modify frozen inputs or scripts, relaunch, or refit after the study starts. Original Vx/VY artifacts and unrelated workspace files remain unchanged.
