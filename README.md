# iPIN

Research code, protocols, provenance and evaluation results for protein–protein interaction prediction. This workspace documents reproduction of native PLM-interact, successive iPIN training experiments, and benchmarks on human and nonhuman datasets.

## Current results

- [V5 model benchmark](benchmark-v5/REPORT.md): frozen iPIN ESM2 and ESMC checkpoints, native PLM-interact, and other published predictors on the original Bernett test and a custom ILP-negative test.
- [V5 simple-baseline benchmark](benchmark-v5-baselines/REPORT.md): three fixed CPU methods refitted on v5 TRAIN and evaluated on v5 validation and both tests. Their AP and AUROC are approximately 0.500–0.506; the iPIN checkpoints score substantially higher. This addresses the tested mechanisms, not every possible dataset bias.
- [STRING V11 simple baselines](literature/string-v11-baselines/REPORT.md): the same homology-transferred degree, sequence-propensity and interolog methods on V11 human validation and five species.
- [Exact V11 TRAIN-degree assessment](literature/string-v11-train-degree-baseline/REPORT.md): independent verification of AP 0.8360 and AUROC 0.9635 on the released human validation data.
- [Five-species model benchmark](benchmark-v5-nonhuman/REPORT.md): mouse, fly, worm, yeast and E. coli.

The two v5 tests share their positive pairs and are not independent replications. Their 1:1 class balance also differs from V11's approximately 1:10 ratio, so raw AP should not be compared across these settings without accounting for prevalence.

## Workspace layout

| Location | Purpose |
|---|---|
| `plm-interact-reproducability/` | Native-model reproduction and source verification |
| `retrain-v1/` through `retrain-v5/` | Training code, frozen configurations and validation records |
| `benchmark-v1/` through `benchmark-v5/` | Evaluation code, comparisons, reports and figures |
| `data-preparation-v5/` | Protected tests, sequence partitioning and ILP negative-sampling implementation |
| `benchmark-v5-baselines/` | Fixed simple baselines fitted on v5 TRAIN |
| `literature/` | Research notes and baseline/data audits |
| `images/` | Container documentation and checksums |
| `improvment-proposal-*.md` | Recorded development proposals; historical filenames are retained |

## Reproduction and artifact storage

This Git repository contains source code, protocols, structured result summaries, figures and provenance. Large datasets, containers, model checkpoints, fitted binary artifacts, numerical prediction arrays, downloaded publications and runtime environments remain in the HPC archive and are excluded by `.gitignore`.

The experiment manifests describe the original completed HPC runs, including hashes of files intentionally omitted from Git. A `COMPLETE.json` file is therefore a record of that verified run, not a claim that a fresh clone contains all its assets. Reports link some archived artifacts that must be restored from the recorded sources before rerunning an experiment.

Scripts preserve the recorded Arrhenius environment and, in some cases, absolute workspace paths. Review each experiment's protocol and image documentation, restore its required assets, and adapt paths before running elsewhere. The new simple-baseline studies use the existing ARM64 SIF for Python and MMseqs2 on CPU; they do not run GPU inference.

With the recorded assets available on Arrhenius, the v5 baseline workflow is:

```bash
bash benchmark-v5-baselines/scripts/run.sh
```

The V11 baseline folder was renamed from `literature/V11-baselines` to `literature/string-v11-baselines`; older provenance retains the original absolute paths. Preserve or remap those paths when restoring that archived run.

The nonhuman benchmark folder was renamed from `bechmark-nonhuman-v5` to `benchmark-v5-nonhuman`. Historical result and provenance records retain their original paths and hashes. The HPC workspace has an ignored compatibility symlink from the old name to the new name; recreate that link when using those archived absolute paths. Original completion manifests are preserved in the benchmark archives. The current nonhuman comparison includes X-PAIR humanV11 and verifies reuse of the earlier predictions and bootstrap samples.

Method definitions, source attribution, uncertainty estimates and limitations are recorded in the individual protocols and reports.
