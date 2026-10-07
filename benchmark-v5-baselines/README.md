# v5 and Bernett cheap-baseline comparison

Read the updated [REPORT.md](REPORT.md) for both the original human evaluations and the five-species extension. The existing v5 fits, training graph and degree values are reused unchanged. The corresponding Bernett baselines use the full original TRAIN release and the identical fixed method configuration.

The study now contains:

- The three original v5 human holdouts: validation, original Bernett test and ILP-negative test. Their scores and intervals are preserved.
- Both TRAIN references on all five species: 242,000 observations, three fixed cheap baselines per reference plus the exact-degree controls.
- Reused nonhuman predictions for v5 ESM2/ESMC, v2 clean BCE, v2 length-capped and native PLM-interact Bernett, with the same rows and paired bootstrap samples.

The Bernett cheap baselines capture modest signal and remain below both Bernett-trained v2 models. The v5 interolog baseline beats both v5 neural models in AP on every species. Below-chance v5 homology-degree results are retained in their original direction and diagnosed explicitly; they must not be called evidence of no predictive association.

| Artifact | Contents |
|---|---|
| [Combined figure](results/comparison.png) | Original human results plus both five-species training references |
| [Five-species figure](results/nonhuman-comparison.png) | All 13 fixed predictors, AP and AUROC |
| [Paired-difference figure](results/nonhuman-paired-differences.png) | Bernett minus v5 for each cheap baseline |
| [Metrics](results/metrics.csv), [intervals](results/confidence-intervals.csv), [differences](results/paired-differences.csv) | Combined machine-readable results |
| [Class-specific coverage](results/nonhuman-coverage.csv) | Positive/negative homolog and interolog support |
| [Subset results](results/nonhuman-subsets.csv) | Common exact-exposure removal and unique nonconflicting pairs |
| [Fallback diagnostic](results/nonhuman-v5-fallback-diagnostic.csv) | Explanation of the inverse v5 homology-degree ordering |
| [Verification](results/nonhuman-verification.json) | Independent source, score, model and bootstrap checks |

Every figure is also exported as PDF and SVG. The original frozen [PROTOCOL.md](PROTOCOL.md) remains unchanged; the [NONHUMAN-ADDENDUM.md](NONHUMAN-ADDENDUM.md) defines the extension. Its assumptions were fixed before new baseline metrics, while historical neural results were already known. Neither a baseline nor its scoring direction was selected on these species tests.

The original report, figures, metrics, scripts and completion manifest are archived in `archive/before-nonhuman/`. Original inputs, fitted models, human per-row predictions and search output remain in place. Larger artifacts stay on the HPC and are excluded from Git; the archive is required for reproducing the combined report.

Run in the same Arrhenius ARM64 environment with the retained original inputs and SIF:

```bash
bash benchmark-v5-baselines/scripts/run.sh
```

Completed stages verify their hashes and reuse results. The v5 regressor is never refitted. No neural training/inference or new SLURM job is needed. The initial two searches can use 32 CPUs concurrently; cached reruns do not repeat them. Historical standalone human-stage scripts remain for provenance, but the current entry point is the combined runner above.

The auxiliary integer-export audit is documented in the report: adding backtraces changed no baseline-relevant alignment field. Direct alignment-string counting resolved one apparent boundary discrepancy. No prediction was changed by these checks.
