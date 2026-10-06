**Benchmark-v1**

An interim comparison of the validation-selected reference and symmetric retraining checkpoints against native PLM-interact on the complete Bernett test set. The completed comparison is in [REPORT.md](REPORT.md). See [PROTOCOL.md](PROTOCOL.md) for the fixed evaluation rules and [provenance/selection.json](provenance/selection.json) for immutable model identities.

The shared container is `../images/plm-interact/plm-interact-native-arm64-v1.sif`. From the workspace root:

```bash
bash retrain-v1/scripts/container.sh python benchmark-v1/scripts/qualify.py
sbatch benchmark-v1/slurm/benchmark.sbatch
bash retrain-v1/scripts/container.sh python benchmark-v1/scripts/analyze.py
bash retrain-v1/scripts/container.sh python benchmark-v1/scripts/write_report.py
```

Resubmit the same Slurm script to resume inference; completed hashed chunks are verified and reused. Never edit the frozen prediction configuration or scripts in place to resume a different experiment. The running training jobs and their selection rules are independent of this benchmark.

Raw predictions and chunk manifests are in `predictions/`; merged results, uncertainty intervals and figures are written to `results/`. The final interpretation is written to `REPORT.md` after all expected predictions have passed coverage and numerical checks.
