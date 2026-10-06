# Frozen v1–v4 controls for the nonhuman benchmark

The user selected both available interpretations of “closest to native”: v2 length-capped (best DEV checkpoint at update 8,000) and v2 clean BCE (best DEV checkpoint at update 4,000). The first preserves the native ESM2 CLS-linear architecture, standard attention, masked BCE/MLM and 2,193-residue combined training cap. The second has the closest previous Bernett AP/AUROC among the benchmarked v1–v4 models. Neither is selected using nonhuman outcomes.

Both models were trained on the cleaned Bernett data: clean BCE uses 163,085 training pairs; capped uses the 130,461 retained pairs within the length cap. Both select on the same 59,258 uncapped validation pairs. This does not reproduce the separate humanV11 training dataset used for Figure 2.

The existing parent benchmark provides the exact five released tests, pair union and source-row mappings. Preserve all 242,000 rows, including original duplicates, with full supplied sequences. Score both orientations and pool raw logits, retain original-order diagnostics, and use the native SIF with the same qualified BF16 settings as the earlier v2 benchmark. There is no retraining or nonhuman calibration.

`provenance/selection.json` freezes the prior weights and records why these models were chosen. Checkpoint paths reference already frozen benchmark-v2 snapshots and are verified by SHA-256. Short/middle/long saved DEV predictions and independent original-backbone forwards must pass before GPU jobs launch. Four independent GPU shards commit hashed prediction chunks for safe resumption.

The comparison uses both native releases and the two v5 models, reusing their completed predictions. The original eleven-model parent benchmark continues independently. AP/AUROC, paired protein-bootstrap intervals, duplicate and common human-exposure sensitivities, per-pair predictions and figures will appear in `REPORT.md` and `results/` after completion. Subset construction uses source membership only.

From the project root:

```bash
bash bechmark-v5-nonhuman/scripts/container.sh native python scripts/historical_infer.py --model v2-capped --qualify
bash bechmark-v5-nonhuman/scripts/container.sh native python scripts/historical_infer.py --model v2-clean-bce --qualify
sbatch --job-name=nhv5-v2-capped bechmark-v5-nonhuman/slurm/historical.sbatch v2-capped
sbatch --job-name=nhv5-v2-clean-bce bechmark-v5-nonhuman/slurm/historical.sbatch v2-clean-bce
bash bechmark-v5-nonhuman/scripts/container.sh analysis python scripts/historical_analyze.py
```

Resubmit the same model command after an interruption; verified committed chunks are reused. Do not launch simultaneous workers for the same model/rank. The analyzer refuses incomplete coverage and never reruns baseline inference.
