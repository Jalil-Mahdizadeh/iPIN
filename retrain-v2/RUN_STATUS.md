**V2 production completion status — 30 September 2026**

All four initial v2 jobs completed successfully, with SLURM exit code `0:0`, 12,745 updates, 815,680 pair exposures and final validation finished. Each used four GH200 GPUs. The completed validation audit is in [the results report](analysis/completed-20260930T082601Z/REPORT.md).

| Model | Array task | Node | Best update | Validation AP | AUROC | Elapsed |
| --- | --- | --- | ---: | ---: | ---: | --- |
| reference-official-seed2 | `3130215_0` | n537 | 4,000 | 0.653895 | 0.648293 | 20:15:44 |
| capped-official-seed2 | `3130215_1` | n584 | 8,000 | 0.651579 | 0.644744 | 11:10:50 |
| positive10-official-seed2 | `3130215_2` | n472 | 4,000 | 0.654795 | 0.654988 | 20:07:42 |
| clean-bce-official-seed2 | `3130215_3` | n524 | 4,000 | 0.652156 | 0.647778 | 20:04:49 |

Submission used `--array=0-3%4`. Array indices follow the frozen release's four enabled models. SLURM also assigns internal numeric job IDs to array tasks; the array IDs above are the stable user-facing mapping.

All 52 validation payloads were independently checked and their metrics recomputed. Full SHA-256 verification passed for all eight selected/final checkpoint payloads. Logged losses and gradients are finite, and final model hashes agree across four ranks in each run. The capped control has 130,461 training pairs, the others 163,085, and all validate on 59,258 pairs. Total allocated cost was 286.61 GPU-hours.

Positive10's selected AP exceeds reference by only 0.000900; its conditional protein-bootstrap interval spans zero and raw Brier is substantially worse. Clean BCE's early advantage did not persist. All four best checkpoints precede completion, so `best.json` and `latest.json` identify different states. These validation results do not establish a native test-performance improvement.

The subsequent [test benchmark](../benchmark-v2/REPORT.md) evaluated all four selected checkpoints. Clean BCE has the highest v2 test AP, 0.690596, versus native 0.690319; its AP difference from native is inconclusive. Its AP gains over both v1 models and lower Brier than native are reported with paired intervals. The validation-only findings above and the later test comparison concern different splits.

**V1 shutdown**

The user requested stopping both v1 jobs. Reference job `3112402` stopped at update **10,053**, and symmetric job `3112401` at **10,113**. Both committed their final resumable checkpoints and exited successfully. SLURM therefore records COMPLETED rather than CANCELLED, despite stopping before their planned training horizon. Final checkpoint SHA-256 hashes were verified; their previously selected best checkpoints remain unchanged. `REQUEST_STOP` markers remain in the v1 run directories.

**Records and monitoring**

- [Completed results and scientific interpretation](analysis/completed-20260930T082601Z/REPORT.md)
- [Metric, checkpoint and uncertainty audit](analysis/completed-20260930T082601Z/snapshot.json)
- [Final v2 SLURM accounting](provenance/final-accounting-3130215.txt)
- [Submission and concurrency override](provenance/submission-3130215.json)
- [Initial health check and task mapping](provenance/production-startup-health.json)
- [Transition, user authorization, and v1 checkpoint records](provenance/production-transition-20260929.json)
- [Final v1 SLURM accounting](provenance/v1-final-accounting.txt)

Run `python retrain-v2/scripts/status.py` for recorded run states and use `sacct -j 3130215` for completed job accounting. The [startup health record](provenance/production-startup-health.json) preserves the original launch snapshot. Final resumable states and the validation-selected checkpoints remain available.

The earlier STOPPED-BEFORE-PRODUCTION file records the preparation handoff; the subsequent explicit user instruction authorized this submission. Later objective/architecture/fold templates remain outside this four-model launch.
