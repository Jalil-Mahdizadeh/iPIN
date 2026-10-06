# V5 native PLM-interact retraining

**Stopped safely at the user's request on 2026-10-05, at approximately 08:36 Stockholm time.** Both production jobs exited with code 0 and released their allocations. This was early termination of the planned 54,748-update budget after validation AP plateaued; the full budget was not completed.

| Model | Job | Final saved update | Best DEV-AP update | Best DEV AP |
| --- | --- | --- | --- | --- |
| ESM2 650M | 3301121 | 45,494 | 27,374 | 0.615327 |
| ESMC 600M | 3301122 | 45,227 | 21,899 | 0.592462 |

Both the final resumable checkpoints and the selected best checkpoints were preserved and verified against their SHA-256 hashes. `REQUEST_STOP` remains in each run directory and prevents automatic continuation. Each run's `stopped.json` records the final state; `status.json` is the last periodic training-progress snapshot. SLURM reports `COMPLETED` because the safe-stop wrapper exited successfully, not because all planned epochs finished. Detailed evidence is in [manual-stop-completed-20261005.json](provenance/manual-stop-completed-20261005.json).

Two independent production runs use the original CLS → ReLU → linear PPI head and active masked-language-model objective: ESM2 650M (`esm2-native-ilp-seed2`) and ESMC 600M (`esmc-native-ilp-seed2`). Both start from their original language-model pretrained weights. The immutable production release is `releases/20261002-native-ilp`.

See [PROTOCOL.md](PROTOCOL.md) for the complete architecture, objective, data, fixed budget, validation and restart contract. [HEALTH_REPORT.md](HEALTH_REPORT.md) records the production startup observations once verified. Submission receipts are in `provenance/submission-*.json`.

Both production jobs were submitted at **2026-10-02 22:17:50 Stockholm time**, without dependencies:

| Model | SLURM job | GPUs |
| --- | --- | --- |
| ESM2 650M | 3301121 | 4 |
| ESMC 600M | 3301122 | 4 |

They were initially queued for priority. The read-only `scripts/watch_startup.py` audit passed for both models on 2026-10-02 and then exited. [HEALTH_REPORT.md](HEALTH_REPORT.md) is the historical startup report; the stop record above gives the final job status.

## Data and training budget

- TRAIN: 700,764 pairs; DEV: 165,742 pairs, both balanced between positives and ILP negatives.
- The two models use identical physical pairs and full sequences; no length truncation.
- Each run uses one node with four GH200 GPUs, 64 physical pairs per update, learning rate 2e-5 and 54,748 updates (five dataset passes, with the last batch completed).
- Full DEV is evaluated 20 times; the first evaluation is at update 2,738. Highest pooled DEV AP selects the checkpoint, with earlier exact ties retained.
- Each allocation requests 48 hours. Time-limit warnings trigger a checkpoint and requeue toward the same fixed training horizon, with at most eight automatic restarts.
- Neither frozen test is loaded. Benchmarking is a separate later task.

## Qualification completed before production

Both adapters passed reference-model and gradient comparisons, active MLM-decoder checks, masking checks and loss-normalization checks. Both full-size models processed the longest training pair (17,652 model-input tokens) with backward and optimizer steps, and the longest validation pair (42,878 tokens), without truncation.

Qualification job **3300703** completed successfully. For both full-size models on four GPUs, uninterrupted training and stop/resume training produced bitwise-identical model, optimizer, RNG and sampler state, and validation predictions. This included interruption with validation pending. Checkpoint corruption/fallback, SLURM wrapper and production-guard checks also passed. Reports are under `qualification/`; their hashes and source dependencies are pinned by `releases/20261002-native-ilp/release.json`.

The small qualification sets establish execution correctness only. Their AP/AUROC values are not estimates of production performance.

## Monitoring and recovery

From the project root:

```bash
squeue -u jalil -o '%.18i %.24j %.10T %.10M %.20R'
tail -n 10 retraining-v5/runs/esm2-native-ilp-seed2/events.jsonl
tail -n 10 retraining-v5/runs/esmc-native-ilp-seed2/events.jsonl
```

Each run writes `latest.json`, `checkpoint-status.json`, hashed checkpoints and, after full validation, `best.json`. Checkpoints include optimizer state, all four ranks' RNG, dataset cursor and pending-validation state. An early checkpoint is made at update 10; regular commits occur every 100 updates or 15 minutes, at validation boundaries and on safe exit.

After inspecting and resolving an unexpected failure, the launcher resumes the latest valid committed state using the same frozen release. For example:

```bash
python3 -B retraining-v5/releases/20261002-native-ilp/code/launch.py --run esm2-native-ilp-seed2 --submit
```

The launcher rejects duplicate active runs. Creating `REQUEST_STOP` inside a run directory requests a checkpoint and safe stop. A manually stopped run requires an intentional decision to remove that marker before continuation. Do not edit a frozen configuration or discard checkpoints to restart a run.
