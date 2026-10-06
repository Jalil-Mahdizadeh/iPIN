# PLM-interact retraining v3

<!-- FINAL_V3_STATUS -->
**V3 finalized.** See [the final results](FINAL_RESULTS.md) and [complete benchmark](../benchmark-v3/REPORT.md).

<!-- V3_AUTHORIZED_EARLY_STOP -->
**Training stopped by user request at update 8,446.** The update-4,000 checkpoint remains the validation-selected best. The separate [benchmark-v3 comparison](../benchmark-v3/README.md) is now authorized. This supersedes earlier full-horizon/deferred-benchmark status below; the original frozen training release is preserved.

**Current stage, 2026-10-01:** one final full-data residue-MLP model is running at LR **2e-5**, seed 2, under SLURM job **3206610** (four GPUs). This task is **retraining only**. It uses 163,085 training pairs, 59,258 validation pairs and a fixed 12,745-update budget. The best official-validation checkpoint is retained. Benchmarking is deferred to a later task in the separate `benchmark-v3` folder, with no automatic handoff. See [the active protocol](FINAL_PROTOCOL.md) and [submission](provenance/final-submission-3206610.json).

The brief initial job **3206387** was safely stopped at update **26** to remove its automatic benchmark phase. Job 3206610 resumes the same configuration, model/optimizer/RNG/sampling state and training fingerprint; this is **one model**, not a second experiment. [Correction provenance](provenance/final-training-only-amendment.json) preserves the change. No new v3 test inference has run.

**Completed development stage:** all six readout tasks in array **3196827** completed successfully. Residue-MLP improved mean best validation AP from **0.616488 to 0.620374**, but its **+0.003886** gain fell below the preset +0.005 advancement threshold. CLS-MLP averaged **0.614789**. The original frozen decision promotes neither candidate; the later single exploratory run is an explicit user-authorized override, not a passing gate. See [the complete results](READOUT_RESULTS.md), [decision](decisions/readout-window5000.json) and [completion/checkpoint audit](provenance/readout-completion-3196827.json).

| Stage | Comparison | Status |
| --- | --- | --- |
| Controls | Length and composition baselines, three folds | Completed; [diagnostics](diagnostics/cheap-baselines.json) |
| Learning rate | Clean BCE + CLS-linear; 2e-5 versus 5e-6, three folds each | Ended early; mean best AP 0.616488 versus 0.600486; all folds favored 2e-5 |
| Readout | CLS-MLP versus residue-MLP at 2e-5, three folds each | Completed; residue shows a small gain, neither candidate passes the full advancement rule |
| Final full-data run | One residue-MLP, seed 2, LR 2e-5 | Job 3206610; user-authorized exploratory run despite the missed advancement gate |
| Multi-seed confirmation | Previously proposed six full-data runs | Not pursued; replaced by the single final run |
| Native adaptation / external evaluation | Previously considered alternatives | Not pursued in this final task |

The completed readout stage tested additional nonlinear CLS capacity and residue information at equal added parameter count. Each run starts from pinned ESM2, with full backbone fine-tuning, clean BCE, both pair orientations and the unchanged development folds. No historical test results are used for selection.

## Selection window and compute

The original LR screen was stopped before its planned 6,000 updates. The completed readout stage used **five** common validation opportunities, updates 1,000–5,000, matching the available controls. It kept the original 6,000-update learning-rate schedule; all six workers stopped safely at update 5,001 after the committed update-5,000 validation. Changing the schedule length would have changed the learning-rate trajectory. `SCREEN_COMPLETE.json` records completion of this window; it does not claim full-schedule completion.

This is an adaptive development amendment made before readout production. Readout production consumed **178.16 allocated GPU-hours**, within the planned 125–225; qualification is additional. The ended LR screen consumed approximately **190.8 allocated GPU-hours**. No result yet establishes superiority over native PLM-interact.

## Inspect and analyze

From the iPIN workspace:

```bash
python retrain-v3/scripts/status.py
sacct -j 3196827 -X --format=JobID,State,ExitCode,Elapsed
```

The readout release is [20260930-readout](releases/20260930-readout/release.json), selected by `releases/CURRENT_READOUT`. The original adaptation release and `releases/CURRENT` are preserved. Production uses frozen code and the pinned SIF; mutable trainer invocation is rejected.

The readout launcher now rejects all six completed screening windows. The frozen comparison has already run and preserved [its decision](decisions/readout-window5000.json); it refuses to overwrite that record. Its invocation is recorded in [review provenance](provenance/readout-review-20261001.json). The original `compare_development.py` expects completed 6,000-update schedules and must not be used for this amendment.

[Validation curves](reports/readout-validation-ap.png) show the selected peaks at updates 3,000–4,000 and subsequent decline. All 18 retained checkpoint payloads passed SHA-256 checks. Results use saved development predictions; no new test inference or production jobs were launched during the review.

## Resume and stop

For the current final full-data run, use `python retrain-v3/scripts/launch_final.py --submit` for same-contract continuation after a job has stopped. It selects `releases/CURRENT_FINAL`, presently `20261001-final-training`, and rejects duplicate active submissions. The job ends after training. Its `best.json` and `completed.json` do not indicate test evaluation. Historical readout controls below remain preserved.

Every run retains atomic, hashed checkpoints containing model, optimizer, per-rank RNG, sampling position, pair exposures, selection state and pending validation. Automatic checkpoints occur every 100 updates, 15 minutes, validation and safe stop. Same-release, same-configuration, four-GPU continuation is required. Corrupt latest commits can fall back to retained valid dependencies; missing state cannot silently initialize a new model.

To stop safely, create `runs/<name>/REQUEST_STOP`. To intentionally continue an unfinished readout window, inspect the checkpoint, remove the stop marker, and use `launch_readout.py --run <name> --submit` with the same release. Completed windows and duplicate active campaigns are rejected. Time-limit signals retain task-specific SLURM requeue, up to eight attempts.

Full-size qualification found a residue-head resume mismatch, which was fixed before production by storing gradients independently of DDP communication buckets. Both actual-size heads then passed bitwise interrupted/resumed comparisons on four GPUs in job **3196606**. The model, data and checkpoint implementation are unchanged. The original failure, minimal-code-change check, memory cost and scope are documented in [the engineering record](qualification/READOUT_ENGINE.md); no tolerance replaced exact equality.

Additional records: [data card](DATA_CARD.md), [original protocol](PROTOCOL.md), [experiment ledger](EXPERIMENT_LEDGER.md), [initial qualification](QUALIFICATION.md). Original pre-submission readiness records and frozen protocols retain their historical state.
