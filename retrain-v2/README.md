**PLM-interact retraining v2 — initial four runs completed**

This folder implements the first staged campaign from [proposal v2](../improvment-proposal-v2.md) using the existing [ARM64 SIF](../images/plm-interact/plm-interact-native-arm64-v1.sif). On 29 September 2026, the user authorized stopping both v1 jobs and launching all four initial v2 models simultaneously. Array **3130215** used four four-GPU nodes (16 GPUs total). All four runs completed successfully by 30 September 2026. See [the completed results and audit](analysis/completed-20260930T082601Z/REPORT.md), [run status](RUN_STATUS.md), and [submission record](provenance/submission-3130215.json).

Selected validation AP ranges from 0.6516 to 0.6548. Positive10 leads reference by only 0.0009 AP, with no robust improvement established. All final checkpoints are worse than their earlier validation-selected checkpoints; `best.json` identifies the appropriate selection for subsequent comparisons. These results have not established improved native test performance.

The subsequent [benchmark-v2 test comparison](../benchmark-v2/REPORT.md) is complete for all four selections. Clean BCE has test AP 0.690596 versus native 0.690319; their tiny difference is inconclusive. Clean BCE improves on both v1 models in this historical comparison and has lower Brier than native. Native and v1 predictions were verified and reused. See the benchmark report for all seven models and paired intervals.

Bounded disposable qualification runs are recorded separately under `qualification/`; their metrics are software checks, not evidence of improved PPI performance.

The first release enables four matched experiments: full-coverage reference, the ≤2,193-residue coverage control, positive-class weight 10, and clean classification without MLM. Each uses the same pinned ESM-2 initialization, 12,745 updates, 64 physical pairs per update, and complete validation. [Full scientific protocol](PROTOCOL.md).

Training/validation contain 163,085/59,258 pairs; the capped arm has 130,461 training pairs. Three additional protein/homology-group development folds are frozen. No historical test pairs, labels, scores, or unused test sequences are included in the v2 inputs. [Data card and fold counts](DATA_CARD.md).

The later masked-BCE, separate clean/MLM pass, lower-LR, parameter-matched CLS/residue heads, and chain-aware attention configurations are prepared. The new attention/readout mechanisms have been qualified with the full backbone. These later experiments are not automatically submitted by the initial release. New evidence-curated data, interface supervision, and an independent external holdout remain conditional work; no generalization or outperformance claim is justified yet.

**Qualification and handoff**

Read [the qualification report](QUALIFICATION.md) for numerical, longest-sequence, checkpoint-corruption, four-GPU, and scheduler-requeue evidence. The release freezer checks that the qualified code/data hashes still match. A release holds read-only copies of production code and the four enabled configurations; the mutable source configs are not directly production-launchable.

From the workspace root, the following command verifies the frozen inputs and writes a **dry-run** launch plan:

```bash
python retrain-v2/scripts/launch.py --max-concurrent 4
```

The resulting [launch plan](provenance/production-launch-plan.json) contains the SLURM command. Actual submission requires explicitly adding `--submit`; this was executed once after the user's authorization. The launcher checks for an already active campaign before submission. Each of the four array tasks uses one four-GH200 node, and this submission permits all four to run concurrently. The earlier [preparation handoff](provenance/STOPPED-BEFORE-PRODUCTION.json) is a historical record of the requested pause before the later launch authorization.

```bash
python retrain-v2/scripts/status.py
```

Status reports each experiment independently. For each run, `runs/<name>/best.json` identifies its selected checkpoint, `latest.json` its latest resumable state, `events.jsonl` its history, and `validation/` the complete predictions. A manual stop uses `REQUEST_STOP` in that run directory. To continue an interrupted run later, remove that intentional stop marker and use the same launch script with `--run <name>`; its default is still a dry run. Resume requires the same frozen release, four GPUs, and data contract. The initial four runs have already reached their prescribed horizon.

**Resource planning**

The completed initial four runs used **286.61 allocated GPU-hours**, including validation and checkpoint overhead. Three full-coverage runs each took approximately 20 wall-hours on four GPUs; the capped run took 11h 10m 50s. The original planning estimate was 320–400 GPU-hours. Later folds, seeds, and architecture runs add cost. The combined chain-aware/residue/decoupled-MLM profile is substantially slower on the longest training pair, so its budget must be measured separately. Project allocation remaining is not established.

Plan roughly 25–40 GB of checkpoint space per active full model, plus retained qualification evidence and inputs. Automatic recovery keeps a previous valid state and the appropriate validation-selected best; a hard kill can lose work since the last committed update. SLURM requeue was exercised with the new trainer, but still depends on scheduler availability. The SIF, software versions, hashes, job accounting, and exact check scope are documented in the qualification artifacts.
