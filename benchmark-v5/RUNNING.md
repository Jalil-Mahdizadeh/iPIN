# Re-running and resuming

## Added human STRING-trained releases

The extension is defined in [HUMAN-RELEASES-ADDENDUM.md](HUMAN-RELEASES-ADDENDUM.md), with immutable checkpoint, runtime, inference-code and data identities in `provenance/human-releases.json`. The previous nine-model sealed artifacts are preserved in `archive/before-human-releases/`. The one-time preparation script must not be rerun.

Resume native humanV11 on a four-GPU allocation with `sbatch benchmark-v5/slurm/native-human.sbatch`; its existing full-length qualification is checked before any production chunk. To reproduce that check on an allocated GPU, use `bash benchmark-v5/scripts/container.sh native python scripts/native_human_bernett.py --qualify`. Resume TUnA human seed 47 with `bash benchmark-v5/scripts/container.sh tuna python scripts/tuna_human_bernett.py`. Both reuse verified atomic outputs. Neither script performs training or selects a model from test scores.

`scripts/finish_human_releases.sh` rebuilds the eleven-model collection, analysis, exposure figures and report once both additional predictors are complete, then seals and verifies the artifacts. Existing nine-model scores and bootstrap samples must remain exactly unchanged.

Run host commands from `/nobackup/proj/disk/theo-storage/personal/jalil/iPIN`. The images are ARM64 and GPU model inference requires an allocated compatible GPU node. Container working directory is already `benchmark-v5`, so Python script arguments begin with `scripts/`. Original sources are mounted read-only; only this benchmark is writable.

D-SCRIPT was added later under [DSCRIPT-ADDENDUM.md](DSCRIPT-ADDENDUM.md), using its separate `container_dscript.sh` and existing length-safe original-model adapter. Its prepared features are verified before four-GPU scoring. Rebuilding those features uses `bash benchmark-v5/scripts/container_dscript.sh python scripts/dscript_benchmark.py --stage prepare`.

No retraining or fetching new model revisions is involved. `provenance/selection.json`, `runtime-inputs.json` and `prepared.json` identify the exact inputs. `scripts/prepare.py` is deliberately one-shot; do not rerun it over completed data. The selected checkpoint files are hard links to the frozen selected checkpoints; do not overwrite or change their permissions.

## Resume neural inference

The completed score files should ordinarily be reused. If a job was interrupted, resubmit the same script and arguments:

```bash
sbatch --job-name=bv5-ipin-esm2 benchmark-v5/slurm/pair.sbatch ipin-esm2 esm2
sbatch --job-name=bv5-ipin-esmc benchmark-v5/slurm/pair.sbatch ipin-esmc esmc
sbatch --job-name=bv5-native-plm benchmark-v5/slurm/pair.sbatch native-plm native
sbatch benchmark-v5/slurm/xpair.sbatch
sbatch benchmark-v5/slurm/dscript.sbatch
```

Each PLM job reruns the saved-DEV qualification, then assigns independent shards to four GPUs without distributed-training networking. Atomic chunks are reused only after signature, checksum and row-identity checks. X-PAIR reuses verified per-sequence features and score chunks; its two checkpoints remain separate. Changing scripts, checkpoint identities or data invalidates their signatures and requires a new explicitly identified run, not silently mixing predictions.

TUnA and RAPPPID are quick single-GPU jobs. Run sequentially on the interactive GPU if recovery is needed:

```bash
bash benchmark-v5/scripts/container.sh tuna python scripts/tuna_benchmark.py
bash benchmark-v5/scripts/container.sh rapppid python scripts/rapppid_benchmark.py
```

The TUnA script reuses verified sequence features and its own completed missing-feature records. RAPPPID reuses its verified singleton endpoint cache. Both rerun native numerical checks before scoring. Cached training parameters are never updated.

## Resume SPRINT

```bash
bash benchmark-v5/scripts/sprint_pipeline.sh
```

This pipeline runs on the host and invokes the dedicated SPRINT image for native executables. HSP generation uses 64 CPU threads; scoring uses native serial arithmetic with the qualified requested-endpoint optimization documented in [SPRINT-EXECUTION-ADDENDUM.md](SPRINT-EXECUTION-ADDENDUM.md). The script uses the optimized binary only if its qualification exists. Completed stages are marked and reused. The native HSP computation is **not internally checkpointed**: interruption before its success marker requires restarting that stage. The same applies to the serial scoring stage. Interrupted append-mode score files are preserved under `logs/` before a fresh attempt. The Python parser preserves native full self-HSPs, including the one 12-residue TRAIN protein, and never drops test rows.

## Recreate analysis

After every predictor is complete:

```bash
bash benchmark-v5/scripts/container.sh analysis python scripts/collect.py
bash benchmark-v5/scripts/container.sh analysis python scripts/analyze.py
bash benchmark-v5/scripts/container.sh analysis python scripts/report.py
```

`collect.py --available` produces an explicitly incomplete collection for monitoring. Final analysis refuses an incomplete roster. Metric and bootstrap results are deterministic for the fixed saved scores and seed. Reanalysis overwrites derived tables/figures only, not inference shards. Use `bash benchmark-v5/scripts/container.sh analysis python scripts/verify_complete.py` to check the final artifact manifest after completion. Add `--inputs` to rehash large external images and checkpoints. After deliberately regenerating analysis, review the results and use `--seal` to refresh the completion manifest.

Initial failed launches are preserved in `provenance/jobs.json` and `logs/`. Two container-specific issues were corrected: commas in `CUDA_VISIBLE_DEVICES` must bypass Apptainer's comma-separated `--env` parser, and independent inference workers must not require a `localhost` distributed rendezvous unavailable inside the isolated image. The successful inference jobs and numerical checks use the corrected launcher.
