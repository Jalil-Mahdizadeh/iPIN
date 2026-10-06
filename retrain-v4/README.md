# V4 retraining preparation

**Status:** S1 and C1 were **stopped at the user's request on 2026-10-01 at 19:01 UTC**, after saving resumable checkpoints at updates **3,070 / 3,015**. Both allocations ended cleanly, releasing eight GPUs. C0 and the reused v3 standard-attention control remain running. See the verified [stop record](provenance/chain-aware-stop-20261001T190127.json). The chain-aware runs did not complete the planned horizon; their negative results and original arm identities are retained. `REQUEST_STOP` markers prevent accidental restart.

All three production jobs were originally submitted on **2026-10-01 at 10:38 UTC** following the user's instruction. The [submission record](provenance/submission-20261001T103847826554.json) binds their commands and IDs to the verified [20261001-production release](releases/20261001-production/release.json). Implementation, full-length execution, recovery checks and actual four-GPU bitwise resumption passed before submission. The [preparation report](PREPARATION_REPORT.md) records that earlier handoff; the [dry-run plan](provenance/production-launch-plan.json) is retained as historical evidence.

| Condition | Job ID | Initial node | Status at stop verification | Production log |
| --- | --- | --- | --- | --- |
| S1: ESM2 chain-aware | 3210885 | n563 | Stopped, update 3,070 | [production-3210885.log](logs/production-3210885.log) |
| C0: ESM C standard | 3210886 | n154 | Running | [production-3210886.log](logs/production-3210886.log) |
| C1: ESM C chain-aware | 3210887 | n165 | Stopped, update 3,015 | [production-3210887.log](logs/production-3210887.log) |

**Investigation, 1 October:** the chain-aware arms show poor learning. Controlled diagnostics identify substantial disruption of pretrained sequence modeling under the hard cross-chain attention switch, with no observed numerical or gradient-flow failure in the tested cases. See [CHAIN_ATTENTION_INVESTIGATION.md](CHAIN_ATTENTION_INVESTIGATION.md). The investigation itself left production unchanged; the subsequent stop was explicitly authorized by the user and is recorded above.

The agreed experiment contains four conditions, with **three new production runs**. The reused control is the final full-data v3 ESM2 standard-attention residue-MLP run; it is not retrained here. The three new runs are ESM2 chain-aware (S1), ESM C standard (C0), and ESM C chain-aware (C1). Each uses the common residue-MLP design, full official data and fixed v3 training recipe. See [PROTOCOL.md](PROTOCOL.md) and [the v4 proposal](../improvment-proposal-v4.md).

## Scope and implemented behavior

- Official train/validation: **163,085 / 59,258 pairs**, unchanged from v3. No length cap or new negative sampling. No test split is imported or loaded.
- Seed 2; LR 2e-5; 2,000 warmup updates; 12,745 total updates; four GPUs and 64 physical pairs per update. FP32 weights/AdamW state and BF16 autocast. Clean BCE, no MLM/masking.
- Both orientations in training and validation. Select the maximum full-validation AP checkpoint from the fixed thirteen evaluations; ties retain the earlier checkpoint.
- Original ESM2 model/data/checkpoint implementation and optimization/validation loop retained. ESM C uses final-only embeddings with its original block operations, query/key normalization and rotary convention. Unused sequence-decoder parameters are frozen.
- Chain-aware attention removes the rotary distance term only across chains. Both chains share one attention softmax. The implementation avoids quadratic score/mask allocation, preserves the original scale, and captures chain metadata for backward recomputation.
- Training jobs end after completion and validation selection. No benchmark handoff is attached.

## Completed engineering checks

[Model and data report](qualification/model-and-data.json): exact original-SDK valid-token embeddings on the tested full pretrained ESM C inputs; small-model embedding and encoder-gradient equivalence; both attention modes against a dense forward/gradient reference; cross-chain information flow and padding exclusion; checkpoint recomputation with different chain metadata; identical initial heads within each backbone pair; unchanged official rows and verified residue-token identities. The ESM2 head match is also recorded in its full-length report.

The longest official pairs ran without cropping, with **both orientations** and the actual full-size models. Training measurements include backward and one disposable AdamW step; validation measurements retain the optimizer state. These are single-pair engineering measurements, not whole-run duration estimates or accuracy results.

| New run | Longest training: 16,322 tokens | Longest validation: 39,391 tokens |
| --- | --- | --- |
| S1: ESM2 chain-aware | 33.22 s; 11.91 GiB peak allocated | 12.35 s; 16.37 GiB |
| C0: ESM C standard | 17.83 s; 9.68 GiB | 5.97 s; 10.45 GiB |
| C1: ESM C chain-aware | 35.13 s; 10.10 GiB | 12.09 s; 11.32 GiB |

[ESM2 full-length report](qualification/full-length-esm2.json), [ESM C full-length report](qualification/full-length-esmc.json). Successful execution does not establish predictive reliability beyond ESM C's pretrained context.

[Checkpoint fault report](qualification/checkpoint-faults.json): incomplete files ignored, corrupt-pointer/sidecar recovery, corrupt-latest fallback, exact restored model/optimizer/RNG, refusal of changed contracts or GPU counts, refusal of missing committed state, and preservation/restoration of the previous best checkpoint during rollback.

[Wrapper report](qualification/wrapper.json): ten simulated cases covering completion, fatal failure, safe stop, warning/requeue, requeue limit, missing completion and duplicate lock. Scheduler calls were replaced by fixtures for this test. The actual distributed checks used short qualification jobs, recorded in [the initial submission](provenance/qualification-submission-3209856.json) and [the corrected submission](provenance/qualification-submission-3210358.json); neither can submit production jobs.

[Four-GPU resume report](qualification/ddp-resume.json): all three actual full-size models passed uninterrupted four-update training versus stopping after update two with validation pending and resuming. Model weights, optimizer, per-rank RNG, sampler/selection state and validation predictions were bitwise identical. These checks used short inputs, eleven training pairs, eight validation pairs and an uneven nine-pair global batch to exercise accumulation. The qualification checkpoints are separate from production and are never used to initialize it.

[Production guards](qualification/production-guards.json) reject changed fixed hyperparameters, unfrozen code, and qualification/short-run flags applied to production before an output directory is created. [Worker-environment check](qualification/worker-environment.json) verifies the ESM C interpreter and imports in actual distributed worker processes.

The initial distributed job passed S1, then exposed an ESM C launcher issue before ESM C model construction: the inherited `torchrun` executable bypassed the ESM C virtual environment. The launcher now uses `python -m torch.distributed.run`. The correction changes no model/training core, retains the passing S1 result, and is recorded in [launcher-correction.json](provenance/launcher-correction.json), with the original job/code/log preserved. Only the remaining C0/C1 checks were repeated under the corrected entry point.

## Launch and resume

The frozen release is finalized. The launcher defaults to a **dry run**, which has passed with exactly three commands:

```bash
python3 -B retrain-v4/scripts/launch.py
```

The production command below was executed once after explicit submission authorization. The launcher rejects runs that are already queued or running:

```bash
python3 -B retrain-v4/scripts/launch.py --submit
```

That submits exactly three independent jobs, each requesting one node with four GPUs, 72 CPU cores, 400 GB host RAM and an initial 48-hour allocation. Queue/allocation availability controls concurrency. The launcher verifies the frozen code/configurations, qualification evidence, data, pretrained weights and images. Completed, manually stopped or already queued/running runs are rejected. It never submits the reused v3 control.

Checkpoints are atomic and hashed, with periodic/validation/signal commits and retained fallback/best dependencies. Resumption restores optimizer, RNG, sampler, pending validation and the unchanged schedule, using the same four-GPU contract. The worker has independent gradient storage and deterministic rank-ordered reduction. Use `REQUEST_STOP` in the relevant run directory for a safe manual stop. Inspect the retained state and remove that marker only for intentional continuation; then use `--run RUN_NAME --submit`. Time-limit warnings checkpoint and requeue the same job, with a limit of eight automatic restarts. Fatal errors preserve state and require investigation.

## Files and dependencies

- `scripts/`: implementation, qualification and launch tools.
- `configs/`: the three full-horizon production configurations.
- `qualification/`: disposable engineering artifacts; never used as production initialization.
- `releases/`: the frozen qualified release; `CURRENT` selects `20261001-production`.
- `runs/`: production output only; the three submitted jobs now write their state here.
- `provenance/`: image/weight/data identities and the reused v3 control contract.

Prepared official arrays are copied into this folder. `assets/esm2` points to the existing v3 pretrained ESM2 assets; `assets/esmc` points to the verified pretrained ESM C assets under `images/plm-interact-esmc/`. The pinned SIF files remain in `images/plm-interact/` and `images/plm-interact-esmc/`. These linked assets/images must remain present and byte-identical. Their hashes are verified on launch; training requires no downloads.

The ESM C image's unused SDK `torchtext` dependency exception remains documented in [its image README](../images/plm-interact-esmc/README.md). No package or existing experiment is modified by this preparation.
