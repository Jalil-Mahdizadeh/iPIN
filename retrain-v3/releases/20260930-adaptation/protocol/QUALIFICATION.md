# V3 qualification evidence

All required initial-stage checks passed before release freezing. Core files `train.py`, `model.py`, `data.py`, `state.py` and `release.py` are byte-identical to the qualified v2 release. V3 adds isolated inputs, clean-objective configurations, staged selection, diagnostics, release packaging and a renamed launcher; it does not claim to have rewritten the training engine.

| Check | New v3 evidence | Scope |
| --- | --- | --- |
| Data / configurations | [v3-contract.json](qualification/v3-contract.json) | All six enabled configurations, 12 head templates and optional cap; hashes, fold reconstruction, homology grouping, full sampling cycles and no test tokens |
| Exact distributed resume | [ddp-clean-comparison.json](qualification/ddp-clean-comparison.json), [ddp.json](qualification/ddp.json) | Full 650M, four physical GPUs, clean BCE at 5e-6, global 64; stop at update 2 of 4 with pending validation; model, optimizer, RNG, sampler and predictions bitwise equal |
| Readout DDP | [ddp.json](qualification/ddp.json) | Both added heads, tiny ESM; uneven accumulation and a rank with zero pair contribution |
| Real signal | [signal-resume.json](qualification/signal-resume.json), [signal-comparison.json](qualification/signal-comparison.json) | Actual SIGUSR1, tiny model on one GPU, fresh process resume, bitwise comparison |
| Checkpoint failures | [checkpoint-faults.json](qualification/checkpoint-faults.json) | Partial write ignored; corrupt pointer/latest fallback; model/optimizer/RNG restoration; changed contract/world rejected; lost state cannot silently restart; old best survives rollback |
| Longest pairs | [full-model-profile.json](qualification/full-model-profile.json) | Full 650M clean CLS and residue heads; forward/backward/optimizer and inference at 16,322 tokens |
| Scheduler wrapper | [launcher-checks.json](qualification/launcher-checks.json) | Real shell with mocked SLURM: six indices, success/failure/manual stop, array-task-specific requeue, default dry run creates no production directory |
| Decision plumbing | [decisions.json](qualification/decisions.json) | Synthetic stage orchestration; incomplete trials refused, zero gains not promoted, failed mechanisms stop confirmation, confounded LR/readout comparisons rejected |
| Production guard | [production-guard.json](qualification/production-guard.json) | Unfrozen production and excessive qualification horizon refused before creating outputs |
| Cheap controls | [cheap-baselines.json](diagnostics/cheap-baselines.json) | Fixed train-only-fitted sequence baselines on all three internal folds |

The four-GPU run was qualification job **3176832**, completed with exit 0. It used 67 short training pairs and eight validation pairs, not full scientific validation. Its scores must never appear as v3 research results. Exact restart was demonstrated under this SIF/hardware/world-size contract, not for arbitrary software versions or GPU counts.

## Resource measurements

| Model / action | Pair tokens, two orientations | Peak allocated GiB | Peak reserved GiB | Seconds |
| --- | ---: | ---: | ---: | ---: |
| Clean CLS, training step | 16,322 | 11.43 | 13.04 | 16.38 |
| Clean CLS, inference | 16,322 | 10.74 | 13.04 | 1.19 |
| Clean residue, training step | 16,322 | 11.44 | 13.02 | 16.15 |
| Clean residue, inference | 16,322 | 10.75 | 13.02 | 1.19 |

Measured on the interactive GH200 with approximately 95 GiB visible memory. The profile used disposable optimizer steps and retained no trained candidate. These are single-pair microbatch measurements; the new four-GPU check used short pairs. They do not constitute an exhaustive bound for every four-GPU long-pair combination. The v2 unchanged engine also completed production and previously qualified full official 39,391-token validation inference; that evidence is explicitly inherited, not rerun here.

Clean CLS has 649,401,601 trainable parameters; the residue head has 650,057,090, an extra 655,489. The matched CLS-MLP has the same extra count by architecture. New checks verify identical initial logits and exclusion of special/padding tokens; the inherited v2 science report also verifies objective gradients, attention equivalence and checkpointed context handling.

## Inherited evidence and integrity

[provenance/v2-qualification](provenance/v2-qualification/) preserves the original v2 reports. The v3 contract audit checks their hashes and exact correspondence of core code and prepared data to the v2 frozen release. V2 performed an actual scheduler requeue on the full model; v3 performed real process interruption and new wrapper simulations, **not a second real scheduler requeue**.

The pinned SIF is `images/plm-interact/plm-interact-native-arm64-v1.sif`, SHA-256 `e064e38053d6dfcacc65a23467d97f75f79ca6095e6f760def4125ccf452ffc2`. It contains Python 3.12.3, PyTorch 2.8.0a0+34c6371d24.nv25.08, CUDA 13 and Transformers 4.40.1. Execution uses offline, local pinned ESM2 assets. Training precision, deterministic reduction and inference orientation policy are fixed in the protocol.

The immutable release records code/configuration/protocol hashes, input manifests, initialization downloads, SIF and required qualification reports. Launcher validation rehashes all referenced inputs before printing a launch plan; the trainer verifies the frozen release and run contract before production initialization. Production directories remain empty at preparation completion.
