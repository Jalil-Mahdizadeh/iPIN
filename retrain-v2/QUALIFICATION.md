**Qualification completed on Arrhenius, 29 September 2026**

The initial production campaign was qualified with the frozen SIF/code/data contract. At the preparation handoff, no v2 production retraining had been submitted; every optimizer step described in this report was a bounded, disposable check under `qualification/`. Their small-sample AP values are not research results. The user subsequently authorized production array **3130215**; see [the launch status](RUN_STATUS.md).

| Check | Outcome and scope | Evidence |
| --- | --- | --- |
| Inputs and folds | Original train/validation rows preserved; cap count verified; test sequences removed from token cache; fold unions reconstruct parent rows; no exact or detected component edge crosses a fold boundary. | [Science report](qualification/science.json), [token-subset record](provenance/token-subset.json) |
| Sampling and masking | Every row appears once per cycle; fixed-size batches cross cycles without loss; cursor reproduces the next batch; corruption is stateless and coupled across orientations; special tokens are excluded. | [Science report](qualification/science.json) |
| Objectives | All four masking/MLM combinations and positive weight 10 agree with independently computed losses. All expected parameters receive gradients; disabling MLM retains trainable shared encoder embeddings. | [Science report](qualification/science.json) |
| Reference compatibility | V1 and v2 reference forward outputs, losses, and parameter gradients are bitwise identical for identical tiny-model weights and inputs in FP32. This does not assert identical full training schedules. | [Science report](qualification/science.json) |
| Chain-aware attention | Expanded efficient attention agrees with a dense FP32 oracle, including padding. Maximum absolute output/gradient differences: 5.42e-6 / 6.96e-5, within specified tolerances. | [Science report](qualification/science.json) |
| Readout and recomputation | CLS-only/residue heads have equal added parameter counts and preserve baseline logits at initialization. Two live graphs with different chain metadata and decoupled MLM yield gradients matching non-checkpointed computation. | [Science report](qualification/science.json) |
| Four-GPU reference resume | Five updates uninterrupted versus stopping after update two and resuming: model, optimizer, RNG, sampler, best selection, and validation predictions are bitwise identical. Includes a zero-contribution rank (3 pairs on 4 GPUs). | [Reference comparison](qualification/ddp-reference-comparison.json) |
| Four-GPU experimental resume | Full 650M chain-aware/residue/clean-plus-MLM model, uneven microbatch accumulation (9 pairs on 4 GPUs). Interrupted final validation is recovered without losing the final evaluation. | [Experimental comparison](qualification/ddp-experimental-comparison.json), [DDP report](qualification/ddp.json) |
| Real scheduler requeue | The same full experimental model stopped after update one via batch SIGUSR1, committed, requeued its array task, restarted, and performed updates two and three. Every compared model/optimizer/RNG/state tensor and validation prediction matches the uninterrupted run bitwise. | [Requeue report](qualification/scheduler-requeue.json), [comparison](qualification/ddp-experimental-requeue-comparison.json) |
| Corruption and rollback | Partial payload ignored; broken latest pointer recovered via sidecar; same-size payload corruption falls back; model/optimizer/RNG restore exactly; prior best is retained/restored; changed contract/world size and lost committed state are refused. | [Fault report](qualification/checkpoint-faults.json) |
| Stop and wrapper handling | Real single-GPU SIGUSR1 resumes exactly. Shell tests cover success, worker failure, manual stop, and array-task-specific requeue. | [Signal report](qualification/signal-resume.json), [wrapper checks](qualification/launcher-checks.json) |
| Production safeguards | Unfrozen production is rejected before creating a run or initializing CUDA. Default launcher invokes no submission command and creates no run directory. All 48 experiment configs have only their permitted differences. | [Production guard](qualification/production-guard.json), [config audit](qualification/configurations.json) |

**Measured full-model memory and runtime**

The single-GPU profile used the real longest training/validation pairs, two orientations, FP32 parameters/AdamW, BF16 autocast, gradient checkpointing, and the existing SIF. The device exposed 95.0 GiB. Training included forward, backward, clipping, and an optimizer step; validation was inference only.

| Model / operation | Input tokens | Seconds | Peak allocated GiB | Peak reserved GiB |
| --- | ---: | ---: | ---: | ---: |
| Reference, longest training pair | 16,322 | 16.57 | 11.44 | 13.04 |
| Reference, longest validation pair | 39,391 | 6.07 | 13.76 | 18.31 |
| Chain-aware + residue + separate MLM, longest training pair | 16,322 | 64.85 | 17.29 | 19.74 |
| Chain-aware + residue, longest validation pair | 39,391 | 12.29 | 16.38 | 26.14 |

[Raw profile](qualification/full-model-profile.json). Reserved peaks are allocator observations, not a production memory ceiling. The full-model DDP checks observed approximately 19.75 GiB peak allocated on rank zero for their short inputs, including communication buffers. Longest-pair and DDP tests were separate workloads. The approximately 3.9× training-time ratio combines attention, readout, and a second encoder pass; it does not isolate attention overhead or predict a full epoch's ratio.

**Execution record**

- Qualification job **3128847**: four GPUs on n202, completed successfully in 6 minutes 6 seconds. [Accounting](provenance/qualification-accounting-3128847.txt), [log](logs/qualification-3128847.log).
- Requeue qualification array task **3129099_0**: first allocation 53 seconds, one real requeue, resumed allocation 1 minute 31 seconds, final exit 0. [Accounting including both attempts](provenance/qualification-accounting-3129099.txt), [log](logs/requeue-qualification-3129099_0.log). “Cancelled due to job requeue” in the first attempt is the expected scheduler transition, not a failed recovery.
- These two tests consumed approximately **0.57 allocated GPU-hours**, excluding interactive-node checks. This uses top-level allocation elapsed times without double-counting batch/extern steps. They were not production jobs. Qualification uses at most five updates per invocation and tiny train/validation subsets; the profile performed two disposable full-model optimizer steps.
- Interactive checks used the current single-GPU allocation. [Software/device versions](provenance/environment.json). Production reuses the same SIF; no image rebuild or package installation was required.

The pinned SIF SHA-256 is `e064e38053d6dfcacc65a23467d97f75f79ca6095e6f760def4125ccf452ffc2`. The [release manifest](releases/20260929-initial-screen/release.json) records hashes of code, configurations, input manifests, and qualification reports. Source changes require fresh qualification before freezing another release.

The [frozen runtime verification](qualification/frozen-release-verification.json) passed, including refusal of a mutable configuration. The actual launcher completed its [dry run](provenance/production-launch-plan.json), verified the frozen inputs, and submitted zero jobs. Its shell-control checks used simulated worker/scheduler commands; the separate full-model array test above supplied the real scheduler-requeue evidence.

**Limits of this qualification**

These checks establish numerical correctness and tested recovery paths for the prepared implementation. They do not establish convergence, an AP gain, absence of every possible hardware/storage failure, or sufficient remaining project allocation. A hard failure may discard updates since the last commit. Bitwise restart equivalence applies to the tested GH200/SIF/world-size contract. New data, external confirmation, arbitrary architecture combinations, or a different software/hardware stack need their own qualification and scientific review.
