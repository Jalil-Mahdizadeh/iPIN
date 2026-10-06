# V4 production training protocol

Prepared from `improvment-proposal-v4.md` on 2026-10-01. This folder is for training and its engineering qualification. No test split is imported; no automatic benchmark or additional model-selection stage is attached.

## Fixed conditions

| Condition | Backbone | Attention | Production action |
| --- | --- | --- | --- |
| S0 | ESM2 650M | Standard | Reuse v3 `clean-residue-mean-official-seed2`; no new job |
| S1 | ESM2 650M | Chain-aware positional attention | `esm2-chain-aware-official-seed2` |
| C0 | ESM C 600M | Standard | `esmc-standard-official-seed2` |
| C1 | ESM C 600M | Chain-aware positional attention | `esmc-chain-aware-official-seed2` |

All new arms start independently from their pinned unsupervised pretrained weights. The ESM2 model, pair sampler, collator, checkpoint engine and optimization/validation loop are reused from v3. ESM C uses the original SDK blocks and final normalization, with a final-embedding-only forward, block activation checkpointing, and an explicitly selected efficient SDPA backend. The original ESM C sequence decoder is retained in the state dictionary and frozen.

Standard attention uses rotary queries and keys throughout. Chain-aware attention uses rotary scores within chains and unrotated scores across chains, with one joint softmax and the original head-dimension scale. There are no gates, linkers, position gaps, sequence truncation or new interaction data. ESM C's SDK `sequence_id` chain mask is not used to separate the proteins. A broadcast valid-key mask excludes padding, and separately captured chain metadata survives backward recomputation.

The input is `[CLS] A [EOS] B [EOS]`; both AB/BA orientations are processed. The same v3 residue-mean residual MLP (hidden width 128) supplements its ReLU(CLS) linear score. Initial heads match within each backbone pair; the residual output starts at zero. Head input widths follow the backbones' different embedding widths.

## Data, optimization and selection

Official training has **163,085 physical pairs**, including 81,550 positives. Official validation has **59,258**, including 29,628 positives. Row arrays, residue token arrays and offsets match v3 byte-for-byte. Only official train/validation partitions are copied here. The longest inputs have **16,322 / 39,391 tokens**, respectively. All symbols in the prepared arrays are checked against the ESM C tokenizer; the integer identities and pair layout match.

Use seed 2, four GPUs and 64 physical pairs per update (128 orientations), with the original sampler order and rank assignment. Average clean BCE over the two orientations per physical pair and multiply classification loss by 10. No class reweighting, masking or MLM loss is active.

Run exactly **12,745 optimizer updates**, with 2,000 warmup updates and the existing linear decay. LR is 2e-5; AdamW uses FP32 parameters/state, `foreach=False`, weight decay 0.01 on matrix parameters and zero decay on lower-dimensional parameters. Clip gradient norm at 1. Use BF16 autocast, TF32 off and deterministic rank-ordered reduction with independent gradient storage. Keep training/validation token budgets 8,192/16,384 and maximum physical microbatch sizes 4/8; process an oversized single pair whole.

Evaluate the complete official validation set at updates 1,000, 2,000, …, 12,000 and 12,745. Maximum pooled AP of mean AB/BA raw logits selects the checkpoint; exact ties retain the earlier checkpoint. The horizon gives 815,680 physical pair exposures, approximately five data passes. All thirteen validations must complete. No test-driven checkpoint choice, LR sweep, training extension or replacement arm is included.

S0 is independent of preparing/submitting the new arms. Its eventual checkpoint must be selected only after its own complete horizon using the same validation rule. The reused run/configuration identity is pinned in `provenance/reused-control.json`; its final checkpoint is intentionally not guessed here.

## Containers and saved state

S1 uses the existing native/ESM2 ARM64 SIF. C0/C1 use the new ESM C ARM64 SIF (`esm==3.2.3`, Transformers 4.48.1) with the same NVIDIA Torch/CUDA build. Launch workers with `python -m torch.distributed.run` so ESM C workers inherit `/opt/esmc/venv`; the base image's `torchrun` executable uses the base interpreter. Exact image and asset hashes are pinned. The common data are copied, while pretrained assets are explicit symlinks to the existing workspace assets; those source paths and image paths must remain available and unchanged. No download occurs during training.

Atomic, hash-checked checkpoints retain model, optimizer, per-rank Python/NumPy/Torch/CUDA RNG, sampler cursor, scheduler position, best selection, and pending validation. Save at update 0, every 100 updates, every 15 minutes, every validation, completion and safe stop. Keep the current and previous commits and the selected-best dependencies needed by rollback. Corrupt-latest fallback is supported; missing previously committed state, changed configuration/code/data/image, and changed GPU count cause refusal rather than fresh initialization.

Resume keeps the same schedule and four-GPU contract. Recompute the pending validation after interruption before continuing updates. `REQUEST_STOP` stops safely without automatic requeue. A scheduler warning 300 seconds before the time limit creates `REQUEST_REQUEUE`; the worker commits at a safe boundary and the same SLURM job requeues, with a limit of eight restarts. Fatal worker errors are retained and reported rather than silently retried. Run-level locks prevent concurrent writers.

One four-GPU node, 72 CPU cores and 400 GB host RAM is requested for each of the three new production jobs, with an initial 48-hour allocation. This is a wall-time allocation, not a measured whole-run duration. The job exits after training/validation selection. Benchmarking belongs in a later, separate folder.

## Bounded acceptance and stopping point

Qualification consists of mathematical forward/gradient checks, official-SDK embedding equivalence, token/data integrity, one disposable longest-pair optimizer step and longest-validation inference per arm, checkpoint corruption/recovery fixtures, shell stop/requeue fixtures, and actual full-size four-GPU interrupted/resumed trajectories on short inputs. These are engineering checks, not candidate training or a scientific screening stage.

Resume checks compare model, optimizer, RNG, sampler/selection state and saved validation predictions bitwise. They qualify the observed GH200/software configuration and tested workloads, not arbitrary future hardware or software. Full-length execution establishes computational feasibility; it does not establish predictive reliability beyond ESM C's pretrained context.

The requested endpoint is a qualified frozen release, three verified production commands, and **zero production submissions**. A later explicit submission instruction will start those jobs. The primary native-model comparison and practical AP gain criterion remain as stated in the v4 proposal, for the later benchmark task.
