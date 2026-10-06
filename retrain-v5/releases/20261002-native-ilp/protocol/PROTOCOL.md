# V5: native PLM-interact retraining on the frozen HIPPIE/ILP dataset

Exactly two production runs are authorized: `esm2-native-ilp-seed2` and `esmc-native-ilp-seed2`. The user requested this `retraining-v5` directory on 2026-10-02; it supersedes the proposal's provisional `retrain-v5` pathname. Benchmarking remains a separate future task in `benchmark-v5`.

## Architecture and initialization

Both models concatenate each protein pair as `CLS A EOS B EOS` and its reverse. Standard joint attention allows information to cross the protein boundary. The PPI head is the original **CLS → ReLU → one linear logit**. There is no additional residue pooling, MLP readout, symmetry penalty or chain-aware attention modification. Both pretrained MLM decoders are active and receive gradients.

ESM2 initializes from the verified public `facebook/esm2_t33_650M_UR50D` language-model weights. ESMC initializes from the verified original `esmc_600m_2024_12_v0.pth` weights. Neither initializes from a supervised native/v1–v4 PPI checkpoint. The ESMC implementation keeps the original blocks, query/key normalization, rotary convention, final normalization and sequence decoder. It computes final embeddings without retaining every layer's outputs. Both implementations use efficient standard attention and activation checkpointing for full-sequence execution.

## Frozen data and scoring

- TRAIN: **700,764 physical pairs**, exactly 350,382 positives and 350,382 ILP negatives.
- DEV: **165,742 physical pairs**, exactly 82,871 positives and 82,871 ILP negatives.
- All retained sequences are used at full length. Maximum combined model inputs, including three special tokens, are **17,652 TRAIN / 42,878 DEV**.
- The same physical pairs, deterministic order and corruption scheme are used for both backbones. Token arrays are separately exported and verified for each vocabulary.
- The trainer accepts only TRAIN/DEV arrays. Neither frozen test's pair arrays are copied into this folder or loaded by the trainer.
- Selection uses maximum full-DEV pooled AP, computed from the mean of raw AB/BA logits. Exact AP ties retain the earlier update. AUROC, Brier, AB-only AP and orientation differences are diagnostic metrics.

The selected positive partition remains the completed one. The one additional partition attempt found no retention gain; the tighter bound did not change a single training or validation pair. Input provenance is retained under `provenance/` and hashes are verified before launch and resume.

## Shared objective and fixed horizon

The loss is **10 × BCE + 1 × MLM**. BCE uses positive-class weight 1 and averages over both orientations of all physical pairs in an optimizer update. MLM averages over **all supervised masked residue tokens in the entire distributed optimizer update**. Normalizers are global, so changing a GPU's microbatch packing cannot reweight the objective. The encoder forward is shared between classification and MLM on corrupted inputs. Validation and inference use clean inputs.

For each protein occurrence, 15% of residue positions are selected independently, with at least one selected position per protein. Of selected positions, 80% become MASK, 10% are replaced by a uniformly sampled canonical amino-acid token, and 10% remain unchanged. Targets always retain the original residue. CLS/EOS/PAD are never supervised or corrupted. AB and BA share the same residue corruption after reordering. Corruption is a deterministic function of seed, dataset cycle, physical row and protein side, and does not depend on rank or microbatch packing.

These are explicitly chosen v5 training settings. The published architecture is retained, but this is not an exact replay of the unavailable historical native training trajectory. The public native trainer used 22% masking and positive weight 10, whereas the agreed v5 proposal uses the paper's 15% masking recommendation and unweighted balanced-data BCE. Canonical-only random replacements and update-wide MLM normalization are also explicitly pinned here. The separate loss scale of 10 does not constitute a positive-class weight of 10.

| Setting | Fixed value |
| --- | --- |
| Seed | 2, one seed per backbone |
| GPUs | Four GH200 GPUs per run, one node each |
| Physical pairs per update | 64, giving 128 oriented examples |
| Total updates | **54,748** |
| Physical-pair presentations | **3,503,872**: five complete dataset passes plus 52 presentations to finish the last batch |
| Learning rate | 2e-5 |
| Warmup / schedule | 2,000 updates; linear decay over the remaining fixed horizon |
| Optimizer | AdamW; weight decay 0.01 on matrix parameters, zero on biases/one-dimensional parameters |
| Gradient clipping | Global norm 1.0, with nonfinite gradients treated as fatal |
| Precision | FP32 parameters/optimizer; BF16 autocast; no GradScaler |
| Microbatch budgets | 8,192 oriented padded tokens for training; 16,384 for validation; an oversized single pair is processed whole |
| Full-DEV evaluations | **20**, approximately every quarter epoch; exact update list in the frozen configurations |
| First model-selection validation | Update **2,738** |

The common five-pass budget extends the earlier five-pass recipe to the new corpus rather than treating the old 12,745 updates as five epochs of v5. No learning-rate screen, architecture screen, extra seed or test-driven extension is authorized by this protocol.

The longest full-size pairs passed execution on the interactive GH200. A small set of measured length quantiles provides a rough throughput estimate, not a promised walltime: an isolated-pair extrapolation suggests several days per production run, with batching, distributed communication, validation and the long tail affecting actual time. Both runs request 48-hour allocations and can requeue to finish the same fixed horizon, with at most eight automatic restarts. Their horizons will not be reset or extended on requeue.

## Resumability and stopping

The inherited checkpoint engine writes atomic, hashed commits containing model and active MLM decoder weights, optimizer state, per-rank Python/NumPy/Torch/CUDA RNG, sampler cycle and cursor, examples seen, update counter, pending validation, and best-checkpoint selection. The immutable schedule plus saved update counter determines the exact scheduler state. Masking is stateless from saved sample identities and cycle. A resume requires the same data, code, configuration, image, pretrained initialization identities and four-GPU count.

An early checkpoint is committed after update 10; regular commits occur every 100 updates or 15 minutes, at validation boundaries, and on a safe stop or time-limit warning. Before full validation, a checkpoint marks validation pending. An interrupted validation is rerun from that committed model; it never publishes partial coverage. The checkpoint engine retains a fallback and all best-checkpoint dependencies needed for rollback. Corrupt or incomplete commits cannot silently reset training.

`REQUEST_STOP` requests a safe manual stop. `REQUEST_REQUEUE` is created by scheduler warning/termination handling and causes a safe checkpoint followed by requeue, unless a fatal worker failure occurs. Fatal failures preserve committed state and require inspection. Each run has an exclusive lock; the launcher refuses duplicates, completed runs or accidental restarts of manually stopped runs.

Production is enabled only by an immutable release bound to passing adapter/objective, longest-sequence, four-GPU exact-resume, checkpoint-fault, wrapper and production-guard checks. Disposable qualification weights never initialize production. The user's requested handoff is reached when both production jobs have progressed through multiple real optimizer updates, emitted finite BCE/MLM losses and gradient norms, used four distinct GPUs, and committed a verified checkpoint. This confirms execution health; model performance remains unknown until full validation and later benchmarking.
