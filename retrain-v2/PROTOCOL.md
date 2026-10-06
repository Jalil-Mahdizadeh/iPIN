**Prespecified v2 training protocol**

This implements the staged experiment sequence in [proposal v2](../improvment-proposal-v2.md). Its immediate target is human Bernett PPI ranking. It is not an exact reconstruction of the unavailable native training run, and it does not establish improvement in the paper's separate cross-species task.

The initial screen is deliberately limited to four runs. It establishes matched controls before spending on combinations or expanded data. The remaining configurations are templates for later stages, not an automatic job matrix.

| Enabled first | Coverage | Positive-class weight | Classification input | Auxiliary MLM |
| --- | --- | ---: | --- | --- |
| `reference-official-seed2` | All pairs | 1 | Masked | Same masked pass |
| `capped-official-seed2` | Combined residues ≤2,193 | 1 | Masked | Same masked pass |
| `positive10-official-seed2` | All pairs | 10 | Masked | Same masked pass |
| `clean-bce-official-seed2` | All pairs | 1 | Clean | None |

The capped/reference comparison isolates coverage at matched optimizer updates and pair exposure. Positive10/reference isolates positive weighting. Clean-BCE/reference tests the combined removal of corruption and MLM; it **does not isolate which of those two changes caused any difference**. The subsequent masked-BCE and clean-plus-MLM arms complete that objective comparison. A capped-positive10 arm is available if the coverage/weight interaction needs resolution. None of these controls is labelled “native retraining.”

**Fixed training contract**

- Initialization: the same pinned `facebook/esm2_t33_650M_UR50D` weights, revision `08e4846e537177426273712802403f7ba8261b6c`. All encoder layers are trainable; unused contact and absolute-position parameters are frozen. There is no continuation from native or v1 PPI weights.
- Four GPUs on one Arrhenius node, seed 2, 12,745 optimizer updates, and exactly 64 physical pairs per global update. Both orientations are evaluated, yielding 128 sequences. There are 815,680 pair exposures: 5.001564 equivalent passes through full training or 6.252290 through the capped subset.
- AdamW, FP32 parameters and optimizer state, BF16 autocast, learning rate 2e-5, 2,000 warmup updates, linear decay, weight decay 0.01 on matrix parameters and zero on vectors, gradient norm limit 1. TF32 is disabled. No GradScaler is used for BF16.
- Length-bucketed, deterministic pair stream with no dropped remainder. A batch crossing an epoch boundary is filled from the next cycle. Stateless masking depends on seed, cycle, original source row, and physical chain; corresponding residues have identical corruption in A–B and B–A.
- Corruption selects 15% of residues independently, with at least one selected residue per chain. At selected positions, 80% become MASK, 10% random standard residue IDs, and 10% remain unchanged. Specials/padding never become MLM targets. ESM's pretrained token-dropout convention remains shared across arms.
- Per-pair classification loss is the mean of the two orientation BCE losses. This is **not** BCE on a pooled logit. Positive weighting acts inside each BCE. The total objective is `10 × BCE + MLM_weight × MLM`; MLM averages selected tokens within each orientation, then orientations within a physical pair. The global optimizer gradient averages physical pairs, including unequal microbatch allocations correctly.
- All inference uses clean sequences. A pair's ranking score is the mean of A–B and B–A logits. AP uses this raw pooled score; AUROC and Brier are secondary. Brier is computed from its sigmoid on benchmark labels.
- Validate on the complete uncapped development partition every 1,000 updates and at the final update. Select the maximum pooled validation AP, keeping the earlier checkpoint on exact ties. No early stopping or test-based reselection. Preserved validation predictions allow paired analyses.
- Save resumable checkpoints every 100 updates, at least every 15 minutes at an update boundary, on validation, on completion, and on handled stop/requeue requests. Keep recent fallback state plus the selected best and any best checkpoint needed by the fallback.

Compared with v1, the fixed-size stream adds 255 exposures over its five-epoch run and changes epoch-boundary batching. V2 also selects by pooled AP; v1 selected by its original orientation rule. Therefore the existing v1 checkpoint is historical evidence, not a substitute for this fresh reference. For identical inputs and weights, the reference model's forward pass, objective, and gradients were verified against v1.

The published trainer's positive weight 10 is a motivation for a control, not evidence that the released Bernett checkpoint used precisely that configuration. Its selected update, full schedule, and complete run metadata remain unverified.

**Implemented subsequent interventions**

| Family | Configurations | Control and intended inference |
| --- | --- | --- |
| Objective | `masked-bce`, `clean-plus-mlm` | Complete masked/unmasked × MLM/no-MLM comparisons. Clean-plus-MLM uses separate encoder passes. |
| Adaptation | `lower-lr` | Fixed 5e-6 versus 2e-5, all other reference settings shared. This first version does not add adapters or a layer-unfreezing schedule. |
| Readout | `cls-mlp`, `residue-readout` | Compare equal additional parameter counts against the CLS-linear baseline. |
| Attention | `chain-aware` | Change cross-chain positional treatment with the same reference readout/objective. |
| Combination | `chain-residue` | Combine only after component evidence is reviewed; no automatic promotion. |

The residue readout uses CLS together with symmetric features of the two masked-by-position residue means: their sum, absolute difference, and elementwise product. Special and padding positions are excluded. It adds a residual MLP to native ReLU(CLS)-linear classification. Its last layer is initialized to zero, preserving initial baseline logits. The CLS-only MLP control has exactly the same added parameter count (655,489 for ESM-2 650M), with four hidden blocks averaged before output. This controls parameter count, not every possible feature geometry.

Chain-aware attention retains rotary scores within a chain and uses unrotated scores across chains, with **one joint softmax**. An algebraic expansion of Q/K from 64 to 256 features per head expresses this through efficient SDPA while keeping V at 64 features. No dense length-squared chain mask is built. The two chains retain the native concatenation layout; CLS and the first separator belong to the first chain, the final EOS to the second. This is an implementation inspired by the proposal's mechanism, not a claim of exact PPLM reproduction. Output pooling across orientations remains necessary because the encoder/CLS layout is not exactly permutation invariant.

Non-reentrant gradient checkpointing captures chain metadata per forward and per recomputation. Qualification specifically covers two live graphs with different chain boundaries and the separate clean/MLM passes. This avoids a subtle error where recomputation could otherwise use another forward's chain layout.

**Selection and staged spending**

The 12 arm definitions each have an official-split configuration and three internal homology-group fold configurations. Each fold comparison uses 6,000 updates, 1,000 warmup updates, and 64 pairs per update, shared across its control and candidate. Different folds have different row counts, so effective epochs differ; the matched quantity is pair exposure. All fold models start from ESM-2, since native PPI weights have already seen these labels.

Before promoting an intervention, require mean fold AP gain at least 0.005, gains on at least two of three folds, no single-fold AP decline greater than 0.01, and mean AUROC decline no worse than 0.005. These are screening rules, not significance tests. Review length strata, partner retrieval, and nuisance controls; retain unfavorable comparisons. The [campaign configuration](configs/campaign.json) freezes these thresholds.

The [development comparison script](scripts/compare_development.py) requires completed runs, their recorded validation-selected checkpoints, aligned rows/labels, and matched horizons. It reports AP/AUROC/Brier, length strata, macro partner AP among proteins with both classes, and paired Poisson protein-bootstrap AP intervals. Those intervals condition on the observed graph and exclude seed and selection uncertainty. It does not read test data. The three-fold promotion decision remains an explicit scientific review, not automatic training submission.

Candidate/control confirmation later requires seeds 2, 17, and 42 and a frozen independent holdout. Prepare those chosen configurations only after fixing the candidate; do not launch all candidate-by-seed combinations opportunistically. Target at least +0.01 AP with a paired interval excluding zero and stable seed behavior. Use common efficient FP32 inference where feasible, or qualify a shared precision protocol before opening outcomes. No such improvement is established by the qualification metrics.

The historical Bernett test has already informed research decisions. Any eventual reuse is exploratory historical comparison. An improved-generalization claim additionally requires the evidence/exposure audits and independently reserved holdout described in [the data card](DATA_CARD.md). Expanded evidence-aware data and experimental interface supervision remain conditional; neither is silently substituted by the old benchmark.

**Resume contract and failure handling**

Each checkpoint contains model, optimizer including current LR, complete RNG state for every rank, sampler cycle/offset, exposure count, best-validation state, and pending-validation state. The LR schedule is a deterministic function of the recorded update and frozen config. Atomic write/rename, fsync, SHA-256 sidecars, and retained fallback checkpoints prevent loading partial or corrupted state. A rollback restores the best pointer appropriate to that older state.

Resume refuses changed code/config/data, changed initialization manifest, changed GPU count, or missing previously committed checkpoints. Four-GPU gradients use an explicit rank-ordered FP32 reduction to preserve exact continuation across DDP bucket rebuilds. An exclusive run lock prevents simultaneous writers. A hard kill can lose uncommitted updates; replay begins from the last valid committed boundary. Bitwise equality is qualified for this fixed SIF/hardware/software/world-size contract, not arbitrary platforms.

SIGUSR1/SIGTERM and request files ask the worker to finish a safe update and commit. Interrupted validation stays pending and must complete before the next update, including after the final optimizer step. SLURM supplies USR1 five minutes before walltime. The wrapper requeues only the interrupted array task, with an eight-restart limit; manual REQUEST_STOP takes precedence. Requeue requires scheduler policy/account availability. Failed workers retain state and return a nonzero exit instead of silently restarting from initialization.
