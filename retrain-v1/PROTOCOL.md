**Retrain-v1 scientific protocol — fixed before production training, 28 September 2026**

This is the first controlled experiment from [the improvement proposal](../improvment-proposal.md). It tests pooled, symmetric PPI training against a corrected orientation-wise reference on human Bernett data. Both arms start from the original pretrained ESM-2-650M, update all 33 encoder layers, retain full sequences, and use the same cleaned partition and training budget. This is end-to-end PPI fine-tuning, not ESM pretraining from scratch. The released supervised PLM-interact checkpoints are never used as initialization.

The comparison isolates the classification objective **within this corrected, full-length regime**. The reference is not an exact recreation of the authors' original trainer. A difference from the published Bernett score could reflect the data exclusions, length coverage, masking, class weighting, batch/schedule corrections or numerical implementation. It cannot be assigned to symmetry alone. The paired reference-versus-symmetric comparison is the relevant first test.

**Inputs and partition**

Original ESM-2 is pinned to Hugging Face `facebook/esm2_t33_650M_UR50D` revision `08e4846e537177426273712802403f7ba8261b6c`. The authors' benchmark is pinned to `danliu1226/Bernett_benchmarking` revision `5d2ad03baa165c27df32a2eadf066462a2a83073`. URLs, sizes and SHA-256 hashes are in [provenance/downloads.json](provenance/downloads.json); available publisher LFS hashes were checked. Both arms use the user's existing [SIF image](../images/plm-interact/plm-interact-native-arm64-v1.sif).

Exact-sequence and unordered-pair audits found no exact protein overlap between splits and no within-split duplicate or conflicting unordered pairs. MMseqs2 searched validation/test sequences against training, and validation against test, with sensitivity 7.5, identity at least 40%, coverage at least 80% of both sequences, and E-value at most 0.001. The commands and results are preserved in [audit_homology.sh](scripts/audit_homology.sh), [heldout-vs-train.tsv](data/heldout-vs-train.tsv) and [val-vs-test.tsv](data/val-vs-test.tsv). This operational search rule does not exclude all remote homology, shared domains or search false negatives.

Two qualifying cross-split matches were found. Removing the implicated training protein excluded 107 training pairs; removing the implicated validation protein excluded two validation pairs. The test set is unchanged. No model predictions or test labels informed these exclusions. All label values and the existing negative sampling policy are otherwise preserved.

| Split | Retained pairs | Positive pairs | Removed for similarity | Maximum paired input tokens |
| --- | ---: | ---: | ---: | ---: |
| Train | 163,085 | 81,550 | 107 | 16,322 |
| Validation | 59,258 | 29,628 | 2 | 39,391 |
| Test | 52,048 | 26,024 | 0 | 7,426 |

Token counts include CLS and two EOS separators. There are **zero length exclusions or sequence truncations**. In particular, 32,624 retained training pairs exceed 2,196 tokens. [The manifest](data/prepared/manifest.json) records all counts, exclusions and file hashes. The original CSVs and pre-exclusion arrays remain available.

Negatives remain sampled unreported interactions, not experimentally confirmed noninteractions. Broader evidence-aware curation, temporal holdouts, alternative negative sampling and additional seeds remain later work. This first experiment does not implement the full proposed data package. Bernett is an already studied historical benchmark, including our previous reproduction; its test set is not a newly prospective holdout. ESM-2 pretraining sequence exposure is also not excluded by these supervised partitions.

**Models and matched objective**

The encoder reads `CLS A EOS B EOS` and the reversed order. A shared linear classifier acts on the ReLU-transformed CLS representation. One encoder computation per orientation supplies both the classification head and the masked-language-model head. The unused contact head and unused absolute-position table are frozen; every useful encoder parameter, the MLM head and PPI classifier are trainable.

Let `z_AB` and `z_BA` be the two logits for a physical pair. The only scientific difference between arms is:

| Arm | Classification term per physical pair | Primary validation prediction |
| --- | --- | --- |
| `reference-seed2` | `(BCE(z_AB,y) + BCE(z_BA,y))/2` | `z_AB` |
| `symmetric-seed2` | `BCE((z_AB+z_BA)/2,y)` | `(z_AB+z_BA)/2` |

Both losses are `10 * classification + 1 * MLM`, with positive-class weight 1 for this nearly balanced training set. MLM averages masked-token loss within each orientation, then averages the two orientations. Fifteen percent of residues are selected independently, with the usual 80% mask / 10% random canonical residue / 10% unchanged policy; at least one selected residue per chain is required. Chain-specific masks are identical in the two orientations. Classification uses the same masked input as MLM during training. Validation uses complete, unmasked sequences. These settings are explicit protocol choices, not a claim to reproduce every archived training default.

The symmetric predictor is invariant to exchanging the two proteins, subject to numerical precision. Underlying orientation logits can still disagree. `order_logit_gap_mean` in training logs measures that underlying disagreement and is not the gap of the final pooled predictor. Validation also records the reference's pooled-inference AP so that an improvement from averaging predictions is distinguishable from an improvement requiring a pooled training objective.

**Fixed training settings**

Each arm uses seed 2, five epochs, 64 physical pairs per global optimizer update (128 orientation passes), and all retained training rows exactly once per epoch. There are 2,549 updates per epoch and 12,745 total updates; the final partial batch in each epoch is normalized by its actual number of pairs. The run represents 815,425 physical-pair exposures and 1,630,850 orientation exposures.

AdamW uses learning rate 2e-5, weight decay 0.01 for parameters with at least two dimensions, zero decay for biases/normalization vectors, and gradient clipping at norm 1. The learning rate warms up for 2,000 actual optimizer updates and then decays linearly over the remaining updates. Parameters and optimizer moments are FP32; computation uses BF16 autocast, with no loss scaler. The analytic schedule is determined by the saved global update and immutable configuration.

Length buckets are built within shuffled pools, and the resulting batches are shuffled, avoiding a global short-to-long curriculum. Microbatch sizes adapt to padded tokens while keeping the same global batch and normalized gradient. Both orientations always remain together. Sequence masks are deterministic functions of seed, epoch, raw row ID and chain, so restarting does not change corruption or sampling. The code preserves labels alongside row IDs throughout.

The SIF's ESM implementation is adapted to PyTorch memory-efficient attention with explicitly preserved pre-rotary query scaling, bidirectional padding masks and ESM token-dropout behavior. Activation checkpointing is enabled. Qualification found exact FP32 native-versus-math-attention agreement on the test fixture. BF16 native-versus-efficient attention had maximum logit difference 0.0004883 and relative gradient L2 difference 0.06692. These are different floating-point implementations; they are not claimed to have identical native training trajectories. Both scientific arms use the same efficient implementation. Full-length execution also does not by itself establish long-context predictive accuracy.

One Arrhenius node with four distinct GH200 GPUs is used per run. Gradients are gathered, then summed in fixed rank order and averaged, avoiding topology-dependent reduction order after requeueing. This uses more communication than ordinary all-reduce. The final qualification checks exact continuation on a different allocated node, including a partial global batch with an empty local rank. Multi-node training is not used.

**Selection and interpretation**

Full validation runs every 1,000 optimizer updates, at epoch boundaries, and at the final update. The checkpoint with highest validation average precision (AP, the paper's implementation of AUPR) is retained; ties retain the earlier checkpoint. AUROC, Brier score, pooled-reference AP and the orientation gap are logged, with per-pair raw logits saved. No early stopping or automatic hyperparameter search is enabled. Test inference is absent from the trainer.

For subsequent comparison, evaluate the validation-selected reference with ordinary and pooled inference, and the validation-selected symmetric model with pooled inference. Any additional selection for pooled-reference inference must use validation only and be disclosed. Report AP, AUROC, validation-chosen operating points, prevalence, calibration, full coverage and length strata, together with group-aware uncertainty. One seed cannot establish seed-robust improvement. No accuracy gain is claimed from engineering qualification or initial training losses.

**Resume contract**

A checkpoint commits only after a completed optimizer update with cleared gradients. It contains model weights, all AdamW state, global update, epoch, next global batch, examples seen, validation-selection state, pending-validation status, and Python/NumPy/CPU/CUDA RNG state for every rank. Data/code/configuration hashes, software versions and GPU count bind the checkpoint to its run. A pending validation resumes before another training update, including interruption immediately before final validation.

Files are written to a temporary file, flushed and fsynced, atomically renamed, hashed, then committed through an atomic manifest. The last two complete checkpoints and the best validation checkpoint are retained. Resume verifies the checksum and falls back to a retained valid checkpoint if needed. Incomplete files are ignored. Missing all previously committed checkpoints causes an error instead of silently starting again. Changed scientific code/configuration/data or GPU count is rejected. An OS file lock prevents concurrent writers to the same run directory.

A 5-minute Slurm warning requests a checkpoint and requeue. Manual `REQUEST_STOP` pauses without automatically restarting. Abrupt hardware/process failure can lose work since the last complete checkpoint; that work is recomputed on resume. No training batch before the loaded checkpoint is repeated, and none after its cursor is skipped. Filesystem loss is outside this guarantee: `/nobackup` checkpoints are not an off-cluster backup. Exact numerical continuation is qualified for the current image and GH200 hardware with four ranks, not promised across arbitrary software or GPU changes.
