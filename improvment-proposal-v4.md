# Improvement proposal v4: backbone × chain-aware attention

**Date:** 2026-10-01. **Scope:** four predefined conditions, one reused control and **three new full-data training runs**. No linker insertion, artificial position gap, additional head comparison, learning-rate screen, new training data or seed campaign is included. This document proposes future retraining; it does not submit it. Benchmarking remains a separate, later task.

The two questions are whether ESM C 600M provides more useful PPI representations than ESM2 650M under our existing supervised recipe, and whether removing artificial sequence distance from cross-chain attention improves either backbone. The combination is worth testing, but neither improvement is assumed in advance.

## 1. The four conditions and the exact reused control

| ID | Backbone initialization | Attention | Common classification head | Action |
| --- | --- | --- | --- | --- |
| S0 | Pretrained ESM2 650M | Standard joint attention | Existing residue-MLP | **Reuse the final full-data v3 run** |
| S1 | Same pretrained ESM2 650M | Chain-aware positional attention | Same residue-MLP | New run |
| C0 | Pretrained ESM C 600M, December 2024 checkpoint | Standard joint attention | Same residue-MLP design | New run |
| C1 | Exactly the same ESM C checkpoint as C0 | Chain-aware positional attention | Same residue-MLP design | New run |

S0 is `retrain-v3/runs/clean-residue-mean-official-seed2`, currently running under job **3206610**, with the training-only release `retrain-v3/releases/20261001-final-training`. Reuse its **maximum official-validation AP checkpoint after the fixed full horizon completes**, together with its complete configuration, training contract and validation records. Its selected update and checkpoint hash are not yet known. They must be recorded from the completed run; do not invent them or choose them from later test performance.

This choice is fixed now, before the final v3 test comparison. It keeps the readout and qualified training engine consistent with the new ESM2 attention arm. The completed v2 clean-BCE model has a CLS-linear head, so it is an additional historical comparator, **not a substitute for this factorial control**. The authors' native PLM-interact checkpoint is also an external target baseline, not S0: its training objective, readout and supervised history differ.

S1 starts from the same **unsupervised ESM2 initialization** as S0, not from the fine-tuned S0 weights. C0 and C1 likewise start independently from the same unsupervised ESM C weights. Otherwise the experiment would confound attention with an extra stage of supervised training. Use seed 2 and identical initial head weights within each backbone pair.

If S0 is interrupted, resume it under its existing contract. Do not launch a duplicate control or silently replace it with a fold checkpoint. Its training was briefly stopped at update 26 to remove an automatic benchmark handoff and then resumed with identical core/configuration; this is one run. See [the current v3 protocol](retrain-v3/FINAL_PROTOCOL.md).

## 2. What the existing evidence supports

V2 clean BCE essentially matched native test ranking: AP **0.690596 versus 0.690319**, difference +0.000277, descriptive paired 95% interval [-0.007765, +0.008157]. It improved on our earlier reference recipes but did not establish AP superiority over native. [V2 benchmark](benchmark-v2/REPORT.md)

V3 residue-MLP improved internal mean selected AP from **0.616488 to 0.620374** (+0.003886), with positive changes on all three folds. It missed the original +0.005 advancement gate. Its final full-data test performance remains unknown. The residue head is retained here as the already chosen common head, not because v3 has demonstrated a native-model win. [V3 development results](retrain-v3/READOUT_RESULTS.md)

All four v2 production runs and all current v3 production runs used `attention_mode: standard`. An optional chain-aware implementation exists and passed earlier engineering checks, but has not received a production scientific test. Its prior qualification does not establish accuracy gains or automatically qualify an ESM C port.

ESM C is a substantial representation change at a similar parameter scale. The authors report improved structural representation performance relative to larger ESM2 models; that supports investigating transfer, not predicting a particular Bernett AP gain. The backbone comparison also changes pretraining data and architecture, so any gain is a practical backbone-upgrade result rather than an isolated architectural effect. [ESM C release and evaluations](https://www.evolutionaryscale.ai/blog/esm-cambrian)

## 3. Precisely define chain-aware attention

All four conditions jointly encode both proteins and allow attention across them. Standard attention applies rotary positional information to every residue pair in the concatenated input. The proposed intervention changes only the treatment of positional information across chains.

Let `q_i` and `k_j` denote the backbone's normal projected queries and keys, including ESM C's original query/key normalization where applicable. Let `R(p)` be its rotary transformation, `c_i` the chain assignment, and `d` the original attention head dimension. Use the score

\[
s_{ij}=\frac{1}{\sqrt d}
\begin{cases}
\big(R(p_i)q_i\big)^\top\big(R(p_j)k_j\big), & c_i=c_j,\\
q_i^\top k_j, & c_i\ne c_j.
\end{cases}
\]

Apply **one joint softmax over all valid keys**, followed by the unchanged value aggregation and output projection. This preserves within-chain sequence relationships while giving cross-chain attention no arbitrary covalent-distance interpretation. It does not block cross-protein information flow, create separate softmax denominators, introduce learned cross-chain gates or replace the backbone's normalization.

The idea has a relevant precedent in PPLM, which combines rotary within-chain scores and non-positional cross-chain scores. PPLM additionally changes pretraining data, attention parameters and downstream prediction; its reported results are not evidence that our minimal intervention alone will win this benchmark. [PPLM, Nature Communications 2026](https://www.nature.com/articles/s41467-026-70457-5)

Implementation requirements are narrow and consequential:

- Keep the paired input layout `[CLS] A [EOS] B [EOS]`, with no inserted residues or large position offsets. Use explicit residue and chain masks. Preserve the current assignment of CLS/first EOS to the first chain and final EOS to the second; padding is excluded from attention keys and readout. Both orientations remain in training and evaluation.
- Preserve each backbone's original scaling, projection layout, normalization, rotary convention, feed-forward blocks and residual scaling. Do not copy ESM2's attention replacement into ESM C without accommodating ESM C's query/key normalization.
- Adapt the existing four-block query/key expansion in [the v3 model implementation](retrain-v3/readout-code/model.py). It represents the above mixed scores in one memory-efficient attention call. **Keep scaling based on the original head dimension**, not the expanded dimension. ESM2's pre-scaled-query implementation is algebraically equivalent and should remain unchanged.
- Do not materialize full per-head attention matrices at production lengths. Do not enable automatic fallback to a quadratic eager implementation. Standard and chain-aware conditions within a backbone must use the same precision and explicitly selected efficient SDPA backend.
- The pinned ESM C SDK's `sequence_id` is used to mask attention between different sequence IDs. **Passing A/B chain IDs there would disable cross-chain attention**, which is not this experiment. Carry chain metadata separately; maintain valid-token masking and joint pair attention.

The published ESM C tokenizer's default template handles a single sequence. Construct and verify the pair layout explicitly. Its available chain-break token is not introduced in v4. Verify token identities for every residue symbol in the prepared data instead of assuming ESM2's token integers are interchangeable. [Pinned SDK distribution](https://pypi.org/project/esm/3.2.3/), [inspected attention source](images/plm-interact-esmc/sources/esm/layers/attention.py), [tokenizer source](images/plm-interact-esmc/sources/esm/tokenization/sequence_tokenizer.py)

## 4. Hold the remaining model and training choices fixed

Use the existing v3 residue-MLP: a linear score from ReLU(CLS), plus a zero-initialized residual MLP using `[CLS, mean(A)+mean(B), |mean(A)-mean(B)|, mean(A)*mean(B)]`. Mean pooling includes real residues only and uses the current FP32 accumulation. Keep hidden width 128 and GELU. ESM2 has embedding width 1280 and ESM C 1152, so input dimensions and head parameter counts necessarily differ; the head design and hidden width are fixed, not its exact parameter count.

Fine-tune the complete sequence backbone. Freeze unused language-model output-head parameters: ESM C's sequence decoder is not the PPI classifier. Use a final-embedding-only training forward that preserves the SDK's final normalized representations but avoids storing every layer's outputs and computing unused MLM logits. Check this adapter numerically against the original SDK before production. Calling the SDK inference convenience methods would disable gradients and/or cast weights to BF16; they are not training entry points.

| Item | Fixed choice for S0/S1/C0/C1 |
| --- | --- |
| Train / validation | Official **163,085 / 59,258** physical pairs; **81,550 / 29,628** positives |
| Data additions, negatives, partition changes | None |
| Length cap / cropping | None; preserve the same real residues and pair coverage |
| Objective | Clean, unweighted binary cross-entropy; average the two orientation losses per pair; classification multiplier 10 |
| Masking / auxiliary MLM | Disabled |
| Initialization | The respective pinned unsupervised pretrained backbone, independently for each arm |
| Seed / global batch | Seed **2**; **64 physical pairs**, 128 orientations per optimizer update |
| Optimizer | FP32 AdamW state, LR **2e-5**, weight decay **0.01**, gradient-norm clip **1**, existing `foreach=False` behavior |
| Schedule | Existing linear warmup/decay; **2,000 warmup updates**, **12,745 total updates** |
| Exposure | **815,680 pair exposures**, approximately five official-data passes; same sampler order and rank assignment |
| Training microbatches | Token budget **8,192**, maximum **4 physical pairs**; oversized single pairs still processed whole |
| Validation microbatches | Token budget **16,384**, maximum **8 physical pairs**; full validation coverage |
| Precision | FP32 weights, BF16 autocast, TF32 disabled |
| Distributed execution | Four GPUs, existing deterministic rank-ordered reduction and independently stored gradients |
| Selection opportunities | Updates 1,000–12,000 at increments of 1,000, plus 12,745: **13** evaluations |
| Selected checkpoint | Maximum pooled official-validation AP; earliest checkpoint wins exact ties |

The same LR is a controlled transfer test, not a claim that it is optimal for ESM C. A negative result would concern this bounded recipe. There is no automatic additional LR sweep. Do not change scheduler length when resuming or extend a weak arm after seeing validation results.

## 5. Container decision and reproducibility

**A separate ARM64 ESM C SIF has been built under `images/plm-interact-esmc/`.** The current ESM2 SIF remains byte-for-byte unchanged, and v3 jobs were not modified. Both C0 and C1 will use the same new image; S0/S1 use the existing image and training stack.

The current image contains Transformers 4.40.1 and no ESM C SDK. The newest SDK examined, `esm 3.4.1.post1`, requires Torch >=2.11,<2.12, whereas the existing GH200 runtime contains the NVIDIA Torch 2.8 development build. Upgrading the shared image would change the reused experiment's numerical environment and introduce an unnecessary CUDA/runtime migration. [Current upstream dependencies](https://github.com/Biohub/esm/blob/43b4548b86762edfa747b07d5f440aad3c33acee/pyproject.toml)

The scoped ESM C runtime uses **official `esm==3.2.3`**, Transformers **4.48.1**, and Tokenizers **0.21.4**, retaining the existing Torch/CUDA build. It loads the original SDK-format ESM C 600M checkpoint directly and strictly, without executing model code fetched dynamically from the Hub. The package's unused `torchtext` dependency has no ARM64 wheel and no code references in the inspected SDK distribution; the image uses a documented ESM-C-specific dependency set rather than claiming to provide every feature of the entire SDK.

Pinned model identity:

- Repository: `biohub/esmc-600m-2024-12` (the original EvolutionaryScale repository redirects here).
- Revision: `e4d83bc7e10fd55c92e598e545f4a76bf04a6e5c`.
- Weight file: `esmc_600m_2024_12_v0.pth`, **2,300,275,866 bytes**.
- SHA-256: `8ef856e1a237ee3f995442df997a962e70057faadecf38fc0c8561bd3c2f4324`, verified against the upstream LFS record.
- Backbone: **36 layers, width 1152, 18 attention heads**. The loaded SDK model has **575,036,992 parameters**, including its sequence decoder; “600M” is the published family name.

Record the base-image hash, definition file, downloaded wheel hashes, installed distribution versions, model hash, source hashes, build log, final SIF hash and runtime test report. Runtime inference and later training must work offline. The new SIF's successful build/import test alone does not qualify a four-GPU resumable trainer or the chain-aware adapter.

**Completed runtime artifact:** [plm-interact-esmc-arm64-v1.sif](images/plm-interact-esmc/plm-interact-esmc-arm64-v1.sif), 13,578,153,984 bytes; SHA-256 `ed6aeac781502090632bc2a705300b161b6a65ccfbcf6c7cf64d8d61d8e5b5ef`. The final packaged model passed strict weight loading, BF16 inference, finite encoder backpropagation and a disposable AdamW step with FP32 parameters/state on a GH200. The test used two short synthetic pairs, saved no candidate checkpoint, and consumed no experimental split. All added package versions match the frozen lock; the documented unused `torchtext` omission is the only dependency-check exception. [Image README](images/plm-interact-esmc/README.md), [GPU report](images/plm-interact-esmc/manifests/sif-gpu-smoke.json).

This verifies the ESM C runtime, not the future chain-aware/PPI adapter, longest-pair feasibility or four-GPU resumability. Those checks remain below. No v4 production training or test benchmarking was initiated by this proposal task.

## 6. Long sequences and a bounded engineering acceptance check

ESM C's documented training context reaches **2,048 tokens**. Our existing full-length policy reaches **16,322 tokens in training** and **39,391 in official validation**. Running beyond that context is extrapolation, even if a rotary implementation accepts the tensor. Its predictive reliability is not certified by a successful memory test. [ESM C training specifications](https://www.evolutionaryscale.ai/blog/esm-cambrian)

Before production, perform one finite engineering acceptance pass—not another scientific screening stage:

1. Verify the common data, explicit pair tokenization, residue/chain masks and full-length retention. Verify the final-embedding adapter against the unchanged official ESM C forward on short and padded batches.
2. Check standard and chain-aware attention against an explicit small-tensor reference, including gradients. Confirm unchanged within-chain scores, nonzero cross-chain information flow, correct scaling, masking and chain metadata during gradient recomputation.
3. Test one real longest training pair through backward and an optimizer step, and the longest official validation pair through inference, for the actual ESM C model and both attention modes. Measure memory/time. These are disposable engineering steps; no resulting checkpoint is a candidate or production initialization.
4. Check four-GPU interrupted/resumed execution of the actual new model/attention paths, including optimizer/RNG/sampler and pending validation. Use the existing short exact-resume harness and retain the v3 gradient-storage fix. New backbone/runtime evidence is required; do not claim the old ESM2 check covers ESM C automatically.

Use gradient checkpointing and the existing small microbatches to fit memory; do not truncate sequences or change the global batch. If the specified full-length computation is infeasible or numerically invalid, stop that implementation and report the concrete limitation. Any alternative length policy would require a separately stated comparison; it is not a hidden fallback in this proposal.

These checks have no biological model-selection scores and add no candidate models. Passing them permits implementation readiness, not automatic submission in the current proposal-writing task.

## 7. Fixed evaluation and what would count as improvement

Training stays in a future `retrain-v4` directory and ends with completed training, saved validation predictions and the selected checkpoint. No trainer loads the test split or starts a benchmark. The current `benchmark-v3` preparation stays separate and deferred; a later v4 comparison should have its own results directory.

In that later task, evaluate each of the four validation-selected conditions on exactly the same **52,048 official test pairs** (26,024 positives), with full sequences and mean AB/BA raw logits. Reuse native, v1/v2 and completed v3 predictions when their row identities, checkpoint hashes, precision and scoring are compatible. Neither the native model nor S0 needs retraining for v4.

Report pooled AP as primary, AUROC and Brier as secondary, plus the existing protein-macro AP and length-stratified diagnostics. Derive any classification threshold from validation only. All four conditions and unsuccessful outcomes remain in the report.

The planned factorial contrasts are:

- **C0 − S0:** backbone change under standard attention.
- **S1 − S0:** chain-aware attention in ESM2.
- **C1 − C0:** chain-aware attention in ESM C.
- **(C1 − C0) − (S1 − S0):** descriptive interaction contrast on the AP scale.

Use paired protein-resampling uncertainty with the same resamples for every model and contrast. Retain the existing 1,000-replicate seed-20260929 procedure for compatibility, including its limitations. The **predeclared primary native-model comparison is C1 versus native**; it is not switched to whichever test score looks best. Other native comparisons and factorial contrasts are descriptive.

For this bounded proposal, define a **meaningful historical-benchmark gain** prospectively as **at least +0.010 absolute pooled AP over native** (one AP percentage point), with a paired 95% interval for that primary difference above zero. This is a practical success criterion, not an estimate of the gain we expect. Report smaller positive changes honestly as smaller gains. Lower Brier alone does not satisfy the primary ranking objective.

Even a passing result remains exploratory: the historical test has informed earlier research, each condition has one training seed, and endpoint bootstrap intervals omit training-seed and some homology/selection uncertainty. Do not call it independent confirmation or a general guarantee. No additional seed campaign or new test dataset is silently appended to v4.

## 8. Arrhenius feasibility and endpoint

Arrhenius supplies ARM64 GH200 nodes with four GPUs; the current interactive allocation exposes approximately 95 GiB GPU memory. Existing full-size v3 training uses roughly 22–24 GiB at logged peaks, providing useful headroom evidence for the model scale. It is not an ESM C or chain-aware memory measurement.

Plan one four-GPU node, 72 CPU cores and 400 GB host memory per new production run. Three simultaneous runs would use **three nodes / twelve GPUs**, after a separate submission instruction. A 48-hour initial allocation per run corresponds to a maximum **576 allocated GPU-hours across those initial allocations**, not an expected runtime prediction. The partition currently permits up to 72 hours per job; exact wall-time requests should use the acceptance-pass timings. The earlier combined chain-aware/separate-MLM profile cannot be used to assign an isolated chain-attention multiplier.

Reserve about **150 GB** for the three runs' rolling full optimizer checkpoints and transient commits, with additional room for the image, pretrained assets, logs and validation outputs. The current 7.8 GB ESM2 checkpoint and retention of up to four payloads provide the sizing reference; measure the actual ESM C checkpoint size before production. Available shared disk is not a personal quota or a verified GPU-allocation balance.

All new runs must retain the existing resumability requirements: atomic hashed model/optimizer/RNG/sampler/selection commits; periodic, validation and signal checkpoints; fixed four-GPU/configuration/software contracts; no silent fresh initialization on missing state; safe requeue; corrupt-latest fallback; and preservation of the selected checkpoint during rollback. Checkpoint resumption never resets the warmup schedule.

**The scientific scope ends with these three new full-data runs and their later fixed comparison.** There is no linker experiment, extra head, data expansion, LR search or automatic rescue branch. A failure to obtain the specified improvement is a valid v4 outcome.
