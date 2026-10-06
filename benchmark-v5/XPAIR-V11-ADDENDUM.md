# Released X-PAIR V11 extension

The user requested the released human STRING V11-trained X-PAIR checkpoint as a twelfth predictor in both existing v5 human tests. The fixed entry is `xpair-v11`, displayed as **X-PAIR (humanV11)***, using `interaction_dscript.ckpt`. It is the interaction-only model associated with the paper's D-SCRIPT/Sledzieski cross-species benchmark (Fig. 2C), distinct from `interaction_bernett.ckpt` and the STRING v12 multitask default `multitask_xfair.ckpt`.

Sources: [X-PAIR paper and supplementary information](https://doi.org/10.64898/2026.07.20.739596), [author checkpoint directory](https://gitlab.lcqb.upmc.fr/srescalli/X-PAIR/tree/main/pretrained_models), and the released checkpoint's embedded training configuration. The chosen release is fixed by this request; no checkpoint, threshold or inference convention is selected using either test's performance. This addition follows observation of earlier project results and is not a new blind validation.

## Frozen inference

- Keep both 52,048-row tests, their labels and source-row mappings unchanged. Score their 76,918-pair union once with gradients disabled, FP32 parameters and arithmetic, and TF32 disabled.
- Reuse the 3,022 existing full-length Ankh-Large residue embeddings only after sequence identity, length, dtype, provenance and file-hash checks. Do not regenerate embeddings or repeat inference for the previous eleven predictors.
- Load the released interaction-only head strictly. Reuse the qualified upstream cross-attention and scoring modules. Cache only the independent linear projection, with the existing batch-size and attention-cell limits.
- Before production scoring, compare native singleton forwards with projected-cache forwards, swapped inputs and heterogeneous padded batches, including the longest sequence and longest actual test pair. Verify finite outputs and unchanged weights/buffers. Preserve all sequences through 5,183 residues; do not truncate to the source training lengths.
- Use one allocated GPU. Save atomic prediction chunks, signatures, row IDs and checksums so interruption can be resumed without mixing runs.

## Exposure and comparison

The checkpoint metadata identifies human D-SCRIPT training/validation files. The paper identifies their source as STRING v11. Recheck exact and Ankh-normalized sequence membership against the documented public human TRAIN/validation data already used for the humanV11/TUnA audit, applying the checkpoint's recorded length eligibility. The exact processed author TSVs are not bundled locally; this is an audit of documented public source membership, not proof of the complete checkpoint training history.

Mark the model with an asterisk for documented source overlap. Preserve the existing X-PAIR-default, D-SCRIPT and combined endpoint-exclusion definitions and use identical retained rows for every predictor. Explicitly check whether the D-SCRIPT mask also removes all identified X-PAIR V11 exposure. If it does not, document that limitation rather than silently changing the historical subsets.

Update all main and subset figures and their PNG/PDF/SVG variants, result tables, per-pair exports and report for twelve predictors. Keep the existing 1,000 paired protein-bootstrap replicates and seed 20260929. Reuse prior score arrays and bootstrap samples exactly; compute only the added model's samples. Include paired differences against native Bernett, both iPIN models, the other X-PAIR releases, native humanV11 and TUnA human seed 47.

## Preservation and verification

The prior eleven-model sealed artifacts are preserved under `archive/before-xpair-v11/`. Four source files had only nonhuman-folder path edits after the previous seal; their original sealed versions were recovered by a byte-for-byte verified inverse path substitution, and their as-found versions are also archived. `provenance/xpair-v11-archive.json` records this explicitly. Scientific results, figures and predictions matched the prior seal without reconstruction.

Completion requires full finite coverage, unchanged test inputs and prior predictions/metrics/bootstrap samples, independently recomputed exported AP/AUROC, all figures containing the new model, and a refreshed artifact manifest. No retraining is performed.
