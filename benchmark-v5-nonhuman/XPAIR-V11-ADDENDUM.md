# Added X-PAIR humanV11 release

At the user's request, add the released `interaction_dscript.ckpt` as **X-PAIR (humanV11)** to the five-species benchmark. This is the same frozen checkpoint already evaluated in `benchmark-v5`, with SHA256 `b2629356df2206e6d29058c4f65daf75e30c9b874da27fa282c282affb6738bf`. Its embedded configuration records human D-SCRIPT/STRING v11 TRAIN and validation, interaction-only training, Ankh-Large, seed 123, epoch 3 (zero-based), and global step 52,724. It is distinct from X-PAIR Bernett and the multitask X-fair default. The user requested this release after earlier benchmark results were available; no release or threshold is selected using these target results.

Preserve all five released datasets and their source order: mouse, fly, worm, and yeast have 55,000 rows each; E. coli has 22,000. The 242,000 observations map to 238,025 unordered sequence pairs over 56,634 exact sequences. Retain original duplicate/conflicting-label observations in the main analysis and retain the existing unique-pair sensitivity analysis. Sequence lengths are at most 800 residues; use their complete supplied strings.

Reuse the 56,634 existing full-length FP32 Ankh features, checking sequence hashes, lengths, dtype, feature-file checksums and encoder/runtime provenance. Load the new interaction checkpoint strictly, freeze all parameters, and run in evaluation mode with FP32 and TF32 disabled. Cache only its independent learned per-residue projection; execute the original cross-attention and interaction head. The projection cache is specific to this checkpoint. Before production scoring, compare native singleton forwards against projected-cache forwards, swapped pairs and padded batches, including the longest supplied pair and a longest-protein self pair. Require maximum absolute logit error below 2e-4 and padded probability error below 1e-5, unchanged model state, full-length inputs and finite outputs. These checks use no target performance metrics.

Run on one allocated GPU with atomic, checksummed pair chunks and an immutable inference signature. Reuse completed chunks only if their checkpoint, code, data and indices match. Preserve all eleven earlier primary predictors' score arrays and bootstrap samples, as well as both historical v2 controls. Existing feature files, checkpoints, source inputs and test labels remain frozen; no retraining or encoder inference is required.

Apply the existing statistical definitions: per-species sklearn AP and AUROC, 1,000 protein-endpoint bootstrap replicates with seed 20261005, and paired differences using identical resampling weights. Compute new bootstrap samples only for X-PAIR V11. Continue reporting the same common known-source and human-source endpoint-unexposed subsets. Audit documented human TRAIN/validation membership with exact and Ankh-normalized sequences and checkpoint length eligibility. Confirm that the existing common masks cover the added release before reusing them; otherwise revise the masks transparently for every model. Public source membership is not proof of the entire checkpoint lineage, homolog exclusion, or PLM pretraining exclusion.

The original frozen protocol and eleven-model roster remain historical records. The effective primary roster adds X-PAIR humanV11 as the twelfth model; the four main figure groups additionally retain v2 length-capped update 8,000 and v2 clean BCE update 4,000, for fourteen displayed models. Regenerate AP, AUROC, common-unexposed and PR/ROC figures, tables, report, per-row exports and completion verification. Preserve the prior sealed artifacts under `archive/before-xpair-v11/`, including the original bytes of files whose paths or documentation changed during the directory rename.

Run from the project root:

```bash
bash benchmark-v5-nonhuman/scripts/run_xpair_v11.sh
bash benchmark-v5-nonhuman/scripts/finish_xpair_v11.sh
```

The first command qualifies and scores only the added checkpoint. The second audits exposure, combines the saved predictions, updates the comparison and verifies its artifacts. A successful inference rerun verifies and reuses the completed prediction file.
