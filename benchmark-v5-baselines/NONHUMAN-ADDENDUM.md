# Five-species extension: v5 and original Bernett TRAIN

Date: 2026-10-07. This exploratory extension was requested after the neural transfer results and FADI/TRIQ audits were known. Freeze this addendum and the inherited configuration before new baseline scores are calculated. Do not tune a predictor, its sign, similarity cutoff, or graph using species-test labels.

## References and evaluated observations

- **v5 TRAIN:** reuse the saved degree counts/propensities, positive interaction graph, and fitted `models/sequence-propensity.joblib`, with their original checksums. No v5 refitting.
- **Original Bernett TRAIN:** the complete, uncapped `retrain-v1/data/raw/pairs_uniprot_seqs_train.csv` release, verified against its download receipt. Retain all original rows/labels and both positive and negative endpoints. Fit the same fixed sequence-propensity regressor once; form TRAIN counts and the positive graph with the original routines. This is the source dataset, not a reconstruction of the capped or cleaned neural training subsets.
- Evaluate each reference on the exact five datasets in `benchmark-v5-nonhuman`: mouse, fly, worm, yeast and E. coli. Preserve every source row (242,000 total), duplicates, self-pairs and conflicting-label observations. Preserve raw complete strings and source row order. Primary prevalence is 1/11 in each species. No source-row exclusion depends on scores.
- Retain all existing human validation/original/ILP results unchanged. The extension does not fit a Bernett model to a human holdout or substitute one human validation set for another.
- Reuse the frozen nonhuman neural predictions for v5 ESM2/ESMC, v2 clean BCE, v2 length-capped, and native PLM-interact Bernett. Verify their exact row mapping, hashes and previously reported AP/AUROC. No new neural inference or checkpoint selection.

## Methods held fixed

Use the configuration in the original [protocol](PROTOCOL.md) and `scripts/common.py` unchanged: smoothed positive/negative TRAIN log-degree ratio; 422 sequence features; 300-iteration histogram gradient boosting regressor; seed 47; and three methods (homology-transferred degree, sequence-only propensity, interolog lookup). Include exact TRAIN-degree lookup as the existing reference control. Both protein-propensity methods add independently computed endpoint scores.

Positive graphs contain unique unordered TRAIN-positive sequence pairs. Degree counts retain source multiplicities and count a self-pair once. Proteins unseen in TRAIN receive zero for the exact-degree reference; no alignment hit receives the reference's mean TRAIN log-degree ratio for homology transfer. No interolog support gives zero. Do not remove zero/tied scores.

Search separately against each reference's TRAIN, using the pinned CPU MMseqs binary and original parameters: sensitivity 7.5, E≤1e-5, identity≥25%, coverage≥50% of both proteins, maximum 1,000 prefilter candidates, alignment mode 3, one iteration, top five retained matches. FADI's 80%-coverage/common-reference search is not sufficient for this different baseline definition and will not substitute for it.

Preserve the existing weighting convention, `reported fident * sqrt(reported qcov * reported tcov)`, and ordering by exact status, bit score, weight, then target sequence hash. The legacy reports use rounded MMseqs exports; do not silently replace those weights and call the methods identical. Export integer identity/alignment coordinates additionally for a precision diagnostic. Internal MMseqs cutoffs remain unchanged. Search E-values use the respective TRAIN database, as in the existing baselines. Exact TRAIN matches are inserted independently of search masking.

Validate the new shared parser by reproducing **every** saved v5 homolog index/weight from its old alignment file. Validate the loaded v5 regressor against saved sequence predictions; independently reconstruct its degree counts and graph before reindexing into the extended sequence universe. Preserve original model files and prediction arrays byte-for-byte.

## Estimation and diagnostics

- Freeze every new baseline prediction before calculating species metrics. Prediction code must not read species label files. Report all methods separately in the prespecified positive direction. Do not invert below-chance scores or select a winner per species.
- AP and AUROC on every primary row; chance AP is prevalence (1/11), AUROC 0.5. Reuse the independently checked metric implementation. Human AP at prevalence 1/2 is not directly comparable to species AP.
- Recalculate **500 paired protein-endpoint bootstrap replicates**, seed 20261006, for all new and reused methods together on each species. Do not combine intervals from unrelated random resamples. Protein multiplicities multiply for distinct endpoints; self-pairs receive one multiplicity. Report descriptive 95% percentile intervals and prespecified paired differences: Bernett minus v5 for each baseline, and baselines minus the relevant frozen neural comparators. No training-seed, phylogenetic-family, dataset-construction or multiplicity-adjusted inference is claimed.
- Report TRAIN-degree variation, exact membership, bilateral homolog coverage, and interolog support **separately for positive and negative pairs**, with score tie counts. These diagnose label association; they do not change the predictors.
- Secondary, common-row sensitivities: (1) remove any pair containing an exact sequence from either TRAIN reference, and (2) retain one observation per unordered sequence pair after removing contradictory-label groups. Apply each identical mask to every predictor, retain the full-row primary results, and disclose changed sizes/prevalence. No new homology-pruned benchmark is introduced.
- Distinguish independent endpoint predictability from conserved interaction transfer. Good endpoint-only performance shows that partner compatibility is not necessary for that performance; it does not prove a particular neural model uses the same signal. Interolog transfer can be legitimate conserved biology. Weak performance rules out neither other shortcuts nor other cheap predictors.

## Artifacts and resources

Archive the verified original human report, figures, metrics, scripts and completion manifest in `archive/before-nonhuman/`; retain original inputs, models and raw predictions in place. Update the main report, combined metric/interval/difference tables, summary, completion manifest, and all comparison image formats. Add species-specific evidence and reproducible scripts within this folder. The original frozen scientific implementation remains unchanged and imported by the extension.

Use CPUs in the existing interactive Arrhenius allocation: at most two independent 16-thread searches concurrently; at most 16 threads during fitting. Preserve per-stage input/output hashes and resumable checkpoints. No GPU computation or new production job. Do not claim that dataset size or FADI alone establishes a shortcut, and do not change methods to obtain an expected Bernett/v5 ranking.
