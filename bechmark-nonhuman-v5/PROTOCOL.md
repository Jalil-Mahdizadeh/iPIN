# Frozen v5 iPIN transfer to five nonhuman species

Freeze the existing best human-trained iPIN v5 checkpoints: ESM2 update 27,374 and ESMC update 21,899, selected by the completed v5 DEV audit. This is cross-species inference without target-species fitting, recalibration, or checkpoint selection.

Use the exact five released PLM-interact/D-SCRIPT sequence-pair CSVs already audited in `plm-interact-reproducability`: mouse, fly, worm and yeast (55,000 rows each), and E. coli (22,000). Preserve all 242,000 rows and labels, including duplicates and conflicting sequence-pair labels. Inference deduplicates unordered exact sequence pairs; mapping restores source observations. A secondary analysis excludes conflicting-label pairs and counts each remaining pair once.

Released proteins are at most 800 residues. Preserve their supplied strings. This evaluates the historical length-limited collection, not all proteins in each species. Positive prevalence is 1/11. No test threshold optimization, class balancing, negative regeneration or outcome-dependent exclusions.

iPIN and native PLM-interact Bernett use full supplied sequences, evaluation mode, FP32 weights, BF16 autocast, TF32 disabled and mean raw AB/BA logits. Save both orientations. HumanV11 uses full FP32 computation: its BF16 qualification failed a predeclared tolerance before any test metrics were calculated. FP32 is checked against independently archived original-runtime logits and against padding changes. Previously reproduced original-order humanV11 predictions, if reused, are a separately labeled paper-convention reference rather than a substitute for symmetric scores.

Use the same competitor families as benchmark-v5: native PLM-interact, TUnA, both X-PAIR releases, RAPPPID, SPRINT and original D-SCRIPT. Record variants before examining new metrics; never select the best checkpoint separately for each test species. SPRINT retains the user-approved human v5 TRAIN-positive graph and may use nonhuman sequences for sequence-only preprocessing, never nonhuman interaction labels.

Report per-species AP (sklearn average precision, the paper's AUPR convention), AUROC, PR/ROC curves and paired differences versus declared native comparators. Use 1,000 paired protein-endpoint bootstrap replicates, seed 20261005, with common endpoint weights across models. Duplicates retain source multiplicity in the primary analysis. Intervals are descriptive and conditional on trained models and historical tests. Report unique-pair sensitivity, exact exposure, length and score coverage. Different species do not guarantee absence of identical proteins or homologs.

All models must cover identical rows. Qualify each competitor's native scoring and preprocessing. Analysis refuses to seal a completed comparison when a roster member is missing. Preserve original folders, weights and images; use atomic, checksummed inference shards for resumption.

Exposure-audit clarification: excluding all known sources, including X-PAIR default's nonhuman supervision, leaves only 53 fly positives and 21 yeast positives. Report this loss of information and changed prevalence explicitly. Also report the common subset excluding only known human supervised sources for assessing human-only transfers; X-PAIR default remains exposed on that latter subset. These masks are defined by source membership, never by model scores, and neither replaces the primary released tests.

The fixed roster has 11 predictors: both iPIN checkpoints, native humanV11 and native Bernett, TUnA human seed 47 and TUnA Bernett, both X-PAIR releases, RAPPPID mult, SPRINT, and D-SCRIPT human_v1. TUnA seed 47 was chosen before target test metrics, with no seed comparison.
