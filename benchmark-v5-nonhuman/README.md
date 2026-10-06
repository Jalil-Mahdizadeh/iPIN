# Frozen v5 iPIN: nonhuman transfer benchmark

This benchmark is stored in `benchmark-v5-nonhuman`. The models are the existing human-trained v5 checkpoints; this folder does not train or select a model on nonhuman data.

This folder was renamed from `bechmark-nonhuman-v5`. Historical provenance retains the original paths and hashes; a local compatibility symlink preserves access to those paths on the HPC and is excluded from Git. The pre-extension completion manifest and original sealed bytes, including rename-related documentation and launcher edits, are preserved in `archive/before-xpair-v11/`. Current completion verification covers the added X-PAIR humanV11 release.

Read [PROTOCOL.md](PROTOCOL.md) for the original frozen evaluation design, [the X-PAIR V11 addendum](XPAIR-V11-ADDENDUM.md) for the requested extension, [the current roster](provenance/current-roster.json) for the 12 primary predictors, and [provenance/prepared.json](provenance/prepared.json) for the five source datasets and checksums. The final comparison is [REPORT.md](REPORT.md). The original eleven-model roster is retained in [provenance/roster.json](provenance/roster.json); the extension's completion is recorded in `xpair-v11-status.json` and `results/COMPLETE.json`.

- Input pairs: 55,000 each for mouse, fly, worm and yeast; 22,000 for E. coli. Positive prevalence is 1/11.
- Frozen iPIN checkpoints: ESM2 update 27,374; ESMC update 21,899. Both were selected on the earlier v5 ILP DEV set.
- Reference variants: native PLM-interact humanV11 and Bernett; TUnA human seed 47 and Bernett. X-PAIR Bernett, default/X-fair and humanV11, RAPPPID mult, D-SCRIPT human_v1 and SPRINT are also included.
- SPRINT uses only the previously authorized v5 human TRAIN-positive graph (350,382 edges), with the complete TRAIN plus nonhuman sequence corpus for sequence-only preprocessing.
- Primary results retain every source row. Secondary results count unique, nonconflicting sequence pairs and use identical rows after removing any endpoint exposed in known public supervised training/validation sources.

Run the SLURM scripts from the project directory. `provenance/jobs.json` records submitted jobs. Pair inference and feature extraction commit checksummed chunks; a rerun resumes verified chunks and rejects a changed contract. SPRINT's native HSP stage resumes only at stage boundaries. Its predictor restarts with a fresh output because upstream output is append-only.

The native humanV11 reduced-precision pilot failed its numeric tolerance before metrics were inspected. That predictor therefore uses FP32 with TF32 disabled, checked against the earlier independently reproduced original-runtime logits. The two iPIN models and native Bernett use the qualified BF16 inference convention from benchmark-v5. TUnA, X-PAIR, RAPPPID and D-SCRIPT use their qualified released scoring behavior. No probability recalibration or threshold selection uses target labels.

After every model completes, run from the project directory:

```bash
bash benchmark-v5-nonhuman/scripts/container.sh analysis python scripts/collect.py
bash benchmark-v5-nonhuman/scripts/container.sh analysis python scripts/analyze.py
bash benchmark-v5-nonhuman/scripts/container.sh analysis python scripts/plot_with_historical.py
bash benchmark-v5-nonhuman/scripts/container.sh analysis python scripts/report_nonhuman.py
bash benchmark-v5-nonhuman/scripts/container.sh analysis python scripts/verify_complete.py
```

Inference uses the existing ARM64 SIF images. CPU-only SPRINT uses the same upstream source compiled on Arrhenius x86 CPU nodes, with Boost headers extracted from its pinned SIF. Numerical qualification compares its native and sparse serial scorers against native SIF outputs. Sparse scoring omits unrequested cells while preserving each requested cell's arithmetic and accumulation order; it leaves native HSP/high-count preprocessing intact.

SPRINT copying recovery (2026-10-05): job 3383829 was stopped before prediction because Python's 4 MiB input buffer refilled on each small random HSP-block read. Its completed 6.34 GB raw similarity file and 2.51 GB canonical prefix were preserved. Job 3402258 resumes validation and scoring with `slurm/sprint-score-mmap.sbatch`, using qualified memory-mapped copying and the unchanged scorer. The new canonical file must match the preserved prefix exactly before scoring. See [recovery provenance](provenance/sprint-copy-recovery.json) and [byte-equivalence checks](qualification/sprint-copy/qualification.json). This does not repeat HSP generation or change the training graph, sequences, HSP records, their ordering, or SPRINT parameters.

All four main figure groups include 14 models: the original eleven, X-PAIR humanV11 and the two completed historical v2 controls (length-capped update 8,000 and clean BCE update 4,000). All thirteen prior models' predictions and bootstrap samples were reused. X-PAIR humanV11 reuses all 56,634 existing Ankh features and the same frozen release used in the human benchmark; only its interaction head and new bootstrap samples require computation. [Combined figure metrics](results/combined-figure-metrics.csv), [intervals](results/combined-figure-confidence-intervals.csv), and [verification](provenance/historical-figure-extension.json) accompany the figures. The prior thirteen-model results are preserved in `archive/before-xpair-v11/`. Rebuild the combined figures with `bash benchmark-v5-nonhuman/scripts/container.sh analysis python scripts/plot_with_historical.py` from the project root.

To resume only the added X-PAIR V11 inference, run `bash benchmark-v5-nonhuman/scripts/run_xpair_v11.sh` on one allocated GPU. To update all comparisons afterward, run `bash benchmark-v5-nonhuman/scripts/finish_xpair_v11.sh`. Both exact and Ankh-normalized source audits confirm that the existing common exposure masks cover this release, so the earlier subset rows are retained.

Exact matching is not a homology audit. The nonhuman tests were not protected during v5 human data construction. Known supervised membership of the precise RAPPPID and X-PAIR Bernett releases is not fully established. X-PAIR default includes substantial nonhuman supervised exposure and is not a clean human-only transfer baseline.
