# Nonhuman benchmark: interim report

As of 2026-10-05T13:22:47.461188+00:00. The benchmark continues; this is not the final result. Seven of eleven predictors have complete, verified coverage of all 242,000 released rows. Check STATUS.json for live completion.

The frozen v5 iPIN models currently trail native PLM-interact Bernett and both TUnA variants on AP in all five species. Removing the union of known human supervised exposures preserves that conclusion. No nonhuman fitting, checkpoint reselection or score inversion has been performed.

## Average precision

| Model | Mouse | Fly | Worm | Yeast | E. coli |
|---|---:|---:|---:|---:|---:|
| iPIN v5 ESM2 | 0.2091 | 0.1864 | 0.1359 | 0.1366 | 0.0807 |
| iPIN v5 ESMC | 0.1768 | 0.1531 | 0.1028 | 0.1165 | 0.0663 |
| Native PLM-interact (Bernett) | 0.2751 | 0.3017 | 0.1856 | 0.1974 | 0.1572 |
| TUnA (human, seed 47) | 0.8832 | 0.8550 | 0.8368 | 0.6441 | 0.6730 |
| TUnA (Bernett) | 0.3528 | 0.3740 | 0.3551 | 0.3527 | 0.2139 |
| RAPPPID (released mult) | 0.1675 | 0.1338 | 0.1649 | 0.1305 | 0.1033 |
| D-SCRIPT (human_v1) | 0.5799 | 0.5524 | 0.5482 | 0.4048 | 0.5350 |



## AUROC

| Model | Mouse | Fly | Worm | Yeast | E. coli |
|---|---:|---:|---:|---:|---:|
| iPIN v5 ESM2 | 0.6254 | 0.6114 | 0.5553 | 0.5680 | 0.3757 |
| iPIN v5 ESMC | 0.6331 | 0.5726 | 0.5239 | 0.5180 | 0.2749 |
| Native PLM-interact (Bernett) | 0.6983 | 0.6940 | 0.6176 | 0.6196 | 0.4839 |
| TUnA (human, seed 47) | 0.9774 | 0.9710 | 0.9602 | 0.8992 | 0.8775 |
| TUnA (Bernett) | 0.7743 | 0.7855 | 0.7789 | 0.7656 | 0.7412 |
| RAPPPID (released mult) | 0.6918 | 0.6823 | 0.7455 | 0.6678 | 0.5935 |
| D-SCRIPT (human_v1) | 0.8329 | 0.8236 | 0.8130 | 0.7890 | 0.8603 |



AP random-ranking reference is approximately 0.0909; AUROC chance is 0.5. Mouse, fly, worm and yeast each have 55,000 rows (5,000 positive); E. coli has 22,000 (2,000 positive). These are full source-row estimates; uncertainty intervals and duplicate sensitivities are pending final analysis.

## Still running

- Native humanV11: job 3383464, full FP32 after passing the original-runtime and padding checks. An earlier BF16 qualification failed before scoring; the failed attempt is preserved in logs.
- X-PAIR Bernett and default: job 3382643; all 56,634 sequence features have completed and pair scoring preparation is underway.
- SPRINT: CPU similarity job 3383717 (256 cores), followed automatically by scoring job 3383829. Its serial sparse scorer passed byte-for-byte comparisons against native outputs, including self-interaction checks.
- Automatic finalization is alive and waits for all 11 predictors before producing REPORT.md and results/COMPLETE.json.

## Interpretation limits

Human-trained native/TUnA cross-species models and Bernett/v5 models use different human interaction evidence and negative distributions. This is a frozen-transfer comparison, not an isolated architecture experiment. Current results do not support claiming broad cross-species improvement for v5 iPIN.

Exact overlaps include 36 mouse positive pairs with v5 TRAIN/DEV and 81 with the native human sources. X-PAIR default has substantial nonhuman supervision. The all-known-exposure filter retains just 53 fly positives and 21 yeast positives, so those sensitivity estimates require caution; a separate common human-source-unexposed analysis preserves almost all fly and yeast positives. Homology and unavailable released-checkpoint membership remain limitations.

## Clarification: the Figure 2 native reference

The native row above is the Bernett-trained release, not the humanV11 checkpoint used in Figure 2. These are different frozen weights. The earlier independent humanV11 reproduction already matches Figure 2; the current symmetric humanV11 run remains in progress. Recomputed archived original-order results are shown separately below.

| Species | Figure 2 deposited AP | Earlier humanV11 AP | Earlier humanV11 AUROC |
|---|---:|---:|---:|
| mouse | 0.903963 | 0.903963 | 0.982147 |
| fly | 0.913299 | 0.913302 | 0.982513 |
| worm | 0.887592 | 0.887592 | 0.973783 |
| yeast | 0.705514 | 0.705514 | 0.912316 |
| ecoli | 0.721570 | 0.721570 | 0.904265 |

The [Figure 2 discrepancy audit](FIGURE2-AUDIT.md) verifies all 242,000 source rows and the iPIN inference paths against independent backbone forwards. It finds no production inference error explaining the poor iPIN transfer. It also documents why the missing humanV11 reference and training-distribution differences must be separated from those observed iPIN results. This does not identify the causal reason for the transfer failure; the benchmark remains in progress.
