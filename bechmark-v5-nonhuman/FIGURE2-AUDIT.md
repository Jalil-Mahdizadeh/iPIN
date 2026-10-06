# Figure 2 discrepancy audit

Audit date: 2026-10-05. Production benchmarking continues. This document separates the missing Figure 2 reference in the interim comparison from the poor observed transfer of the frozen v5 models.

The interim native row used the **Bernett-trained release**. Figure 2 uses **humanV11**, a separately trained checkpoint. The [authors' model catalog](https://plm-interact.readthedocs.io/en/latest/data.html) explicitly distinguishes them. Both were already in the frozen benchmark roster, but the fresh humanV11 run had not completed when the interim table was shown. The earlier independent humanV11 reproduction should have been included as a clearly labeled reference.

## Correct native reference

These are the archived independent FP32 predictions from `plm-interact-reproducability`, recomputed with float64 sigmoid to match the deposited score representation. They preserve the original pair order. They are not the unfinished fresh symmetric AB/BA results.

| Species | Figure 2 deposited AP | Independently reproduced humanV11 AP | Reproduced AUROC |
|---|---:|---:|---:|
| Mouse | 0.903963 | 0.903963 | 0.982147 |
| Fly | 0.913299 | 0.913302 | 0.982513 |
| Worm | 0.887592 | 0.887592 | 0.973783 |
| Yeast | 0.705514 | 0.705514 | 0.912316 |
| E. coli | 0.721570 | 0.721570 | 0.904265 |

The maximum AP difference from the deposited curves is approximately 0.00000257. Native Figure 2 performance therefore reproduces. The fresh humanV11 job is also consistent with that reference: at the saved audit snapshot, **129,477 completed source rows** agreed with the archived original-order predictions to a maximum probability difference below **0.000047**. This is a prediction-concordance check, not a metric calculated on an incomplete test subset. Fresh D-SCRIPT APs also closely match its Figure 2 deposited curves (differences approximately 0.0001).

Evidence: [reference metrics](results/figure2-reference-audit.json), [fresh humanV11 concordance](results/humanV11-fresh-progress-concordance.json), [earlier full reproduction](../plm-interact-reproducability/REPORT.md).

## D-SCRIPT: Figure 2 displays AUPR, not AUROC

The published Figure 2 x-axis is precision–recall AUC (AUPR); it cannot be compared directly with the AUROC table. Inspecting the actual figure and recomputing both metrics from the deposited D-SCRIPT scores gives:

| Species | Figure 2 bar AUPR | Deposited-score AP | Fresh AP | Deposited-score AUROC | Fresh AUROC |
|---|---:|---:|---:|---:|---:|
| Mouse | 0.580 | 0.579788 | 0.579909 | 0.832899 | 0.832902 |
| Fly | 0.552 | 0.552286 | 0.552365 | 0.823619 | 0.823633 |
| Worm | 0.548 | 0.548092 | 0.548227 | 0.813055 | 0.813034 |
| Yeast | 0.405 | 0.404675 | 0.404809 | 0.788972 | 0.788961 |
| E. coli | 0.571 | 0.534892 | 0.534969 | 0.860338 | 0.860335 |

Fresh AP matches the bars at displayed precision for four species. E. coli is an actual bar-versus-deposited-prediction discrepancy: the bar says 0.571, whereas both the deposited predictions and our fresh run yield approximately 0.535. The paper's Methods/Baselines and workbook explicitly distinguish the AUPR values taken from prior publications from the checkpoint reruns used for PR curves. This explains the provenance difference, but does not determine why those earlier experiments differ. Across all five species, the maximum AUROC discrepancy between the deposited scores and our fresh scores is just 0.00002078.

Evidence: [recomputed metric comparison](results/dscript-figure2-metric-audit.csv), [input hashes and audit](provenance/dscript-figure2-metric-audit.json), [actual Figure 2](../literature/plm-interact/figures/article-03.png), and the [paper's Methods](https://www.nature.com/articles/s41467-025-64512-w).

## Checks on the poor iPIN results

- Independently matched all **242,000** source sequence pairs, labels, row indices and union mappings to the released CSVs. The union's placeholder label column is not used as ground truth.
- The frozen selected checkpoint identities are verified, state dictionaries load strictly, models use evaluation mode, and selected DEV predictions are reproduced by the inference path.
- Evaluated AB, BA and their fixed pooled score separately. Poor performance persists in each orientation. Averaging does not explain the collapse.
- Independently tokenized 20 fixed pairs per model, spanning all five species and including the longest pair per species, and tested both orientations. No scores were used to select these fixtures. The optimized FP32 forward agrees with the original ESM2 or official ESMC backbone forward using the same trained classifier:

| Model | Orientations checked | Maximum logit difference | Maximum probability difference |
|---|---:|---:|---:|
| iPIN ESM2 | 40 | 0.000006915 | 0.000001112 |
| iPIN ESMC | 40 | 0.000002742 | 0.000000527 |

Saved production BF16 versus independent FP32 probability differences on these fixtures were at most 0.01157 (ESM2) and 0.005824 (ESMC). This is a bounded numerical check, not a full-test FP32 equivalence claim. It does not reveal an error large enough on the inspected cases to explain the broad gap.

The first ESMC diagnostic failed because it assumed the tokenizer's two-string API inserts CLS/EOS. That API concatenates bare residues. Production training and inference both explicitly construct **CLS A EOS B EOS**, as intended. The corrected independent audit uses official per-sequence tokenization followed by that specified pair format and passes. The failed diagnostic log and its script are retained; no production tokenizer, weights, predictions or worker code were changed.

Evidence: [source and orientation audit](results/source-and-orientation-audit.json), [ESM2 independent forward](qualification/ipin-esm2-independent-nonhuman.json), [ESMC independent forward](qualification/ipin-esmc-independent-nonhuman.json), and `logs/audit-native-forward-*.log`.

## Interpretation

These checks have not identified a production inference error explaining the poor iPIN scores. They establish neither that every possible implementation issue is excluded nor the causal explanation for the failure. The current evidence supports a substantial cross-species transfer limitation of these frozen checkpoints.

The training tasks differ materially. HumanV11 uses the human counterpart of this cross-species benchmark: 421,792 training pairs, including 38,344 positives, with 1:10 positive-to-negative sampling. V5 uses a protected HIPPIE/ILP dataset with 700,764 balanced training pairs, including 350,382 positives, and selects checkpoints on ILP DEV. Interaction evidence, negative construction, protein coverage, homology and model selection distribution all differ. Even the native Bernett checkpoint performs much worse than humanV11 on these same nonhuman tests; both TUnA releases show a similar separation. This is consistent with a substantial training-distribution effect, but does not isolate its cause or prove that bias-aware sampling itself is harmful.

The lower nonhuman positive prevalence can lower AP, but does not explain the within-test model gap or below-chance E. coli AUROC. Changing a threshold or applying a strictly increasing prior correction cannot repair rankings. No score inversion, checkpoint reselection, nonhuman fitting or relabeling is justified by this audit.

The earlier human result remains narrowly valid: v5 ESM2 improved AP on the custom ILP-negative human test, but did not establish a meaningful improvement on the original Bernett test. It does not support a claim of broad cross-species superiority. Complete nonhuman comparisons, uncertainty intervals and planned exposure/duplicate sensitivities remain pending the remaining inference jobs.
