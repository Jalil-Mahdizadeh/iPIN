**PLM-interact reproducibility check — frozen released checkpoints, supplied SIF**

The released checkpoints reproduce the main cross-species, Bernett, and mutation benchmark metrics when the evaluation inputs and actual scoring convention are respected. Two implementation details matter: mutation prediction uses a different formula from the paper's equation, and Bernett's deposited predictions are strongly consistent with an undocumented 3,570-token cap. This is a checkpoint-inference reproduction, with **no retraining**. It is not a claim that every result in the paper has been independently reproduced.

The evaluation covers all **294,889 released test rows**: 242,000 cross-species pairs, 52,048 Bernett pairs, and 841 mutation cases. The humanV11, Bernett, and mutation checkpoints were loaded strictly and kept frozen. The same supplied SIF ran every fresh prediction. All expected rows and labels passed the final coverage checks, and output/checkpoint hashes are retained. The paper and supplements are [available locally](../literature/plm-interact/READING_NOTES.md); the detailed procedure is in [PROTOCOL.md](PROTOCOL.md).

**Figure 2: all five cross-species benchmarks agree.** AUPR here means average precision (AP), as in the authors' code. Trapezoidal PR-AUC is retained separately in the metric files.

| Test | Rows | Paper AUPR | Deposited AP | Fresh FP32 AP | Fresh AP, source precision |
| --- | --- | --- | --- | --- | --- |
| Mouse | 55,000 | 0.904 | 0.903963 | 0.903937 | 0.903963 |
| Fly | 55,000 | 0.913 | 0.913299 | 0.913295 | 0.913302 |
| Worm | 55,000 | 0.888 | 0.887592 | 0.887575 | 0.887592 |
| Yeast | 55,000 | 0.706 | 0.705514 | 0.705472 | 0.705514 |
| Ecoli | 22,000 | 0.722 | 0.721570 | 0.721675 | 0.721570 |

The workbook's probabilities exactly equal a float64 sigmoid applied to the deposited logits, within about 1e-15. Applying that same numeric representation to fresh logits gives a maximum AP disagreement of 2.57e-06 across species. Direct FP32 sigmoid outputs create more tied scores. The small yeast difference crosses the three-decimal rounding boundary: direct FP32 AP rounds to 0.705, while the source-consistent representation rounds to 0.706. Both results are retained; no checkpoint, threshold, or model parameter was selected to improve agreement. Fresh and deposited original-order cross-species predictions make identical binary decisions at 0.5.

The baseline curve files use the same sequence/label multisets, despite accession aliases and different row orders. PLM-interact remains ahead of their deposited APs on all five species. However, the paper's baseline bars and its deposited curves are not always numerically the same experiment: the workbook explicitly says the bars were taken from prior publications or correspondence, whereas the curves came from checkpoint reruns. For example, E. coli TUnA is 0.677 in the bar but 0.688506 from the deposited curve. Thus, some quoted percentage improvements depend on which baseline numbers are used. See [baseline-bars-versus-curves.csv](results/baseline-bars-versus-curves.csv). Baseline models were not rerun in this task.

**Figure 4: Bernett, all 52,048 test pairs.** The threshold is fixed at 0.5. Full sequences were the prespecified primary evaluation; the two explicit truncation settings are sensitivity checks.

| Evaluation | AP | AUROC | Precision | Recall | F1 |
| --- | --- | --- | --- | --- | --- |
| Paper | 0.69 | 0.70 | 0.63 | 0.71 | 0.66 |
| Deposited scores | 0.686620 | 0.697756 | 0.626530 | 0.706194 | 0.663981 |
| TUnA deposited scores | 0.691908 | 0.703356 | 0.648214 | 0.645673 | 0.646941 |
| Fresh, full sequences | 0.686576 | 0.697632 | 0.626226 | 0.706502 | 0.663946 |
| Fresh, cap 1,603 tokens | 0.687971 | 0.699505 | 0.629156 | 0.700161 | 0.662763 |
| Fresh, cap 2,196 tokens | 0.687403 | 0.699120 | 0.628505 | 0.704580 | 0.664372 |
| Fresh, inferred 3,570-token cap (post hoc) | 0.686620 | 0.697756 | 0.626530 | 0.706194 | 0.663981 |

| Sequence handling | Mean absolute score difference | Maximum difference | Classifications differing at 0.5 |
| --- | --- | --- | --- |
| bernett_full | 0.000444 | 0.375 | 39 |
| bernett_1603 | 0.0121 | 0.727 | 914 |
| bernett_2196 | 0.0045 | 0.598 | 365 |

Among the prespecified evaluations, the closest score-level agreement is `bernett_full` (mean absolute difference 0.000444). There are 8,200 pairs longer than 1,603 tokens and 3,392 longer than 2,196 tokens; the maximum is 7,426. Full-sequence inference reproduces every Figure 4 metric at the displayed two-decimal precision, but differs from the deposited classification for 39 pairs. The paper's AP/AUROC tie with TUnA is a rounded tie: the unrounded deposited TUnA ranking metrics are slightly higher, while PLM-interact has higher recall and F1. No rows were dropped for length.

A separately labeled **post-hoc diagnostic** localized almost all Bernett score errors to the longest 1% of inputs. Caps 3,569–3,573 were probed against individual deposited scores, without selecting on labels or AP. Only **3,570 tokens** matched the diagnostic cases closely. Fresh inference with that cap was then checked on **all 520 affected rows**, retaining the full-sequence results elsewhere. It reduced the full-test mean absolute score error to **2.92e-07**, maximum error to **1e-05**, and classification disagreements to **0**. This strongly supports a 3,570-token inference cap as the missing preprocessing detail; that number was not found in the paper or released example commands. The original primary results remain unchanged. All probe results, the criterion, and the additional predictions are retained in [bernett-cap-diagnostic.json](results/bernett-cap-diagnostic.json) and its accompanying CSVs.

**Figure 5: mutation results reproduce with the released code's scoring formula.** The 598 published samples were matched to the released 841-case test set using mutation/interaction metadata. They are exactly the cases with each individual protein at most 2,000 residues. Nine workbook mutation ranges had been converted into Excel dates; the documented metadata/date checks resolve those joins without editing the inference sequences.

| Checkpoint | 598-case scoring | AP | AUROC |
| --- | --- | --- | --- |
| Mutation checkpoint | Deposited scores | 0.612145 | 0.794095 |
| Mutation checkpoint | Fresh, released-code formula | 0.612132 | 0.794078 |
| Mutation checkpoint | Fresh, paper equation | 0.597410 | 0.771896 |
| Zero-shot humanV11 | Deposited scores | 0.233400 | 0.525537 |
| Zero-shot humanV11 | Fresh, released-code formula | 0.233452 | 0.525563 |
| Zero-shot humanV11 | Fresh, paper equation | 0.221384 | 0.471545 |

The implementation scores `sigmoid(mutant_logit - wild_logit)`. The paper instead writes `log(p_mutant / p_wild)`, where `p = sigmoid(logit)`. These formulas share the classification direction but can rank mutations differently. The paper-equation rows above use a stable float64 calculation from the same fresh logits. Consequently, the numerical Figure 5 result is reproducible from the released code, but the written equation does not fully specify the implementation that generated it. The all-layer mutation checkpoint is the authors' already fine-tuned release; this run performed no fine-tuning.

On all 841 released cases, the mutation checkpoint obtains AP **0.563046**, AUROC **0.771844**; zero-shot humanV11 obtains AP **0.230736**, AUROC **0.535905**. The 598-case comparison is therefore not representative of the full test result. The classifier-only fine-tuned result was checked from deposited scores (AP 0.244588, AUROC 0.584039), but the authors' public model catalog does not provide that separate checkpoint, so its inference was not independently reproduced.

Sequence handling also materially affects mutation results:

| Checkpoint | Sequence handling | 598-case AP | 598-case AUROC |
| --- | --- | --- | --- |
| Mutation checkpoint | Full sequences | 0.612132 | 0.794078 |
| Mutation checkpoint | 1,603-token cap | 0.561178 | 0.774608 |
| Mutation checkpoint | 2,196-token cap | 0.590284 | 0.777578 |
| Zero-shot humanV11 | Full sequences | 0.233452 | 0.525563 |
| Zero-shot humanV11 | 1,603-token cap | 0.237452 | 0.545460 |
| Zero-shot humanV11 | 2,196-token cap | 0.223455 | 0.510894 |

At the public CLI default of 1,603 tokens, 169 of the 841 cases (72 of the published 598) have identical tokenized wild and mutant inputs because truncation removes the mutation. At 2,196 tokens the counts are 93 and 36. Full inputs retain every mutation. The released mutation split has no exact sequence-triplet or wild-type interaction-pair overlap between training and test, although 271 test cases have both individual proteins seen in training and 643 have at least one seen. These overlap checks only read the split files; they did not fit or select a model.

**Supplementary results require narrower interpretations.** All deposited masking-ablation APs and sequence-identity-bin APs were independently recomputed. Masking does not improve every reported AP: mouse AP is 0.907840 without masking and 0.903963 at 15% masking. PLM-interact is also below TUnA in four deposited identity bins, including mouse identity (40,60] (0.622185 vs 0.714071). The fly/worm display labels are swapped relative to their hostname fields in the masking table. The reported McNemar p-values agree with exact binomial calculations from the reported discordant counts, but those counts do not reproduce when the deposited scores are classified at a common probability threshold of 0.5. The classification threshold/procedure for that test is insufficiently specified for this audit to verify the counts. These findings are preserved in the supplementary CSVs, without retraining any ablation.

Near-equal AP after reversing proteins does not mean individual predictions are symmetric. The deposited full-test scores, and fresh inference on a fixed every-53rd-row subset, show classification changes:

| Species | Published full-test order flips | Fresh sampled order flips |
| --- | --- | --- |
| Mouse | 612 / 55,000 | 9 / 1038 |
| Fly | 590 / 55,000 | 16 / 1038 |
| Worm | 639 / 55,000 | 20 / 1038 |
| Yeast | 1,012 / 55,000 | 16 / 1038 |
| Ecoli | 2,513 / 22,000 | 54 / 416 |

The E. coli deposited predictions flip for 11.4% of pairs even though the two aggregate APs are close. The fresh reversed subset is compared with both fresh original-order scores and the matching deposited reversed scores in [score-concordance.csv](results/score-concordance.csv). All primary runs preserve the released pair order. Also, E. coli contains 3,770 duplicate unordered sequence/label rows among 22,000 rows; original multiplicities were retained to reproduce the paper, and should not be mistaken for independent observations.

Fresh inference and all prespecified sensitivity runs used approximately **3.48 GPU-hours of measured worker evaluation time**, excluding loading, queueing, the qualification pilot, and the short post-hoc Bernett diagnostic. SLURM job 3110591 used eight GH200 GPUs on two nodes; mutation inference used the existing one-GPU interactive allocation 3085063. Each worker records its GPU UUID. Qualification produced bitwise-identical logits against the upstream model class, and singleton/padding comparisons passed. The publication archive and current inference model computations agree; the relevant source change only strengthens checkpoint loading from non-strict to strict.

The main limits are explicit: there was no retraining; no fresh baseline-model inference; no independent classifier-only mutation checkpoint evaluation; no rerun of structure-generation examples or MMseqs alignments; and no fresh virus-host inference. Figure 7's workbook provides a summary table and training sequences, rather than held-out prediction scores, so its reported AP/F1/MCC are recorded but not independently verified here. The [published paper](https://www.nature.com/articles/s41467-025-64512-w), [publication code archive](https://doi.org/10.5281/zenodo.16643324), and pinned release provenance identify the claims and artifacts under review.

The numerical evidence is in [fresh-metrics.csv](results/fresh-metrics.csv), [published-score-metrics.csv](results/published-score-metrics.csv), [score-concordance.csv](results/score-concordance.csv), and [coverage.json](results/coverage.json). The raw logits, probabilities, labels, and row IDs are saved in `results/predictions/` and the merged result CSVs. The [main PR curves](results/plots/pr-curves-main.pdf) and [mutation scoring comparison](results/plots/mutation-score-definitions.pdf) provide standalone figures; PNG versions are alongside them. Run settings, hashes, commands, and exact scope are documented in [PROTOCOL.md](PROTOCOL.md) and [README.md](README.md).
