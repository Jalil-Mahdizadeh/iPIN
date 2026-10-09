# iPIN experiment walkthrough

Status: **10 October 2026**. Links lead to detailed reports.

**Names:** V1–V6 are retraining/data proposals. Vx, Vx-v2, Vy, Vz and Vx-v3 are MSA pilots. STRING V11/V12.5 are database versions, not additional iPIN model versions.

**Scores:** AP means average precision, not accuracy; higher is better. Compare scores only within the same evaluation set: historical TEST and pilot DEV scores are not directly comparable.

**Native baseline and retraining: V1–V6**

We first [reproduced native PLM-interact](plm-interact-reproducability/REPORT.md). The V1–V5 reference is **native Bernett AP 0.6903** on 52,048 original TEST pairs, averaging A–B/B–A logits. These are existing, historically reused TEST results. Checkpoints were DEV-selected; V1, V3 and V5 training stopped early.

| Version | What we tested | Short result |
|---|---|---|
| [V1](benchmark-v1/REPORT.md) | Full-length retraining, comparing ordinary training with training on the averaged A–B/B–A score. | Reference **0.6757**; symmetric **0.6782**. Both below native; the symmetry benefit was inconclusive. |
| [V2](benchmark-v2/REPORT.md) | Compare reference training, a 2,193-residue combined training-length cap, ×10 positive weighting, and classification alone without masking/auxiliary residue prediction. | AP respectively **0.6694 / 0.6852 / 0.6727 / 0.6906**. Classification alone was best, but did not establish improvement over native. |
| [V3](benchmark-v3/REPORT.md) | Screen learning rates; replace the single summary-token readout with pooled residue features and a small nonlinear classifier. | Internal gain missed the preset threshold. Final exploratory model: **0.6856**, no improvement over native. |
| [V4](benchmark-v4/REPORT.md) | Compare ESM2/ESMC backbones and change how attention handles positions across separate protein chains. | Both modified-attention arms learned poorly and were stopped. Standard-attention ESMC completed: **0.6814**, no improvement over native. |
| [V5 data](data-preparation-v5/REPORT.md) + [models](benchmark-v5/REPORT.md) | Build larger human data with protein/homology exclusions and negatives balanced for protein occurrence and functional similarity; train ESM2 and ESMC. | Original TEST: **0.6917 / 0.6871**, no established native improvement. Custom ILP-negative TEST: **0.6570 / 0.6452**, versus native **0.6377**; ESM2 improved here. Tests share positives, so are not independent replications. |
| [V6](improvment-proposal-v6.md) | Proposed human STRING V12.5 retraining, following humanV11. | **Not pursued; no model result.** Concern about repeating V11 shortcuts shifted research toward Bernett-based MSA pilots. |

**MSA pilots: what caused the Vx gain?**

An MSA aligns a protein with its evolutionary relatives (homologs). These pilots add a small predictor to frozen PLM-interact using a frozen MSA encoder. They share Bernett **4,000 TRAIN / 4,000 DEV** samples and the **958-pair DEV assessment**, where native AP is **0.6409**. Calibration is separate. Findings are exploratory; **no pilot evaluates TEST**.

Vx uses up to **127 homolog rows + 1 query = 128 rows**. Ineligible pairs retain native scores. Vy's different feature availability limits attribution to joint versus separate encoding.

| Version | What we tested | Short result |
|---|---|---|
| [Vx](pilot-vx/results/REPORT.md) | Encode A+B jointly with homolog MSAs; compress inter-chain residue-pair features into 128 global summaries. Compare true/shuffled pairing and quality/profile controls. | **True 0.6644; shuffled 0.6632; quality 0.6509.** True and shuffled improve over native, with **no established pairing-specific benefit**. |
| [Vx-v2](pilot-vx-v2/results/REPORT.md) | Replace some global summaries with local block pooling to preserve sparse interface signal; add a spatial-scrambling control and depth/coverage/family diagnostics. | **Local true 0.6557; local shuffled 0.6625.** No demonstrated improvement in pairing sensitivity or benefit from spatial locality, including the short-pair follow-up. |
| [Vy](pilot-vy/results/REPORT.md) | Encode A/B monomer MSAs **separately**, then combine their summaries with PLM-interact. Include query-only, profile and additive-protein controls. | Monomer-MSA fusion: **0.6409**; calibration gave the added branch zero weight. Query+profile: **0.6459**, with uncertain gain. No demonstrated learned monomer-MSA benefit. |
| [Vz / Q](pilot-vz/results/REPORT.md) | Keep Vx joint encoding/readout but supply **only the query pair: depth 1**. Reuse true/shuffled predictions. | **Q 0.6439**, below true/shuffled; its gain over native is uncertain. It did not recover Vx's gain. Homolog-content and depth effects remain entangled. |
| [Vx-v3 / C](pilot-vx-v3/results/REPORT.md) | Independently shuffle non-query residues within each alignment column. Preserve the query, depth and per-position residue/gap frequencies, while disrupting coherent homolog sequences. | **Completed and verified. C: 0.6501.** Recovery of Vx's gain was not established. C − shuffled: −0.0132, 95% interval [−0.0292, +0.0093]. The difference is inconclusive; any loss could also reflect unnatural synthetic inputs. |

“True” uses the original shared-accession homolog matching; it does not verify biological interaction between every homolog pair. “Shuffled” rearranges intact B homolog rows within taxonomic groups, preserving each chain's family information. “Quality/profile” uses alignment statistics without learned Pairformer features. Similar true/shuffled scores do **not** establish statistical equivalence.

**Supporting versioned benchmarks and audits**

FADI-v1 and TRIQ-v1 are completed local studies; their files await a separate commit.

| Study | Question and result |
|---|---|
| [V2/V5 cross-species comparison](benchmark-v5-nonhuman/v1-v4-comparison/REPORT.md) | V2 capped/clean models beat native Bernett AP on all five species; both V5 models were weaker. This comparison evaluated two V2 checkpoints, not every earlier model. |
| [V5 simple baselines](benchmark-v5-baselines/REPORT.md) | Protein-frequency, sequence-property and homologous-interaction baselines were near chance on human tests. Cross-species interaction lookup beat V5 neural models; Bernett baselines did not explain V2's stronger transfer. |
| [STRING V11 shortcut audit](literature/string-v11-train-degree-baseline/REPORT.md) | A simple training-set positive/negative protein-occurrence score reached validation **AP 0.8360**, without a language model. This establishes strong endpoint shortcuts, not the full explanation of neural performance. |
| FADI-v1 (local) | Audit separation of TRAIN and held-out protein families. On the same original test, separation index: Bernett **11.12/100**, V5 **6.68/100**. **V5 is not fully family-disjoint.** Values depend on annotation rules, not just biological distance. |
| TRIQ-v1 (local) | Audit source consistency, diversity and sampling shortcuts. Primary index: V5 **94.60**, Bernett **92.29**, V11 **26.87**. Changing shortcut-score interpretation reverses V5/Bernett ordering; this remains an exploratory index, not proof of biological dataset quality. |

**Discussed, not run**

- **Independent homolog sampling:** select A/B homologs independently, then encode them **jointly** at current Vx depth. Removes the shared-accession restriction. Unlike Vy, encoding remains joint.
- **R, repeated query:** repeat the query pair to Vx depth to test depth alone. **On hold.**
- **Deeper MSAs (256/512):** discussed; no experiment or result yet.
