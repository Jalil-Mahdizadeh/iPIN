# iPIN experiment walkthrough

Status: **11 October 2026**. Links lead to detailed reports.

**Names:** V1–V6 are retraining/data proposals. Vx, Vx-v2, Vy, Vz, Vx-v3, Vx-v4 and Vx-v5 are MSA pilots. V7 tests pretrained contact-head features. STRING V11/V12.5 are database versions, not additional iPIN model versions.

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
| [Vx](pilot-vx-series/pilot-vx/results/REPORT.md) | Encode A+B jointly with homolog MSAs; compress inter-chain residue-pair features into 128 global summaries. Compare true/shuffled pairing and quality/profile controls. | **True 0.6644; shuffled 0.6632; quality 0.6509.** True and shuffled improve over native, with **no established pairing-specific benefit**. |
| [Vx-v2](pilot-vx-series/pilot-vx-v2/results/REPORT.md) | Replace some global summaries with local block pooling to preserve sparse interface signal; add a spatial-scrambling control and depth/coverage/family diagnostics. | **Local true 0.6557; local shuffled 0.6625.** No demonstrated improvement in pairing sensitivity or benefit from spatial locality, including the short-pair follow-up. |
| [Vy](pilot-vy/results/REPORT.md) | Encode A/B monomer MSAs **separately**, then combine their summaries with PLM-interact. Include query-only, profile and additive-protein controls. | Monomer-MSA fusion: **0.6409**; calibration gave the added branch zero weight. Query+profile: **0.6459**, with uncertain gain. No demonstrated learned monomer-MSA benefit. |
| [Vz / Q](pilot-vz/results/REPORT.md) | Keep Vx joint encoding/readout but supply **only the query pair: depth 1**. Reuse true/shuffled predictions. | **Q 0.6439**, below true/shuffled; its gain over native is uncertain. It did not recover Vx's gain. Homolog-content and depth effects remain entangled. |
| [Vx-v3 / C](pilot-vx-series/pilot-vx-v3/results/REPORT.md) | Independently shuffle non-query residues within each alignment column. Preserve the query, depth and per-position residue/gap frequencies, while disrupting coherent homolog sequences. | **Completed and verified. C: 0.6501.** Recovery of Vx's gain was not established. C − shuffled: −0.0132, 95% interval [−0.0292, +0.0093]. The difference is inconclusive; any loss could also reflect unnatural synthetic inputs. |
| [Vx-v4 / I](pilot-vx-series/pilot-vx-v4/results/REPORT.md) | Sample intact A/B homologs independently, then encode them jointly. Remove the shared-accession requirement while keeping Vx's original cohort, each pair's depth, masks, gate and readout. | **Completed and verified. I: 0.6493.** No established gain over native or recovery of Vx. I − shuffled: −0.0139, 95% interval [−0.0293, +0.0042]; I − true: −0.0150 [−0.0299, −0.0010]. Sampling also changes homolog composition/diversity, so this does not prove accession matching is necessary. |
| [Vx-v5 / depth 256](pilot-vx-series/pilot-vx-v5/results/REPORT.md) | Increase the matched MSA limit from 128 to 256 total rows, preserving the original true rows as a prefix. Compare new matched, shuffled and quality/profile arms with their frozen depth-128 references. Same cohort, gates, encoder and readout; naturally shallow inputs keep their available depth. | **Completed and verified. True256 0.6591; shuffled256 0.6598; quality256 0.6510.** True256 − true128: −0.0053, 95% interval [−0.0182, +0.0077]. No demonstrated depth benefit; the interval excludes the preset +0.010 AP target, but establishes neither a loss nor equivalence. Of 4,453 eligible pairs, 4,072 gained rows and 3,566 reached depth 256. No full benchmark or TEST evaluation. |

The Vx pilot folders are collected in [pilot-vx-series](pilot-vx-series/README.md). The [full Vx-128 TRAIN/VAL continuation](retrain-vx/results/REPORT.md) **completed and passed verification** on 11 October: job **3642615**, two complete four-GPU nodes, **13 h 33 min**. It processed all 163,192 original TRAIN and 59,260 VAL pairs; the small true and quality/profile heads used all 74,576 eligible TRAIN pairs, with both encoders frozen. Unsupported VAL pairs kept exact native scores. **No shuffled arm or TEST evaluation.**

On the primary **13,055 previously unevaluated assessment pairs**, AP was **native 0.6480 / true128 0.6550 / quality 0.6519**. True − native: **+0.0070**, 95% protein interval **[+0.0024, +0.0122]**; true − quality: **+0.0031 [−0.0013, +0.0077]**. The modest native gain missed the preset +0.010 AP target, and improvement beyond quality/profile was uncertain: **no-go/inconclusive under the frozen criteria**, despite successful execution. Full VAL AP was native **0.6422**, true **0.6505**, quality **0.6449**; this is descriptive because it includes calibration. The primary rows can share proteins/families with the pilots, so the result remains exploratory.

The [full Vx-128 TEST benchmark](benchmark-vx/results/REPORT.md) **completed and passed independent verification** on 11 October: job **3667033**, one full four-GPU node, **5 h 47 min**, all **52,048 original TEST pairs**. AP was **native 0.69032 / true128 0.69862 / quality 0.69260**; native/true AUROC was **0.69947 / 0.70379**. True − native: **+0.00830**, 95% protein interval **[+0.00519, +0.01138]**; true − quality: **+0.00602 [+0.00201, +0.00994]**. This supports a modest gain beyond the quality/profile control, but the observed gain missed the preset +0.010 AP target: **no-go/inconclusive under the frozen practical criteria**. MSA evidence applied to **32,340 pairs (62.1%)**; the rest retained exact native scores. TRAIN-fitted heads and VAL-selected weights remained fixed; **no TEST fitting, calibration or shuffled arm**. The benchmark was explicitly authorized after the TRAIN/VAL no-go. This historically evaluated TEST set is not newly blind, and the intervals omit research-selection uncertainty.

“True” uses the original shared-accession homolog matching; it does not verify biological interaction between every homolog pair. “Shuffled” rearranges intact B homolog rows within taxonomic groups, preserving each chain's family information. “Quality/profile” uses alignment statistics without learned Pairformer features. Similar true/shuffled scores do **not** establish statistical equivalence.

**V7: pretrained contact-head pilot**

[V7 v1](pilot-v7-v1/README.md) implements the [V7 proposal](improvment-proposal-v7.md): apply the frozen layer-15 Cβ contact head before compression, then fit six compact contact-map features to test improvement beyond native + Vx. It reused the original 4,000 TRAIN / 4,000 VAL pairs with shuffled, map-coordinate, boundary and quality/length controls. All new fitting and calibration used TRAIN only; **no TEST access**. The protocol was frozen before the Vx TEST results.

**Completed and independently verified** on 11 October in allocation **3665593**: about **8 hours**, **8.44 GPU-hours including preparation**, all **8,000 records and 8,906 contact maps** checked. On the original 958-pair assessment, AP was **native 0.640930 / native + contacts 0.641509 / native + Vx 0.664361 / native + Vx + contacts 0.664466**. Contacts added **+0.000105 AP over Vx**, 95% protein interval **[−0.000627, +0.001011]**, below the preset +0.005 target; pairing and spatial controls showed no convincing separation. **No demonstrated useful gain from these six contact summaries; no-go/inconclusive.** See [results](pilot-v7-v1/results/REPORT.md).

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

- **R, repeated query:** repeat the query pair to Vx depth to test depth alone. **On hold.**
- **Depth 512:** discussed; no experiment or result yet. Depth 256 is now Vx-v5 above.
