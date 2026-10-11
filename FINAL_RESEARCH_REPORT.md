# iPIN: final research report

**Evidence cutoff: 11 October 2026.** This report consolidates completed iPIN experiments and the relevant, corrected findings from the historical iPIN-OpenPPI repository. It uses saved reports, metric summaries, verification records and resource accounting. Preparing it involved no new model fitting, inference, or access to raw TEST examples or predictions.

## Research conclusion

**Close the current round of incremental model development. Preserve Vx as a modest benchmark improvement, without claiming improved biological partner recognition or pairing-specific co-evolution.**

The strongest current result is frozen PLM-interact plus Vx-128 on the original Bernett TEST: AP increased from **0.690319 to 0.698620**, a paired difference of **+0.008302**, with a conditional 95% protein-bootstrap interval of **[+0.005190, +0.011384]**. The improvement also exceeds the tested quality/profile control. This is useful positive evidence. It missed the declared practical target of +0.010 AP, and the historically examined TEST is not independent confirmation. The interval includes +0.010, so the result does not prove that the underlying gain is smaller than that target. [Full Vx benchmark](benchmark-vx/results/REPORT.md)

The broader campaign narrows the interpretation:

- True and pairing-shuffled MSAs produced similar pilot gains; a pairing-specific advantage remains unresolved.
- Joint query-only encoding did not recover that gain. Local pooling, independent monomer representations, independently sampled homologs, increased depth, and compact contact-head summaries did not establish an improvement over the original Vx approach.
- V8 did not demonstrate an incremental Vx benefit for choosing partners within an anchor or for endpoint-balanced partner swaps. Its external assay comparison stopped before inference because the panel missed a prespecified support floor.
- There were other real, task-dependent findings: V2 transferred better than native Bernett across all five tested species, and V5 improved on the custom ILP-negative test. These prevent a conclusion that PLM-interact is unbeatable or that all iPIN work failed.

The evidence supports an evaluation and methodology contribution: separating global benchmark ranking, dataset sampling effects, evolutionary-context features, and partner-specific performance. It does not yet support a general architecture-superiority, binding-mechanism, or novel-interaction discovery claim.

## 1. What the numbers measure

AP is **average precision**, not accuracy. Its noninformative reference depends on positive prevalence. Scores from different rows below must not be treated as one leaderboard.

| Evaluation | Population and purpose | Main qualification |
|---|---|---|
| Original Bernett TEST | 52,048 pairs; 26,024 positives; native Bernett AP 0.690319 | Historically reused during research; sampled negative labels are not verified noninteractions. |
| Custom V5 ILP-negative TEST | 52,048 pairs; same 26,024 positives as original TEST | Different negative distribution; shares 1,154 negatives too. Not independent replication. |
| MSA pilots | Fixed 4,000 TRAIN / 4,000 DEV pairs; primary assessment 958 pairs | Repeatedly examined assessment; 659 pairs eligible for the original Vx branch. |
| Full Vx TRAIN/VAL | 163,192 raw TRAIN / 59,260 VAL pairs; primary assessment 13,055 previously unevaluated rows | New rows can share proteins and families with pilot rows. Full VAL includes calibration. |
| Five-species transfer | 242,000 released observations, approximately 1 positive to 10 negatives | Different source and prevalence; repeated observations and exposure require qualification. |
| Historical OpenPPI C3 | Design-weighted positive–unlabeled concordance; both endpoints held out from its training | Different task, encoder/training regime and metric; C3 is not universal family separation. |
| External Y2H panel | Selected experimentally tested candidates with assay detections/non-detections | A single assay's non-detection is not universal noninteraction; reference sequences are not verified constructs. |

The current native comparator is the **Bernett PLM-interact checkpoint**, evaluated with full sequences and mean A–B/B–A logits. Current native-like comparisons use FP32 parameters, qualified BF16 autocast and disabled TF32. The humanV11 checkpoint is a different model trained on a different dataset. V1–V4 used cleaned TRAIN/VAL counts of 163,085/59,258; full Vx used the raw release counts above. These denominator and scoring differences are intentional and documented. [V1 protocol/results](benchmark-v1/REPORT.md), [Vx TRAIN/VAL](retrain-vx/results/REPORT.md)

Except where stated otherwise, intervals below are saved, paired, descriptive 95% intervals conditional on fitted models. They do not include the uncertainty from retraining, repeated research decisions or all remote-family relationships. An interval spanning zero does not establish equivalence.

## 2. Reproduction and direct PLM-interact retraining

The native reproduction covered all 294,889 released cross-species, Bernett and mutation rows. Main released-checkpoint metrics were reproducible when input handling and scoring conventions were respected. Important qualifications were retained: deposited Bernett scores were consistent with a post-hoc inferred 3,570-token cap; mutation code used `sigmoid(mutant_logit - wild_logit)` rather than the paper's written probability-ratio formula; individual predictions could change under chain reversal despite similar aggregate AP. The subsequent iPIN benchmark kept its declared full-sequence, symmetric scoring convention. These findings validate a usable reference while showing why matching only rounded paper metrics is insufficient. [Reproduction report](plm-interact-reproducability/REPORT.md)

| Study | Intervention | Original Bernett TEST AP | What it established |
|---|---|---:|---|
| [V1](benchmark-v1/REPORT.md) | Full-length reference and symmetric training | 0.675669 / 0.678164 | Both below native; symmetry advantage unresolved. |
| [V2](benchmark-v2/REPORT.md) | Reference; training-length cap; positive weighting; clean classification loss | 0.669386 / 0.685180 / 0.672726 / 0.690596 | Clean BCE was strongest here, but its native difference was +0.000277 [−0.007765, +0.008157]. |
| [V3](benchmark-v3/REPORT.md) | Learning-rate screen and residue-MLP readout | 0.685553 | Internal advancement target missed; final native difference −0.004766 [−0.012982, +0.003517]. |
| [V4](benchmark-v4/REPORT.md) | ESMC backbone and chain-aware positional attention | Standard ESMC: 0.681420 | No native AP gain. Both modified-attention arms were stopped; they received no TEST evaluation. |
| [V5](benchmark-v5/REPORT.md) | Larger human corpus and degree-balanced, functionally matched ILP negatives | ESM2: 0.6917; ESMC: 0.6871 | ESM2 minus native +0.0014 [−0.0095, +0.0120]; no established original-TEST improvement. |
| [V6](improvment-proposal-v6.md) | Proposed STRING v12.5 development | Not run | A proposal, not a failed experiment. |

These comparisons are not a balanced factorial study. Most conditions have one full-data training seed; V1, V3 and V5 stopped early, and selection opportunities differ. V3/V4 changed more than one factor relative to native. Clean BCE removed input masking and the auxiliary masked-residue objective together, so its effect cannot be assigned to either change alone.

V4 had a specific mechanistic diagnosis: removing cross-chain positional rotations strongly redirected pretrained attention and damaged masked-residue prediction despite passing forward/gradient checks. For the ESM2 probe, cross-chain attention mass rose from roughly 21% to 64%, and masked-residue accuracy fell from roughly 51% to 9%. This supports incompatibility of that intervention with the pretrained network, not rejection of every chain-aware architecture. [Saved diagnostic summary](retrain-v4/analysis/chain-attention-investigation-20261001/summary.json)

On V5's custom ILP-negative TEST, ESM2 reached **0.6570 AP versus native 0.6377**, difference **+0.0193 [+0.0092, +0.0305]**. All positives are shared with the original TEST. This establishes sensitivity to the negative population, not two successful independent evaluations. The wider comparator study also recorded X-PAIR Bernett at 0.6992 AP on the original TEST, with exact processed checkpoint-training membership unresolved. Vx therefore should not be described as a demonstrated improvement over every released predictor. [V5 results and exposure audit](benchmark-v5/REPORT.md)

## 3. What the MSA experiments established

### The common Vx experiment

Vx keeps PLM-interact and MSA Pairformer frozen. It reads the inter-chain representation after zero-based layer 15, projects 256 channels onto 32 fixed random directions, and records mean, standard deviation, maximum and 95th percentile: **128 global features**, averaged across orientations. A small TRAIN-fitted scaler/PCA/logistic head supplies evidence fused with the native logit. The quality/profile control uses 68 symmetric alignment/profile statistics without learned Pairformer representations. Fusion weights are selected only on the designated calibration subset. [Original protocol](pilot-vx-series/pilot-vx/README.md)

The original MSA cap is **127 homolog rows plus the query, total depth 128**. Homologs are joined by shared genomic accession; eligibility requires at least eight homologs, sufficient high-quality coverage in each chain, and combined query length ≤1,536. “True” does not establish that every matched homolog pair biologically interacts. Paired-branch availability was **1,825/4,000 TRAIN pairs (45.6%)** and **2,628/4,000 DEV pairs (65.7%)**. These are not estimates of how many proteins lack any monomer MSA. Unsupported pairs retain exactly the native score; computation errors invalidate a run instead of becoming fallback.

Whole sequence components split pilot DEV into 1,068 calibration, 958 assessment and 1,974 crossing rows. Crossing rows are retained in descriptive full-DEV results but do not select fusion. Native AP on the common assessment is **0.640930**; the original quality/profile fusion reaches **0.650942**.

| Study | Change from original Vx | Assessment AP | Decisive result or limitation |
|---|---|---:|---|
| [Vx](pilot-vx-series/pilot-vx/results/REPORT.md) | Original matched and taxonomically shuffled pairing | True 0.664361; shuffled 0.663248 | True − shuffled +0.001113 [−0.009639, +0.013333]: no demonstrated pairing advantage. |
| [Vx-v2](pilot-vx-series/pilot-vx-v2/results/REPORT.md) | Local block pooling and spatial scrambling | Local true 0.655735; local shuffled 0.662517 | Change in true–shuffle gap −0.007896 [−0.023482, +0.007892]; no demonstrated locality rescue. |
| [Vy](pilot-vy/results/REPORT.md) | Independent monomer-MSA representations plus native | Main monomer arm 0.640930; query/profile control 0.645863 | Calibration selected zero weight for the main monomer arm. Different availability/readout limits comparisons with Vx. |
| [Vz / Q](pilot-vz/results/REPORT.md) | Joint two-chain query-only encoding, depth 1 | 0.643861 | Q − shuffled −0.019387 [−0.035510, −0.000528]; joint query processing did not recover the homolog-input gain. |
| [Vx-v3 / C](pilot-vx-series/pilot-vx-v3/results/REPORT.md) | Shuffle homolog columns; preserve query, depth and unweighted column profiles | 0.650079 | C − true −0.014282 [−0.031036, +0.004880]; suggestive point drop, unresolved contrast. |
| [Vx-v4 / I](pilot-vx-series/pilot-vx-v4/results/REPORT.md) | Independently sample intact A/B homologs at the same depth | 0.649347 | I − true −0.015014 [−0.029861, −0.000989]; I − shuffled interval spans zero. No demonstrated recovery. |
| [Vx-v5](pilot-vx-series/pilot-vx-v5/results/REPORT.md) | Nested extension to total depth 256 | True 0.659059; shuffled 0.659753 | True256 − true128 −0.005302 [−0.018242, +0.007690]; interval excludes the targeted +0.010 gain, not every smaller gain. |

Depth 256 was a real extension: 4,072 of 4,453 eligible TRAIN/DEV pairs gained rows; 3,566 reached 256. The true128 rows were an exact prefix of true256. The larger shuffled permutation could change that prefix's pairing. Repeated-query arm **R remains unrun**, and depth 512 was not tested.

### Mechanistic interpretation

**Evolutionary family context remains a plausible explanation, not an identified cause.** Shuffling B homolog rows preserves intact A and B homolog sequences, their marginal family information and coarse taxonomic structure. Conserved residues, domain organization and family compatibility can therefore survive. The experiment does not erase every evolutionary or interface-related signal, and the lack of a resolved true–shuffle difference does not establish exact equivalence.

Q makes joint query-only encoding an inadequate explanation for the observed gain under this readout. However, Q changes both homolog content and input depth, while retaining the original MSA-derived masks and reliability gate. It is neither a clean depth-only test nor a fully MSA-free workflow. Vy's failure similarly concerns its tested representation, coverage and low-capacity fusion, not all uses of monomer evolution.

C disrupts coherent homolog sequences, gap patterns, phylogeny and dependencies simultaneously, and creates unusual encoder inputs. Its decrease cannot uniquely identify inter-protein covariance. Its residual +0.009149 AP over native has interval [−0.005773, +0.021985] and does not separate from quality/profile. **AP 0.650079 in C is not evidence that surviving pair-specific signal explains the residual.** I also changes homolog composition, diversity and coverage while removing the accession constraint. Its result supports keeping the tested original recipe, not asserting that accession matching is biologically necessary.

The early ≤512-residue true–shuffle pattern motivated Vx-v2. Smaller inter-chain maps might dilute a sparse interface less, but that was a hypothesis. The local-pooling follow-up did not establish a larger pairing gap; its short-versus-long local-gap interval also crossed zero. Length, positive prevalence, MSA support and family composition can vary together. No validated short-protein operating rule emerged. [Vx-v2 diagnostics](pilot-vx-series/pilot-vx-v2/results/metrics.json)

Vx-v2 added paired depth, Neff at 80% identity, occupancy/joint support, effective covered area, length asymmetry, taxonomic diversity and actual shuffle-change diagnostics. Original Neff at the deduplication threshold could equal selected depth and should not be treated as an independent information count. Complete family annotations remained unavailable in the pilot-only inputs; witness-removal sensitivity did not establish family-disjoint generalization. These diagnostics improved interpretation without resolving the source of the gain.

## 4. Vx at full TRAIN/VAL and TEST scale

Full retraining fitted the same two small true/quality heads on **74,576 eligible TRAIN pairs**, preserving both frozen encoders. Both fusion weights were 1. There was no full-scale shuffled arm, as requested.

| Population | Native AP | Quality/profile AP | True Vx-128 AP | True − native, paired 95% interval |
|---|---:|---:|---:|---|
| Primary VAL, 13,055 new assessment rows | 0.647962 | 0.651903 | 0.654981 | +0.007019 [+0.002428, +0.012237] |
| Original TEST, all 52,048 pairs | 0.690319 | 0.692603 | 0.698620 | +0.008302 [+0.005190, +0.011384] |

On primary VAL, true minus quality was **+0.003078 [−0.001289, +0.007732]**. On TEST it was **+0.006017 [+0.002012, +0.009944]**. TEST AUROC increased from **0.699467 to 0.703791**. MSA evidence applied to **32,340 TEST pairs (62.1%)**; the other 19,708 retained native scores. [TRAIN/VAL results](retrain-vx/results/REPORT.md), [TEST results](benchmark-vx/results/REPORT.md)

The full-data gain is smaller than the +0.023431 pilot point estimate. This is consistent with the limited precision and repeated examination of a small pilot, but changed fitting scale and population prevent assigning the shrinkage to one cause. Of the 13,055 new VAL assessment rows, 12,908 share a protein with the pilot VAL cohort; they are not a fresh family holdout. [Full-data metrics](retrain-vx/results/metrics.json)

Both full studies returned **`no_go_or_inconclusive` under their frozen practical criteria**: at least +0.010 AP over native with a positive lower interval, no material AUROC loss, and positive separation from quality/profile. The TEST run was explicitly authorized after the TRAIN/VAL no-go. Its weights and fusion were frozen before TEST scoring; no TEST fitting occurred. Nevertheless, prior iPIN TEST results had already informed research decisions. The conditional gain is credible within this saved comparison, while independent generalization and selection-adjusted uncertainty remain unresolved. The absence of a full shuffled arm also prevents transferring a pairing-specific claim from the pilot to this benchmark.

## 5. V7: did the pretrained contact head add useful evidence?

V7 used the frozen layer-15 Cβ contact head and six compact summaries of contact strength, contact extent, residue participation, concentration and a local window. It included pairing-shuffled, coordinate-scrambled, boundary and quality/length controls. All new fitting and fusion selection used TRAIN; no TEST was accessed.

On the same 958 assessment pairs:

| Frozen combination | AP |
|---|---:|
| Native | 0.640930 |
| Native + contacts | 0.641509 |
| Native + Vx | 0.664361 |
| Native + Vx + contacts | 0.664466 |

The increment over Vx was **+0.000105 [−0.000627, +0.001011]**. Its upper interval is well below the declared +0.005 target. Pairing and spatial controls did not show convincing separation. All 8,000 records and 8,906 maps passed independent reconstruction checks. The implemented six-feature readout therefore supplies no useful complementary gain at the targeted scale. This does not rule out every contact-map model; it does remove the empirical justification for automatically expanding this particular idea. Contact-head training exposure is unresolved, its outputs are not calibrated human PPI probabilities, and native TRAIN scores used by the fusion were in-sample for native. [V7 report](pilot-v7-v1/results/REPORT.md), [verification](pilot-v7-v1/results/verification.json)

## 6. V8: partner diagnostics and the external-assay gate

V8 reused the unchanged full Vx, native and quality VAL scores. No model was trained or newly encoded. Its primary view was the same 13,055 new assessment rows, now retrospectively examined. It asked whether the global AP improvement also helped discriminate partners after reducing simple endpoint effects.

Within-anchor concordance gives each eligible protein equal weight when comparing its positive- and negative-labelled partners. Retrieval uses fixed candidate lists with at least 20 negative-labelled candidates. Swaps compare two existing positive edges with two existing negative-labelled edges joining the same four distinct proteins; additive endpoint scores cancel exactly. None of these labels was newly manufactured.

| Diagnostic | Support | Native | Vx | Quality | Vx − native, paired 95% component interval |
|---|---|---:|---:|---:|---|
| Within-anchor concordance | 1,445 anchors | 0.644579 | 0.647601 | 0.645110 | +0.003022 [−0.002613, +0.009953] |
| Recall@10 | 151 anchors | 0.238203 | 0.237507 | 0.240727 | −0.000697 [−0.004187, +0.002450] |
| Swap preference | 2,000 quartets | 0.744000 | 0.746500 | 0.748000 | +0.002500 [−0.023174, +0.030710] |

Vx minus quality was **+0.002491 [−0.003584, +0.009242]** within anchors and **−0.001500 [−0.032213, +0.029766]** on swaps. The secondary all-assessment view gave the same qualitative conclusion. All declared diagnostic support floors passed, but the research screen requiring ≥0.010 improvements and positive lower bounds over native/quality did not. Sequence-composition and 3-mer controls were weaker on swaps; native already outperformed them, so that does not establish a new Vx contribution. These diagnostics do not prove that Vx lacks partner information, nor do they erase the AP gain. [V8 report](benchmark-v8/results/REPORT.md)

### Why no external Vx score exists

V8 audited the historical curated human Y2H panel before any external inference:

| Prespecified stage | Pairs | Detections | Assay non-detections |
|---|---:|---:|---:|
| Historical exposure-excluded primary panel | 1,940 | 77 | 1,863 |
| Remove exact current Bernett TRAIN/VAL pairs | 1,770 | 69 | 1,701 |
| Also remove every pair touching a historical TEST protein hash | 1,116 | 40 | 1,076 |
| Also satisfy Vx combined-length limit | 1,024 | 38 | 986 |

The TEST-protein exclusion reused a cached hash list; it did not reopen TEST examples. It is **stricter than excluding exact TEST pairs**, and materially reduces the population. Even if every length-eligible positive had a usable MSA, the maximum was **38**, below the predeclared **40 usable-positive floor**. Only seven positives had both monomers already cached, but missing cache entries mean unknown MSA availability, not demonstrated absence.

The run therefore stopped before new inference, further training-homology/component qualification, or exact paired-MSA reconstruction. Status: **`no_go_insufficient_panel_support`**. This is a procedural feasibility outcome under a conservative design, **not a failed external Vx performance test**. It does not imply that 38 positives are mathematically useless or that the 40-positive floor guarantees power to resolve a small effect. The floor was not relaxed after seeing attrition. [External feasibility audit](benchmark-v8/results/external_feasibility.json), [frozen proposal](improvment-proposal-v8.md)

## 7. Dataset construction and transfer explain part of the puzzle

**More balanced data did not automatically produce better transfer.** On the five-species benchmark, V2 clean BCE and length-capped checkpoints exceeded native Bernett AP in all five species. Unweighted mean AP was **0.3466 / 0.3167**, versus **0.2234** for native Bernett. V5 ESM2/ESMC reached **0.1497 / 0.1231**. V2 clean BCE reached 0.5241 in E. coli versus native Bernett 0.1572. These are meaningful fixed-checkpoint results on that task, with source/exposure and sampled-negative qualifications. They are not directly comparable to balanced Bernett AP. [Transfer comparison](benchmark-v5-nonhuman/v1-v4-comparison/REPORT.md)

**Some simple explanations work strongly in one dataset and weakly in another.** V11's exact TRAIN endpoint-degree score reached human validation **AP 0.836014 / AUROC 0.963460**; all 15,351 validation sequences occurred in TRAIN. Removing exact repeated pairs barely changed it. Homology-transferred degree also reached mean five-species AUROC 0.8591. This demonstrates a strong sampling/endpoint signal, not a complete explanation of the native humanV11 model. [Exact-degree audit](literature/string-v11-train-degree-baseline/REPORT.md), [V11 transfer controls](literature/string-v11-baselines/REPORT.md)

Conversely, the tested V5 degree/sequence-propensity/interolog baselines were near chance on its balanced human tests. Across species, V5 interolog lookup reached mean AP **0.2071**, exceeding both V5 neural models in every species. Bernett versions of those simple baselines did not reproduce V2's stronger transfer. Conserved interaction lookup can be legitimate biological signal; low performance of a finite control panel does not certify an unbiased dataset. V5 degree matching also creates near-constant targets and, under some audit splits/fallback conventions, inverse rather than absent association. [Matched simple-baseline study](benchmark-v5-baselines/REPORT.md)

Two completed local audits sharpen these qualifications:

- **FADI:** under strict family/clan and annotation-coverage rules, original-TEST separation indices were **11.12/100 for Bernett TRAIN** and **6.68/100 for V5 TRAIN**. Neither dataset is fully family-disjoint. Broad clan relationships are not proof of interaction leakage, and unresolved annotation is not a detected relative. [FADI report](FADI-benchmark-v1/results/REPORT.md)
- **TRIQ:** the declared source-conditional index was **V5 94.60, Bernett 92.29, V11 26.87**. Penalizing inverse as well as positive shortcut association changes V5 to **83.54**, reversing its ordering with Bernett. These are exploratory operational indices, not biological label-accuracy percentages or a validated universal quality ranking. [TRIQ report](TRIQ-benchmark-v1/REPORT.md)

Thus the current bottleneck is partly the meaning and independence of evaluation, not simply a lack of model capacity. That is an inference from the combined evidence, not a measured decomposition of every model's performance.

## 8. What the historical iPIN-OpenPPI repository adds

The historical project supplies useful independent lines of investigation, but not an independent replication of current Vx. It studied a TUnA-derived positive–unlabeled model, different training data and frozen ESM2-150M features. Its released PLM-interact comparator used humanV11, a 1,603-token cap and different orientation handling. Its C3 concordance must not be compared numerically with current Bernett AP. The latest corrected finalization takes precedence over older headline summaries. [Historical final report](../iPIN-OpenPPI/experiments/study_finalization_v1/REPORT.md)

| Historical finding | Result | Relevance to current decision |
|---|---|---|
| Larger corpus versus corrected fresh 17k control | C3 0.7867 versus 0.7698; difference +0.0168 [−0.0015, +0.0350] | Overall benefit unresolved; corpus size, source composition and sampling changed together. |
| Matched-data pooled pair baseline | C3 0.7794 versus default 0.7867; default advantage +0.0073 [−0.0125, +0.0254] | No resolved default advantage despite architectural complexity. Training data/encoder matched, optimization/capacity did not. |
| Early partner retrieval | Recall@10 default 0.1396 versus pooled 0.1733; difference −0.0337 [−0.0528, −0.0158] | Better global concordance did not imply better early retrieval. |
| Separate Y2H panel | Default AUROC 0.6115 [0.5087, 0.7174], AP 0.0556; prevalence 0.0397 | Modest assay discrimination; no resolved advantage over comparators; all five models had zero positives in the top ten. |
| Cached scoring efficiency | Approximately 351,000 versus 2,244,000 pairs/s, default versus pooled | Simpler models can be materially cheaper; these timings exclude shared encoder cost and use different native batch sizes. |

Historical endpoint controls did not explain everything: the default exceeded a linear endpoint-only C3 score, and its swap preference was 0.755 versus the additive control's 0.5. However, swap advantages over amino-acid composition, 3-mer and pooled-embedding cosine controls remained unresolved. Earlier partner-conditioned residue cross-attention also failed to establish superiority over its matched historical TUnA comparator. These findings motivated V8's diagnostic design; they do not prove current Vx learns only endpoint effects. [Expanded controls](../iPIN-OpenPPI/experiments/expanded_controls_v1/results/INTERPRETATION.md), [residue cross-attention study](../iPIN-OpenPPI/benchmark/partner_conditioned_residue_v1/results/RESULTS.md)

Historical implementation corrections matter. An early embedding-index mismatch affected 16,999/17,000 development row mappings; correcting it changed a roughly chance result to C3 0.784142. That old chance score is invalid evidence of model failure. A separate later GP-covariance scoring correction is incorporated in the fresh-17k comparison above. The latest finalization's recorded checks support those corrected comparisons; they do not turn retrospective studies into blind tests or replace biological validation. [Embedding identity reassessment](../iPIN-OpenPPI/docs/reports/m1/M1_Research_Reassessment_and_Embedding_Identity_Findings_2026-09-10.md), [final claim guidance](../iPIN-OpenPPI/experiments/study_finalization_v1/MANUSCRIPT_GUIDANCE.md)

## 9. Evidence strength, integrity and cost

The evidence fits together without requiring either “nothing worked” or “the mechanism is solved”:

| Claim | Assessment |
|---|---|
| Frozen Vx adds some ranking information beyond native and the tested quality/profile control on original Bernett TEST. | Supported within the saved, historically selected comparison. |
| Correct homolog pairing produces the improvement. | Not established; pilot true–shuffle uncertainty remains, and full-scale shuffle was not run. |
| Joint two-chain query encoding alone explains Vx. | Not supported under the tested readout; Q was worse than both homolog-input arms. |
| Coherent homolog context is useful. | Plausible synthesis of Q/C/I results; content, depth and distribution effects are not fully separated. |
| Vx improves biological partner specificity or prospective discovery. | Not demonstrated by V8 or an independent external assay. |
| Every attempt to improve native failed. | Incorrect: Vx, V2 transfer and V5's ILP-negative result are qualified positive findings. |
| More architecture/MSA variants are currently justified. | No convincing empirical basis for another expensive round with the same evaluation. |

Saved qualification and verification cover checkpoint identities, sequence/row joins, frozen prediction reuse, feature reconstruction, scoring rules, fallback, and metric calculations. V7 reconstructed every map-derived feature; V8 independently rebuilt all point metrics, selections and reported intervals, with additional raw bootstrap checks. These are substantial safeguards against the specific execution errors checked. They do not prove absence of every software defect, hidden pretraining exposure or statistical confound. **Successful execution, failure of a scientific advancement criterion, an early-stopped training arm, and a panel that never reached inference are distinct outcomes.**

Important remaining limits are shared across the campaign: repeated validation/TEST examination; mostly single-seed full retraining; uncertain sampled-negative labels; incomplete family independence; unresolved source-checkpoint exposure; MSA eligibility changing the served population; low-capacity summaries potentially discarding information; and controls that change several biological properties together. V8's 2,000-draw component intervals use a heuristic 40%-identity/80%-coverage graph; retrieval leaves shared-partner dependence conditional on fixed candidate lists. None of these intervals incorporates all adaptive research choices.

| Recorded stage | Resource use | Accounting scope |
|---|---:|---|
| Full Vx TRAIN/VAL | 108.46 allocated GPU-hours; 8 GPUs, about 13 h 33 min | Includes batch idle phases; preflight adds 0.62 GPU-hours. |
| Full Vx TEST | 23.13 allocated GPU-hours; 4 GPUs, about 5 h 47 min | Includes batch idle phases; preflight adds 0.45 GPU-hours. |
| V7 | 8.44 GPU-hours | Active allocation interval including preparation. |
| V8 | About 36.8 seconds recorded execution; 8 CPU threads | CPU processes inside an existing GPU allocation; no new allocation or encoder calls. Excludes implementation time. |

Full Vx TRAIN/VAL plus TEST alone consumed approximately **132.66 allocated GPU-hours including their preflights**. This is not total project cost: earlier pilots, retraining, MSA preparation, development and other benchmarks are excluded. The measured gain must be weighed against that recurring MSA/encoding burden and incomplete eligibility; the reports do not establish a deployment cost–benefit threshold. [TRAIN/VAL accounting](retrain-vx/results/resources.json), [TEST accounting](benchmark-vx/results/resources.json), [V7 completion](pilot-v7-v1/results/COMPLETE.json), [V8 accounting](benchmark-v8/results/resources.json)

## 10. Final disposition

**Retain the native Bernett reference and frozen Vx-128 result, and stop the present sequence of incremental model experiments.** Preserve the V2 transfer and V5 negative-distribution findings as task-specific results. Do not relabel Vx's practical no-go as a full success, its modest AP improvement as zero, or V8's unperformed external comparison as a model failure.

The appropriate next output is a research account built around reproducibility, controlled ablations, dataset/exposure audits and the difference between global ranking and partner retrieval. Whether that becomes a publishable benchmark or methodology paper requires a separate assessment of novelty, release completeness and venue; these results alone do not guarantee publication.

Reopen model development only when there is a materially better evaluation opportunity: an independently measured, exposure-audited candidate population large enough to resolve the intended paired improvement; adequate Vx coverage checked before expensive inference; a defined biological task and practical effect size; and frozen comparators tested before adding more model variants. Sample-size planning must consider clustered dependence and paired-score variability, rather than treating 40 positives as sufficient power. Existing TEST results cannot become a fresh blind holdout through another protocol freeze.

This is a decision about the evidence and cost of **this research direction at this stage**. It is not a proof that meaningful improvement over PLM-interact is impossible.

## Source and preservation notes

The linked experiment reports and their machine-readable metrics remain the numerical authorities; this document does not replace their protocols or alter their decisions. The key consolidated result records are [Vx TRAIN/VAL metrics](retrain-vx/results/metrics.json), [Vx TEST metrics](benchmark-vx/results/metrics.json), [V7 metrics](pilot-v7-v1/results/metrics.json) and [V8 diagnostics](benchmark-v8/results/validation_diagnostics.json). Saved JSON/Markdown summary hashes for full Vx and V8 were checked against their completion manifests while preparing this report. Existing scientific source artifacts were preserved.

Repository snapshot: current iPIN HEAD `6ebc138cda79e4278662463129e2b0304953d4af`, plus completed local V8, FADI and TRIQ files; historical iPIN-OpenPPI HEAD `3db109ba2702cdf72f7327f6d067761987e6cde1`, plus its completed local follow-up/finalization studies. Those local studies were not all committed at this cutoff. Historical links require the sibling checkout; large predictions, checkpoints, alignments and containers remain archived outside Git as described by the study manifests. [Short experiment walkthrough](walkthrough.md)
