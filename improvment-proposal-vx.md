# iPIN Vx: paired-MSA evidence on original Bernett

**Revised 7 October 2026. Pilot authorized and implementation initiated in [pilot-vx](pilot-vx/README.md).** This proposal is distinct from the STRING v12.5 V6 proposal. Current execution evidence belongs in the pilot's status and provenance; feasibility assumptions below are not measurements.

## 1. Question and fixed comparison

Does a frozen paired-MSA encoder add interaction information to the frozen **native Bernett-trained PLM-interact checkpoint**, on the original released Bernett task, beyond alignment quality, individual-family profiles and shared phylogeny?

Use one frozen sequence baseline, one frozen MSA Pairformer trunk, and a small regularized TRAIN-fitted evidence head. Retain baseline scores exactly when the MSA channel is unavailable or disallowed. No new negative sampling, STRING training, V5 supervision, PLM retraining, encoder fine-tuning, structure prediction campaign or local database search is part of this pilot.

The baseline is the native leakage-free Bernett checkpoint, SHA256 `207f1bf02e9bebc393fe510a98c0f9af04039d0a9258047ac107c6d24221c7e2`. Follow the established benchmark-v1 full-sequence, mean-AB/BA-logit inference contract, with FP32 weights and BF16 autocast, TF32 disabled. This explicitly differs from the capped, single-orientation historical publication reproduction. Reuse predictions only when row identities, checkpoint, precision, orientation and length policy agree. Native TRAIN predictions are not inputs or residual targets for fitting the evidence head.

Original Bernett and V5 are useful difficult benchmarks, but neither establishes universal family independence; see [FADI](FADI-benchmark-v1/results/REPORT.md). Historical Bernett test results have informed earlier rounds. A later Vx test result is exploratory confirmation, not a newly blind evaluation.

## 2. Exact dataset and scale

| Original released split | Pairs | Positive | Negative | Distinct sequences |
|---|---:|---:|---:|---:|
| TRAIN | 163,192 | 81,596 | 81,596 | 4,286 |
| DEV | 59,260 | 29,630 | 29,630 | 3,711 |
| TEST | 52,048 | 26,024 | 26,024 | 3,022 |

Use the pinned original release (`5d2ad03baa165c27df32a2eadf066462a2a83073`) represented by `retrain-v1/data/prepared/{train,val,test}.raw.npy`. Verify the TRAIN/DEV rows against their source CSVs. Preserve all labels and memberships. The later audited preparation removed 107 TRAIN and two DEV rows; that is a different 274,391-row variant and is not silently substituted here.

The original release has **274,500 pairs and 11,019 distinct sequences**, with no exact sequence overlap between its splits. The pilot reads TRAIN/DEV labels only. Existing test sequence hashes may be used solely to exclude exact held-out human sequences from training-context homolog rows; no test features, predictions or metrics are generated in the pilot.

A later reduced production design would use 50,000 fixed TRAIN pairs plus complete DEV and original TEST: **161,308 pairs**. The custom ILP test and V5 models are optional later comparisons, not pilot requirements. Cache monomer homolog context once per exact sequence.

Before quality masking, combined length exceeds 1,536 residues for 42.16% of TRAIN, 19.34% of DEV and 17.62% of TEST. Long examples remain in the evaluation population through exact baseline fallback. A coverage-only result must not be reported as full-population improvement.

## 3. What paired coevolution would contribute

For candidate human proteins A and B, an informative paired MSA has rows resembling:

```text
human query:       A_human          | B_human
ortholog context: A_species_1      | B_species_1
ortholog context: A_species_2      | B_species_2
                  ...              | ...
```

Positions are aligned within each family. Each non-query row should pair plausible corresponding homologs from the same organism or matched genomic dataset. Interface compatibility can constrain substitutions across the boundary: a change in A can be associated with compensating changes in B.

The desired evidence is **cross-chain dependence beyond the marginal conservation of A and B**. Simply concatenating two independently ordered MSAs does not supply valid paired coevolution. Neither do two mean MSA embeddings, which mainly retain separate-family information.

Three questions must remain distinct:

1. **Partner assignment:** which paralog of family A goes with which paralog of family B in another species?
2. **Contact inference:** which residues could form an interface, given an alignment?
3. **Our task:** does this particular human pair receive the positive interaction label used in the benchmark?

Success on the first two questions does not guarantee success on the third. A family can retain some interacting members while the queried human paralogs do not interact. Conversely, a real recent, transient, or motif-mediated interaction may carry little detectable family-wide covariance.

## 4. Literature: support, limits, and consequences for Vx

The following are primary papers or author resources. Findings are separated from my proposed use of them.

| Source | What it contributes | Consequence for iPIN |
|---|---|---|
| Hopf et al., *eLife* 2014, [EVcomplex](https://doi.org/10.7554/eLife.03430) | Inter-protein evolutionary couplings can identify interface residues and help assemble complexes. Alignment construction and evolutionary diversity matter. | A classical coupling score is a useful mechanistic reference, but small conserved-complex successes are not a human interactome guarantee. |
| Green et al., *Nature Communications* 2021, [EVcomplex2](https://www.nature.com/articles/s41467-021-21636-z) | Extends coevolution to interaction classification at scale, with reciprocal-identity pairing and calibrated residue/interaction scores; much evidence is bacterial. | Supports a score-level coevolution branch. Do not transfer its coverage or performance numbers directly to Bernett. |
| Rao et al., ICML 2021, [MSA Transformer](https://proceedings.mlr.press/v139/rao21a.html) | A pretrained row/column-attention model extracts information from an alignment rather than a single sequence. | A possible technical fallback, although its interface behavior and length constraints make it less attractive than the newer candidate. |
| Lupo et al., *Nature Communications* 2022, [phylogeny in MSA language models](https://www.nature.com/articles/s41467-022-34032-y) | MSA representations encode evolutionary relatedness; controlled experiments distinguish phylogenetic correlations from contact information. | A raw correlation is not necessarily a physical interface. Include phylogeny-aware controls. |
| Lupo et al., PNAS 2024, [DiffPALM](https://doi.org/10.1073/pnas.2311887121) | Uses an MSA language-model objective to optimize within-species paralog assignments, with promising difficult-pairing examples. | Potential rescue for ambiguous families, but its repeated optimization is unsuitable as the default for hundreds of thousands of candidates. |
| Lupo et al., 2024, [DiffPaSS paper and detailed methods](https://arxiv.org/html/2409.16142v1) | Differentiable matching with mutual-information or graph scores offers cheaper pairing optimization. Its standard formulation assumes one-to-one partners. | More plausible than DiffPALM for a restricted future rescue step. Human many-to-many interactions and negative pairs remain problematic. |
| Zhang et al., *Science* 2025, [human-proteome PPI prediction](https://doi.org/10.1126/science.adt1630) | Combines substantially deeper human/eukaryotic MSAs with RF2-PPI, a model designed for PPI identification. | The most directly relevant evidence that this information can help human PPI prediction. Reuse its sequence resources rather than regenerate them. |
| Akiyama et al., *Cell* 2026, [MSA Pairformer](https://doi.org/10.1016/j.cell.2026.06.029) | A 111M-parameter MSA/pair-representation model reports strong interface-contact and interface-variant results despite individual-chain language-model training. | Preferred frozen feature extractor, conditional on exposure and runtime qualification. It has not established an AP gain on the original Bernett test. |
| Luo et al., April 2026 preprint, [depth-over-pairing study](https://www.biorxiv.org/content/10.64898/2026.04.14.718427v1) | On 439 heterodimers, AFM/AF3 experiments attribute much of the alignment benefit to added depth; shuffled pairings can preserve improvements. | Essential counterevidence: test paired information against depth-matched controls. This structure-prediction result neither proves nor disproves our classifier hypothesis. |
| Mirdita et al., *Nature Methods* 2022, [ColabFold](https://doi.org/10.1038/s41592-022-01488-1), and Kallenborn et al., *Nature Methods* 2025, [MMseqs2-GPU](https://doi.org/10.1038/s41592-025-02819-8) | Practical, accelerated sequence search and MSA workflows. | Local batched search is a fallback; published speedups are not timing estimates for our database, lengths, or GH200 installation. |

### The human-proteome paper is encouraging, with important scope differences

Zhang and colleagues report sevenfold deeper MSAs generated from a very large collection of unassembled sequence data. Their system combines alignment improvements, model design, and additional training evidence; its overall gain cannot be attributed solely to “adding paired MSA.” Their accuracy estimation also differs from our balanced Bernett/ILP evaluation. Their RF2-PPI training uses structural and domain-interaction information, so its checkpoint requires a separate exposure assessment. [Paper and methods](https://pmc.ncbi.nlm.nih.gov/articles/PMC13281779/).

For Vx, the most valuable immediate asset is the **unlabeled homolog alignment collection**. Importing the authors' published human PPI probabilities would create a different experiment, with additional training/exposure and candidate-selection concerns.

### Why prefer MSA Pairformer, cautiously

The accessible 2025 preprint describes masked-language-model training on OpenProteinSet MSAs and a contact readout fitted using 20 structural examples. That readout is supervised even though the representation learning is self-supervised. Its interface experiment uses conserved complexes; a contact precision result is not a general human PPI AP result. [Preprint and methods](https://www.biorxiv.org/content/10.1101/2025.08.02.668173v1).

The currently released implementation exposes pair representations and contact heads, with explicit complex boundaries. This makes it practical to extract an inter-chain feature channel without retraining an MSA foundation model. The released weights and training provenance must be treated as distinct artifacts; the final publication's details should not be assumed identical to the earlier preprint. [Author implementation](https://github.com/yoakiyama/MSA_Pairformer).

The recommendation is a choice of an existing feature extractor, not a prediction that it will outperform native PLM-interact. Its pretrained contact probabilities must not be presented as calibrated human interaction probabilities.

### Reasons the idea could fail

- **Wrong partners:** human gene duplication, isoforms, lineage-specific loss and functional divergence complicate ortholog pairing.
- **Wrong biological target:** a physical-interface channel is most naturally aligned with direct binding; database interaction labels can reflect broader experimental associations.
- **Insufficient diversity:** hundreds of close mammalian sequences can contain much less independent information than their raw row count suggests.
- **False confidence from shared ancestry:** correlated substitutions can arise without the particular physical interaction we seek.
- **Generic family compatibility:** a family-level interface can exist without the queried human proteins interacting in the relevant context.
- **Coverage shortcuts:** positives may have deeper, more complete, or better-curated alignments than negatives. A model may exploit that instead of coupling.
- **New architecture, old supervision exposure:** structural contact heads or pretrained PPI models can have seen relevant proteins or complexes.
- **Cost:** a smaller parameter count does not make dense residue-pair computation cheap.

These are reasons to test the information carefully, not reasons to dismiss it. The depth-over-pairing paper especially changes the experimental design: an improvement without a pairing-sensitive control would support useful MSA context, but not the stronger claim of useful explicit coevolution.

## 5. Alignment construction and exposure

Start with the unlabeled [human omicMSA archive](https://datadryad.org/dataset/doi:10.5061/dryad.15dv41p84), Dryad version 396143, file 4357017: `protein_omicMSAs.tar.gz`, 17,244,466,432 bytes, SHA256 `0d81c035db2243f0bad088e47508684caf670b6e53c3b1f96715ba3bd6cca58f`. Public mirrors are acceptable only with the identical checksum. Do not import the deposit's human interaction probabilities, DCA prediction tables, PPI-selected candidates or structures.

Its A3M-like format contains a quality mask, exact human query, lowercase insertions and genome/dataset accession headers with taxonomy. Gaps can reflect assembly failure. Match the complete query sequence exactly; an accession or partial match does not authorize replacing a Bernett isoform.

Alignment-format qualification found homolog residue `J`, which is absent from the encoder vocabulary. Normalize homolog J to unknown X without deleting a column, record replacement counts, and continue to reject other unsupported symbols and invalid coordinate lengths. Original human-query matching remains exact.

The fixed 50% homolog coverage requirement applies separately to each chain's high-quality positions after masking, in addition to the monomer full-query coverage check. Rows supported only by discarded low-quality columns cannot create paired evidence.

Keep original query columns and coordinates. Low-quality columns remain in their original positions and are masked; do not concatenate separated good regions into an invented contiguous sequence. Pair rows by the **same genome/dataset accession**, not row number or genus. Keep taxonomy for balancing and null controls. Detect duplicate/conflicting keys and identical homolog sequences reused in both chain slots; reject ambiguous rows and record limitations, including biological instance identity that the archive does not expose.

To bound storage, cache at most 4,096 homolog rows per protein using a label-independent, accession-hash priority shared across proteins. Report raw and retained depth. This is an explicit computational approximation that can reduce pairable coverage. Remove identifiable exact held-out human sequences from TRAIN context, and exact TRAIN human queries from DEV context, before selection. Broader external homolog exposure is allowed and reported separately from supervised exposure.

Use the pinned MSA Pairformer code commit `875363570df1ae484cf725aba382790444223005` and weights revision `7563e77a87536b5572f91683c39073ef348639a4` from `yoakiyama/MSA-Pairformer`. Preserve upstream licenses. Its released contact heads have structural supervision; incomplete head-training membership prevents an exposure-free claim. The primary pilot uses **frozen trunk representations with a Bernett-TRAIN-only readout**; released-head scores may be clearly marked diagnostic features only after an exposure audit. Do not mistake trunk pretraining on unlabeled homologs for PPI-label training.

There is no bulk public ColabFold querying, new sequence-search database, DiffPALM/DiffPaSS optimization, or missing-query rescue in this pilot. Poor coverage is a result that can stop the experiment.

## 6. Predictor and numerical contract

```text
s_vx(A,B) = s_native(A,B) + alpha * available(A,B) * reliability(A,B) * evidence(A,B)
```

The evidence head is a fixed regularized logistic classifier over compact frozen inter-chain representation summaries. Use 128 fixed projected trunk summaries; fit TRAIN-only standardization and 32-component PCA for both encoder views and the quality/profile control, followed by logistic regression with C=0.1. See the pilot README for the exact fixed feature specification. Fit normalization and head parameters on TRAIN only, without native in-sample logits. Use symmetric AB/BA features. The gate is a fixed bounded function of effective paired depth and quality coverage, not a freely trained protein-identity or missingness predictor. Missing evidence produces exactly the baseline logit, with no special intercept. Numerical errors are execution failures, never biological absence.

Start with at most **128 paired rows including the query** and **1,536 combined query residues**. Require at least eight retained paired homologs and 50% high-quality positions in each chain. These are frozen engineering choices, not universal biological thresholds. Use deterministic taxonomically balanced selection and a declared 90%-identity redundancy filter with 80% comparison coverage. Record effective depth with its exact identity/coverage definition.

The first pilot uses no sequence windows. This avoids uncertain interface coverage and window-count inflation. A later window policy would require a separately frozen protocol and cost estimate. Do not enlarge limits after inspecting labels or performance.

DEV is divided by sequence-homology components into internal calibration and assessment protein groups before Vx results exist. Use a label-independent split; crossing edges remain in full pilot DEV reporting but are excluded from the internal selection masks. Choose alpha from `{0, 0.1, 0.25, 0.5, 1}` on calibration only, assess once on assessment, then report all fixed DEV rows without retuning. Insufficient retained rows or either missing class makes selection inconclusive. Prior native selection using DEV means this is not a wholly independent development set.

## 7. Controls and interpretation

Compare the same model capacity and fitting recipe for:

1. True paired inter-chain features.
2. A pairing-disrupted view: preserve the human query, row count and per-chain marginal columns/gaps, shuffle one chain within families, then within orders for remaining singleton groups, then within classes. Leave final singletons unchanged and record counts at each level and the fraction changed. There is no global shuffle in the primary pilot. This hierarchy was fixed during label-free input qualification because family balancing made a family-only null ineffective. Coarse taxonomy does not remove all phylogenetic confounding.
3. Quality and individual-family profile features without homolog pairing: depth, length, quality fraction, gap fraction and conservation summaries. Inspect this control as a possible coverage/curation shortcut.

Real pairing must add value beyond these controls for an explicit coevolution claim. Similar gains after shuffling support useful MSA context, not proven paired coevolution. Report family/depth/length strata, effective coverage, uncertainty and failure reasons. A physical-contact channel may miss transient, disordered or indirectly associated positives; shallow MSAs do not establish noninteraction.

## 8. Bounded pilot and resource limits

Authorized work is in `pilot-vx/`. Immutable configuration, source hashes, sample identities, controls and split assignments must be frozen before feature/label analysis.

1. Qualify parser/coordinate behavior, deterministic pairing, symmetric features, pretrained state loading, GPU memory, baseline equivalence, cache integrity and interrupted resume.
2. Audit 2,000 fixed TRAIN/DEV proteins across lengths, and archive matches for proteins needed by the fixed pair sample. Sample **4,000 original TRAIN and 4,000 original DEV pairs**, balanced by label and proportional to original within-class length strata, including unavailable and long examples. Do not select pairs by alignment success.
3. Extract primary and null features within the budget; fit the small heads and conduct the predeclared DEV analysis if sufficient complete data exist. The sample counts are upper bounds, not authority to exceed the budget.

Hard ceiling: **24 allocated GPU-hours**, including qualification and any missing baseline inference; **2,000 CPU-core-hours**; **150 GB additional working storage**, including the new runtime, downloaded archive, weights and caches. Reserve at most four GPU-hours for qualification; a subsequent batch stage may use at most 20 GPU-hours. Count failed runs and restarts. CPU preparation is bounded separately and total accounting must include allocated CPUs in GPU jobs. Respect walltime and stop with an explicit incomplete/inconclusive result when limits prevent completion; no automatic budget extension or production continuation.

Use independent GPU workers and atomic compact feature caches. Retain hashes and parameters; never persist dense per-residue tensors for every pair. Build an isolated SIF under `images/` when useful. Existing SIFs/checkpoints stay immutable. ARM64, CUDA/cuEquivariance compatibility and the upstream weight-loader repository typo require qualification; the standard PyTorch triangle-update fallback is acceptable if it passes checks and timing.

Continue beyond the pilot only for a credible assessment benefit beyond controls, not confined to a handful of deeply aligned families, with sound full-population fallback and measured affordable throughput. The proposal's practical production criterion remains **at least +0.010 assessment AP**, a positive paired protein-bootstrap interval and no material AUROC decline. This is a decision threshold, not a promised gain; an underpowered pilot can be inconclusive.

## 9. Conditional production cost

A later production run requires a separate decision after the pilot. No full PLM or MSA encoder training is required. For N physical pairs with effective encoder cost t seconds per pair (including all orientations/windows), GPU-hours are `N*t/3600`. Controls, CPU preparation, qualification and uncached baseline scores add cost; unavailable examples need no MSA forward.

| Effective seconds per pair | All 274,500 pairs: GPU-hours | Reduced 161,308 pairs: GPU-hours | Reduced, ideal four-GPU walltime |
|---|---:|---:|---:|
| 1 | 76.3 | 44.8 | 11.2 hours |
| 5 | 381.3 | 224.0 | 2.33 days |
| 10 | 762.5 | 448.1 | 4.67 days |
| 30 | 2,287.5 | 1,344.2 | 14.0 days |

These are scenarios, not throughput measurements. Retain the earlier **300 total GPU-hour development/production planning ceiling** only if measured extrapolation, including the pilot, fits it. Storage expansion and usable MSA coverage must be measured. Four GPUs per Arrhenius node and roughly 95.6 GiB reported device memory per GPU do not guarantee that arbitrary sequence lengths fit; pair representations scale quadratically and pair updates can scale more steeply.

## 10. Reproducibility and completion

Every result records source sequence/pair hashes, archive checksum, encoder weights/code, configuration, selected genome keys, coordinates, masks, precision, worker runtime and explicit eligibility/failure reason. Outputs are committed atomically and resume validates content hashes. The pilot's source freeze and tests must cover exact fallback, TRAIN-only normalization, label-independent pairing, control marginals, AB/BA invariance and refusal of corrupt or incompatible caches.

Report AP/AUROC and paired protein-bootstrap differences on the assessment and complete fixed DEV sample, with class counts and exclusions explicit. Use matched resamples for all models. Do not describe missing computational outputs as biological fallback or silently omit difficult pairs. No test analysis or automatic production launch belongs to this pilot.

The final deliverable is a measured coverage/runtime/added-value decision and reproducible artifacts, including an honest negative or inconclusive result. The original Bernett test is used only after a future candidate is frozen. An independent holdout would be needed for a broad new generalization claim.
