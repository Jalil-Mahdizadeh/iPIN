**iPIN VY: independent monomer-MSA representations plus native PLM-interact**

8 October 2026. Brainstorm and recommended experiment on the **original released Bernett TRAIN/DEV**. This document proposes work; it does not initiate a pilot, change an existing experiment, or authorize TEST evaluation.

**A small VY experiment is scientifically justified.** The useful question is whether evolutionary context collected independently for each protein adds transferable information to native PLM-interact. The first experiment should use a frozen monomer encoder, reusable per-protein features, and a small symmetric classifier. Its strongest comparison should be against the same encoder given only the query sequence, augmented with inexpensive MSA profiles. A gain over PLM-interact alone would be encouraging but insufficient to establish the value of a learned MSA representation.

### Evidence motivating the idea

The completed studies provide a reason to investigate this direction, together with clear limits on the claim:

| Existing model, on the same 958 assessment pairs | AP |
|---|---:|
| Native PLM-interact | 0.640930 |
| Native + original global true paired-MSA features | 0.664361 |
| Native + original global pairing-shuffled features | 0.663248 |
| Native + original quality/profile control | 0.650942 |
| Native + Vx-v2 local true features | 0.655735 |
| Native + Vx-v2 local pairing-shuffled features | 0.662517 |

The original true-minus-shuffled gap was only +0.00111 AP, with a 95% interval of approximately [-0.00964, +0.01333]. V2 did not recover a pairing or locality benefit: its primary improvement in the pairing gap was -0.007896 [-0.023482, +0.007892]. Equal-alpha diagnostics also showed little true/shuffled separation. The short-pair result does not justify selecting proteins of combined length ≤512 for VY. These are findings from the [original review](pilot-vx/review-20261008/REVIEW.md) and [completed V2 review](pilot-vx-v2/review-20261008/REVIEW.md), not assumptions about a future model.

One plausible explanation is that the useful part of the MSA channel describes each protein's family, conserved functional constraints, or structural tendencies. Another is that the joint encoder infers compatibility from the two human query sequences and their marginal evolutionary context, even after homolog pairing is disrupted. Coverage, phylogeny, and family-related shortcuts remain alternative explanations.

**Shuffled paired-MSA features are not equivalent to two independently encoded monomers.** The shuffled experiment still processes both chains jointly, retains their human query pair, and retains coarse taxonomic relationships. Therefore, the Vx result motivates VY but does not demonstrate that VY will retain its gain. Likewise, failure of one V2 readout does not prove that all localized interface information is useless.

### What information VY would use

The proposed separation of responsibilities is:

```text
A sequence + A's own MSA -> shared frozen monomer encoder -> z_A --+
                                                                 +-> small symmetric evidence head --+
B sequence + B's own MSA -> shared frozen monomer encoder -> z_B --+                                  |
                                                                                                    +-> fused score
A sequence + B sequence  -> frozen native PLM-interact -> native logit -------------------------------+
```

Monomer context can describe conservation, tolerated substitutions, domain composition, and constraints within a protein. A supervised pair head can learn whether combinations of these properties predict the Bernett interaction label. This can provide partner-dependent predictions without paired homolog rows. It cannot measure cross-chain substitution covariance from a correspondence that was never supplied. A successful VY result would support **useful independent evolutionary context**, not direct inter-protein co-evolution.

Three hypotheses should remain separate:

| Hypothesis | Required comparison | Defensible interpretation |
|---|---|---|
| Useful augmentation | Native + VY versus native | The added branch improves this evaluation task. |
| MSA-specific information | VY versus matched query-only, profile, and query-plus-profile branches | Homolog-conditioned representations add information beyond the tested simpler controls. |
| Pair compatibility | Symmetric pair head versus an additive single-protein head | The tested pair terms add information beyond this unary propensity model. |

None establishes universal family generalization, physical binding, or a new interface-contact predictor. Native PLM-interact already contains a learned interaction model; VY need only provide complementary evidence.

### Representation choices worth considering

| Representation | Attraction | Main limitation | Priority |
|---|---|---|---|
| Per-position MSA frequencies, entropy, occupancy, and query agreement, summarized in fixed sequence blocks | Cheap, interpretable, and available from existing caches | Mostly marginal conservation and family information | Required control; potentially sufficient solution |
| Frozen Pairformer query-residue embeddings from a monomer MSA | Existing pinned source, weights, SIF, and parser; contextualizes the actual human query | Still performs dense computation within each monomer; downstream PPI utility unproven | Preferred first neural branch |
| Frozen MSA Transformer query-residue embeddings | Established alignment-conditioned representation with public weights | New runtime qualification, length policy, and exposure review; changing encoder adds another experimental axis | Alternative if Pairformer qualification fails, not a simultaneous search |
| Average ordinary PLM embeddings over independently selected homologs | Tests whether generic homolog aggregation is sufficient | Many sequence encodings; does not use alignment structure directly | Later alternative |
| Monomer intrachain pair-tensor summaries | May retain structural information absent from query embeddings | Reintroduces tensor aggregation choices and greater complexity | Defer |

MSA Transformer provides primary evidence that alignment-conditioned models learn useful protein representations; its published results do not establish this Bernett fusion hypothesis. The authors' implementation supplies an embedding-capable MSA model. [Rao et al., ICML 2021](https://proceedings.mlr.press/v139/rao21a.html), [official ESM repository](https://github.com/facebookresearch/esm).

The pinned Pairformer implementation exposes `final_msa_repr`, including a query-only view, with 464 channels. Its trunk can be called without the released contact heads. This makes the proposed extraction technically plausible; it still needs monomer and single-row numerical qualification. [Pinned author implementation](https://github.com/yoakiyama/MSA_Pairformer/blob/875363570df1ae484cf725aba382790444223005/MSA_Pairformer/model.py), [existing runtime](images/msa-pairformer/README.md).

MSA representations can also encode phylogenetic relationships. More homolog information is therefore not automatically more interaction-specific information. [Lupo et al., Nature Communications 2022](https://www.nature.com/articles/s41467-022-34032-y).

### Why feasibility is better than starting another paired-MSA pipeline

The existing cache is already organized by monomer, matches complete query sequences exactly, and contains the original exposure exclusions. Independent row selection no longer needs an intersection of A/B genome accessions. A representation can be reused whenever the same protein appears with another partner. A per-chain length cap also permits some pairs rejected by the old combined-length cap.

A read-only inventory of `pilot-vx/data/sample.json`, `proteins.json`, and `monomer-catalog.json` gives:

| Property of the existing pilot sample | TRAIN | DEV |
|---|---:|---:|
| Pairs | 4,000 | 4,000 |
| Distinct query proteins | 2,556 | 2,390 |
| Distinct query proteins with cached exact-query MSA | 2,468 | 2,339 |
| Pairs with both exact-query caches | 3,749 | 3,863 |
| Pairs passing preliminary independent-monomer checks | 2,486 / 62.15% | 2,959 / 73.975% |
| Actual evidence-eligible pairs in Vx | 1,825 / 45.625% | 2,628 / 65.70% |

The preliminary checks are: cache exists, original quality-mask fraction ≥0.5, at least eight cached homolog rows, and each complete chain ≤1,536 residues. They inspect catalog metadata, without fitting a model or extracting new features. They **do not** apply the proposed monomer redundancy filter, retained-row coverage checks, or numerical qualification. Thus these percentages are upper bounds under those additional restrictions, not measured VY coverage. Better aggregate upper bounds also do not imply that every Vx-eligible pair will be VY-eligible.

The sample contains 4,946 distinct proteins, of which 4,807 have cached alignments. Only 4,097 pass the preliminary per-protein checks. Even the neural extraction therefore concerns a few thousand reusable monomers, rather than a new full-dataset search. The archived 96.9% TRAIN / 98.4% DEV exact-query match rates came from a different 1,000-protein-per-split coverage audit and must not be substituted for usable pair coverage. [Archive audit](pilot-vx/results/archive-coverage.json).

### Recommended first pilot

The following is a concrete starting design, to be frozen after label-independent runtime qualification. It is one experiment, not a grid over encoders, layers, pooling methods, and lengths.

1. **Retain the original task and governance.** Reuse the exact 4,000 TRAIN and 4,000 DEV rows, original labels, native baseline, and internal DEV assignments: 1,068 calibration, 958 assessment, 1,974 crossing. All preprocessing learned from data uses TRAIN only. Crossing pairs appear in the fixed full-DEV report and never choose parameters. The assessment has already been inspected through Vx, so VY remains exploratory. No TEST data, TEST hash file, new negative sampling, or V5/V11 supervision is needed.

2. **Select homologs independently for each protein.** Reuse immutable cached monomers and their exposure filtering. Keep the human query, original coordinates, ambiguity handling, and quality mask. Select at most 127 homologs plus query, using deterministic taxonomic-family balancing and the established 90%-identity / 80%-comparison-coverage redundancy rule, applied to that monomer. Require ≥50% good-column coverage for each retained homolog, ≥50% good query positions, and at least eight retained homologs. Proposed cap: 1,536 residues per complete chain, with no cropping or encoder windows. Selection and cache identity depend on the protein and frozen protocol, never its partner, pair label, or DEV outcomes. Do not choose the rows of A separately for every B.

3. **Extract one fixed neural representation.** Use the existing frozen Pairformer trunk and precision contract. Feed each MSA as one chain. Use the final query-residue MSA representation, including the final MSA update, rather than searching layers on DEV. This differs deliberately from Vx's intermediate inter-chain contact-oriented tensor. Call the trunk directly: the upstream general forward enables contact heads by default. Keep those heads and the language-model output head out of the predictive feature path. Dense internal tensors are temporary; cache only compact monomer summaries.

4. **Keep a little sequence organization without learning a large readout.** A reasonable fixed summary is the masked global mean and standard deviation, plus masked means in four contiguous equal-coordinate blocks of the complete query. That is six summaries ×464 channels =2,784 numbers per protein. Partition original coordinates before masking; never join distant retained residues. Empty blocks have a fixed zero value and their coverage is recorded. This preserves coarse domain placement but does not localize a binding interface. It is a proposed pooling choice, not a conclusion from V2.

5. **Use a small symmetric classifier with genuine pair terms.** For each representation arm, fit its StandardScaler and PCA(16) using each distinct eligible TRAIN protein once, shared across A/B. This avoids weighting the representation fit by a protein's number of pairs. Let the resulting vectors be `z_A` and `z_B`, and construct:

   ```text
   phi(A,B) = concat((z_A + z_B)/2, abs(z_A - z_B), z_A * z_B)
   evidence(A,B) = w · StandardScaler_TRAIN(phi(A,B)) + b
   ```

   Use the existing regularized logistic-regression convention, `C=0.1`, fitted on eligible TRAIN pair labels. The head has 48 coefficients plus an intercept; the representation transforms are also fitted on TRAIN. The M, S, and P arms below share this recipe and dimensionality; S+P deliberately receives both control feature blocks. Linear concatenation alone would largely model additive protein propensities; the product and difference terms make the prediction depend on the combination. A diagonal product in a 16-dimensional space is restrictive and can miss complementary properties, which limits what a negative result would exclude.

6. **Fuse with the frozen native logit.** Preserve full-sequence native inference, mean AB/BA logits, FP32 parameters/BF16 autocast, and the checkpoint identity in [the Vx configuration](pilot-vx/config.json). Fit the evidence head directly to TRAIN labels, without native in-sample TRAIN logits or residual targets. Proposed reliability is:

   ```text
   g(A,B) = available(A) * available(B)
            * min(1, min(Neff80_A, Neff80_B)/32)
            * min(quality_fraction_A, quality_fraction_B)
   s_VY   = s_native + alpha * g(A,B) * evidence(A,B)
   ```

   Compute monomer Neff80 with the explicit gap-aware 80%-identity / 80%-comparison-coverage convention from the V2 diagnostics. It is an operational diversity proxy, not a count of independent evolutionary observations. All new arms, including query-only controls, use the **same gate derived from the real monomer MSAs**. Choose alpha separately on calibration only from `{0, 0.1, 0.25, 0.5, 1}`, with smallest-alpha tie breaking. On unavailable pairs, return the original native logit exactly; do not add an intercept or a one-sided MSA branch.

7. **Keep the study small and auditable.** Freeze code, input checksums, selected rows, masks, model revision, transformations, controls, and decision rules before new outcome inspection. Verify symmetric predictions, coordinate handling, deterministic caching, depth-one control inference, finite features, TRAIN-only fitting, and exact native fallback. A corrupt/missing computational artifact, OOM, unsupported format, or exhausted compute budget is an execution failure or explicit incomplete study; it is never reclassified as biological absence. Do not silently shorten proteins or lower depth to finish a failing case.

The proposed PCA is a capacity constraint, not proof that all useful information survives compression. Save the compact pre-PCA summaries and report retained variance. Avoid turning a weak result into an assessment-driven PCA, layer, or pooling search.

### Controls that make the experiment informative

The principal learned branch and four control heads would share eligible TRAIN pairs, gate, fitting recipe, and calibration rule. The original Vx arms remain frozen reference results.

| Arm | Input to the small head | What it addresses |
|---|---|---|
| **M: monomer MSA** | Query embeddings conditioned on each independently selected real MSA | Proposed VY model |
| **S: query only** | Same frozen encoder and pooling, with just the human query row and the same position mask | Benefit of adding a second sequence model |
| **P: profile/quality** | Fixed per-protein quality, depth, Neff, gap/occupancy and entropy summaries; amino-acid frequencies and query agreement globally and in the same four blocks | Information available without a learned MSA encoder |
| **S+P: query plus profile** | Concatenate S's and P's 48-dimensional pair features, preserving both independently fitted TRAIN-only transformations; fit one 96-coefficient head plus intercept | A stronger control combining sequence and MSA marginals |
| **U: unary MSA control** | The M arm's 16-dimensional vectors, but only `(z_A+z_B)/2` | Whether the observed gain requires the chosen pair terms |

Specify the P descriptor exactly before fitting; ambiguous residues, gaps, empty blocks, and occupancy denominators need explicit definitions. Use the selected real homolog rows for P so its context depth agrees with M. This control is richer than Vx's 68 global quality/profile features, which were not learned monomer embeddings and did not retain this block organization. Keep both S and P feature blocks in the combined control: compressing their concatenated raw descriptors together could suppress the smaller profile block and create a weak comparator. S+P has more supervised coefficients than M, intentionally making it a demanding alternative, while remaining a tiny classifier. U is a nested lower-capacity diagnostic, not a capacity-matched causal null.

The **primary scientific contrast should be AP(native+M) minus AP(native+S+P)** on the full fixed assessment population, including exact fallback. Also compare M with S, P, U, native, and the unchanged Vx global true/shuffled/quality predictions. Report M alone as a descriptive diagnostic of complementarity, without fitting another model. Count the original Vx comparisons as historical references: their training eligibility, gate, and head dimensionality differ from VY.

Because V2 exposed the effect of separate fusion weights, predeclare two additional diagnostics: the M-trained head evaluated on S features using M's transformations, and common-alpha M/S results over the original grid. Neither selects a new alpha or replaces the primary comparison. Query-only input changes depth and may be outside the encoder's typical pretraining inputs; it is a useful operational control, not a perfect causal intervention. Interpret it alongside S+P.

**A cross-protein row-pairing shuffle has no biological target in VY.** There is no A-row/B-row correspondence entering two independent encoders. Reordering complete homolog rows preserves within-protein column relationships; it is an invariance/sensitivity check, not a destroyed-evolution control. Keep Vx's true/shuffled results for continuity, but do not invent a VY pairing claim from a meaningless permutation.

A later mechanistic control could independently permute non-query amino acids within each MSA column while preserving the query, column frequencies, and gap positions. This disrupts within-protein cross-column relationships. However, it also creates artificial homolog sequences and disturbs phylogeny. A loss under that intervention would not uniquely identify intramolecular co-evolution. It adds another encoder pass per monomer and is not necessary for the first utility experiment.

### Coverage and shortcut diagnostics

VY can improve aggregate AP because its representation is better, because it covers more pairs, or both. Report the full assessment first, then the four fixed availability groups: eligible for both Vx and VY, VY only, Vx only, neither. Within the overlap, compare saved scores on identical rows. These are descriptive decompositions; AP differences are not additive across groups, and the models were fitted on different eligible TRAIN populations. A wider-coverage result alone does not establish a better representation on common support.

Every new arm shares VY availability and reliability. Consequently, M versus S+P tests the representation on a common coverage policy. Do not compare covered-only M against full-population native, and do not let a query-only arm score extra rows in the principal comparison.

Record and report:

- Monomer raw, cached, filtered, and selected depths; Neff80 separately for A/B; cap saturation, comparable-row fraction, and taxonomic diversity. No meaningful paired Neff is implied by these independent inputs.
- Original quality-mask coverage, per-column nongap occupancy, usable residue count, and ambiguous-residue burden. Distinguish poor alignment support from a missing cache and from a computational error.
- Chain lengths, combined length, length imbalance, and the existing ≤512 / 513–1024 / 1025–1536 / >1536 combined-length strata. No short-only optimization.
- Query protein family/domain relationships to TRAIN, annotation completeness, and influence of frequent proteins/families. Reuse exclusively TRAIN/DEV annotations or obtain an explicitly scoped annotation extract. The existing partial family witnesses do not establish family novelty when absent.
- Label prevalence and class counts within every reported group, plus AP, AUROC, and common-prevalence AP. Use cutpoints fixed from TRAIN; small or one-class strata are unresolved, not opportunities to redefine bins.
- Calibration and correction magnitude by availability/quality. A strong P or U result suggests that a sophisticated representation may be unnecessary or that marginal family/coverage information explains much of the gain.

Independent monomer encoding removes the need to infer ortholog partner assignments; it does not eliminate mistaken homolog membership, phylogenetic bias, domain-family transfer, or supervised family shortcuts. Original Bernett's sequence separation must not be described as universal family separation. An additive control can expose a particular shortcut but cannot rule out all family-pair memorization.

### What would justify continuing

Use matched protein-level bootstrap resamples across models, following the existing 1,000-replicate endpoint-multiplier approach. Report intervals conditional on the frozen fitted models; these do not include training-set/model-selection variability. Record multiplicity and keep subgroup analyses exploratory. A useful first result would meet all of the following:

- M improves full-assessment AP over native by at least 0.010, with a positive 95% interval, and does not reduce AUROC by more than 0.005.
- The primary M-minus-S+P AP interval is above zero; M also improves over S and P. Otherwise there is no established reason to pay for the neural MSA branch.
- The result is not solely an availability effect or dominated by one frequent protein/family. Missing family annotations remain a limitation rather than a passed robustness check.
- Extraction completes with verified artifacts, adequate class counts, and a measured cost compatible with caching monomers at the intended scale.

Outperformance of U would support the tested pair terms, but lack of that separation should narrow the interpretation to useful monomer propensity/context. It should not be mislabeled as recovered interface compatibility. A statistically uncertain difference is inconclusive; a sufficiently precise nonpositive primary difference argues against this configuration. Both stop automatic expansion on the same assessment.

If S+P matches M and improves native, pursue the simpler representation if a subsequent confirmation is warranted. If M improves native but does not match the existing Vx gain, report that cost/accuracy tradeoff honestly. Conversely, retaining Vx performance at lower measured cost could be useful even without exceeding it; a formal noninferiority claim needs a declared margin and interval, not similar point estimates. None of these development outcomes authorizes TEST or creates a new blind validation set.

### Ideas to keep available without implementing them together

| Idea | Hypothesis and potential use | Why defer it |
|---|---|---|
| Evolutionary residual `H_MSA − H_query`, computed in the same encoder coordinates | Concentrate a branch on changes induced by homolog context | Subtraction need not isolate beneficial information and can amplify noise; compare only in a separately frozen follow-up |
| A few monomer block descriptors with a small symmetric bilinear matcher and fixed top-k pooling | Retain partner-dependent localized compatibility without an inter-chain encoder | Needs safeguards against length/extreme-value effects; similarity is not automatically physical complementarity |
| Low-rank symmetric bilinear head `z_A^T W z_B` | Capture complementary latent properties missed by coordinatewise products | Extra capacity and optimization would confound the first representation test |
| Fusion with PLM-interact's frozen pair representation instead of only its logit | Let the classifier condition the added evidence on the native model's pair context | Requires new pair-activation extraction, stronger overfitting controls, and possibly honest out-of-fold baseline training predictions |
| Joint fine-tuning, learned cross-attention, or full residue-residue maps | Learn how sequence and evolutionary features interact at higher capacity | Becomes a larger model-training project before the information source is established |

Per-protein embeddings combined through a pair readout are an established design family; D-SCRIPT is one relevant sequence-embedding precedent, with a substantially different architecture and supervision. VY should not claim novelty merely from feature fusion or import another PPI-trained checkpoint without an exposure assessment. Its potentially useful contribution is the controlled demonstration of independent MSA context, its limits, and its measured cost on this task. [Sledzieski et al., Cell Systems 2021](https://doi.org/10.1016/j.cels.2021.08.010).

### Work and compute expectations

This is a moderate engineering task with very small supervised fitting. The nontrivial work is a monomer extraction/cache contract, fair controls, and reliable population accounting. The existing parser, SIF, frozen weights, native predictions, split assignments, and verification machinery substantially reduce setup. A new SIF is not presently justified.

| Work item | Expected effort |
|---|---|
| Inventory, deterministic independent row selection, masks, coverage audit | Small CPU task; cache inventory above already completed |
| Frozen monomer and query-only extraction, compact cache, numerical qualification | Main implementation and GPU work |
| Profile control, TRAIN-only transforms, five small classifiers, fusion | Small CPU fitting task; control design needs care |
| Matched uncertainty, strata, independent verification, report | Material review work, reusable from Vx/V2 |

A rough engineering estimate is **2–4 focused developer days**, excluding scheduler wait and unexpected runtime defects. This is a planning judgment, not a measured delivery time. GPU time should be estimated from label-independent timing at representative monomer lengths and depths before requesting an allocation.

The workload is approximately `sum_over_unique_proteins[t_MSA(L,D) + t_query(L,1)]`, plus cheap pair-head inference. The previous paired branch used four passes per eligible pair: true/shuffled ×AB/BA. Nevertheless, monomer extraction still has dense intrachain pair states and triangle updates, the proposed final representation uses the full stack, and VY admits some new longer examples. It is not defensible to promise a particular speedup from the number of proteins alone. Profile peak memory and allocated GPU-hours, including failures and idle reserved devices. Prefer the smallest allocation meeting the measured deadline.

At 4,097 proteins, two 2,784-dimensional FP32 summary views occupy about 91 MB before metadata, using decimal MB; raw MSAs already exist. This storage estimate excludes temporary tensors and optional residue-level caches. Persisting all residue or pair tensors is unnecessary for the proposed pilot.

Vx/V2 consumed 21.9611 of their 24 allocated GPU-hours. Their remaining balance is not a VY budget. Any implemented VY pilot needs a separate frozen resource ledger and ceiling, and no automatic requeue, parameter sweep, production continuation, or TEST stage. [Completed accounting](pilot-vx-v2/results/resources.json).

The recommended next action is to qualify **one frozen monomer-MSA branch with the S, P, S+P, and U controls**. The evidence justifies testing whether the source of information is useful and economical; it does not yet justify changing PLM-interact's architecture or training a larger interaction model.
