# iPIN v6: retraining native ESM2 on STRING v12.5

Prepared on 5 October 2026. This is a research and implementation proposal. No new training job, production dataset, clustering run, or test inference was launched for this review. The earlier paired-MSA proposal remains unchanged in [improvment-proposal-vx.md](improvment-proposal-vx.md).

**Recommendation: carry out one ESM2-650M retraining run using the native iPIN/PLM-interact architecture and a documented STRING v12.5 counterpart of the humanV11 dataset.** Retain the historical 50–800-residue range, approximately 1:10 positive-to-negative sampling, and interaction-pair redundancy reduction. Use direct human experimental evidence for positive eligibility. Correct duplicate, contradictory-label, and exact test-overlap problems before training. Keep the existing v5 models as separate HIPPIE/ILP models.

This is a reasonable attempt to improve performance on the five-species STRING-derived benchmark. It is **not yet evidence that a newer STRING release will outperform humanV11**, nor that matching its negative distribution will improve performance on the original Bernett and ILP tests. The investigation below makes that distinction particularly important.

**The most consequential findings are these:**

- HumanV11 and TUnA human seed 47 use the **same released human TRAIN and validation sequence-pair/label multisets**. I independently rechecked them against the D-SCRIPT files, preserving row multiplicity and treating AB/BA as the same interaction.
- Their negative pool is substantially broader than the proteins appearing in positive interactions. The training file contains 15,631 distinct sequences, but only 7,492 participate in training positives. **77.44% of training negative rows contain at least one sequence absent from the training-positive graph.** Sampling only from positive endpoints would not reproduce this important property.
- Every sequence in their human validation set appears somewhere in TRAIN. A simple TRAIN-only protein-degree score reaches **AP 0.8360 / AUROC 0.9635** on that validation set, without a PLM. This demonstrates a strong protein-level sampling signal; it does not measure that score's transfer to another species.
- The historical 40% rule concerns redundancy between interaction pairs. It must not be described as proof that every test protein is below 40% identity to every training protein.
- The latest official release is **STRING v12.5**, current since 29 September 2026. Its human direct-experimental, 50–800-residue candidate pool contains **296,306 distinct unordered sequence pairs before clustering, test protection, and other cleanup**. That number is not a final training-set size. [STRING version history](https://version-12-5.string-db.org/cgi/access)
- Native PLM-interact's public trainer uses **positive-class weight 10** in BCE. This is separate from its **classification-loss weight 10 relative to MLM weight 1**. Reusing v5's positive-class weight 1 unchanged would miss a relevant part of the native imbalanced-data training setup.

The supporting source snapshots, hashes, scripts, and numerical audits are in [literature/string-v12.5-review-v6](literature/string-v12.5-review-v6/).

**1. What was actually used by humanV11 and TUnA seed 47.**

The evidence chain is STRING v11 → the D-SCRIPT cross-species collection → TUnA's released human files and PLM-interact's sequence CSVs. TUnA's preprocessing script reads existing D-SCRIPT pair files, shuffles their rows, resolves sequences, and computes ESM2 embeddings. It does not construct a new STRING dataset or regenerate negatives. Its preprocessing seed 47 controls the row shuffle; the seed-47 training configuration also controls model randomness. It is not the provenance seed of an independently sampled negative set. [Pinned TUnA preprocessing](https://github.com/Wang-lab-UCSD/TUnA/blob/b5bda8fee261a4f27821738db995cf5883dcd133/process_xspecies.py), [pinned data utilities](https://github.com/Wang-lab-UCSD/TUnA/blob/b5bda8fee261a4f27821738db995cf5883dcd133/data_processing/data_utils.py)

The current artifacts agree exactly after conversion to unordered sequence-hash pairs and labels:

| Released human split | Positive rows | Negative rows | Total rows | Role in the two releases |
|---|---:|---:|---:|---|
| TRAIN | 38,344 | 383,448 | 421,792 | Supervised training |
| File named `human_test` | 4,794 | 47,931 | 52,725 | Human validation |
| Combined | 43,138 | 431,379 | 474,517 | Public TRAIN + validation evidence |

The ratio is approximately 1:10, with small departures in the released row counts. The train-to-validation ratio is approximately **8:1** among the files actually used here. TUnA's configuration explicitly supplies `human_test_interaction.tsv` as `validation_interactions`. It should not be mistaken for an untouched human test set. [TUnA seed-47 configuration](https://github.com/Wang-lab-UCSD/TUnA/blob/b5bda8fee261a4f27821738db995cf5883dcd133/results/xspecies/TUnA_seed47/config.yaml)

The D-SCRIPT paper describes a larger 80:20 human split. Repository history partly resolves the discrepancy: its August 2021 `human_test.tsv` contains 105,450 rows, while the February 2022 revision contains 52,725. The former has 9,586 positive rows, also slightly different from the paper's stated 9,587. I did not find a documented explanation sufficient to reconstruct a missing third partition. **Do not invent an 80:10:10 split or silently substitute the older validation file for the actual humanV11/TUnA source.** The file change and counts are recorded in [the validation-history audit](literature/string-v12.5-review-v6/dscript-validation-history.json) and [upstream commit history](https://github.com/samsledje/D-SCRIPT/commits/main/data/pairs/human_test.tsv).

The principal pinned identities are:

| Source | Revision or identifying evidence |
|---|---|
| D-SCRIPT repository examined | `23cbb0fbbb454d09ee41ff749c1454d8b3c8a4b8` |
| TUnA repository examined | `b5bda8fee261a4f27821738db995cf5883dcd133` |
| Native PLM-interact repository examined | `ab2ae6ae1aa81accf4c1f7f6f341164baf97809a` |
| Native cross-species dataset | `danliu1226/cross_species_benchmarking`, revision `19dbed25184466fda5341b5ba70c2560ade1bf85` |
| Native humanV11 model | `danliu1226/PLM-interact-650M-humanV11`, revision `e86e392dec13dd0c23252c94947b04a7a9821b0e` |
| TUnA human seed 47 | `yk0/TUnA_models`, revision `eaec69cafc574d984119079220e75b11ffcb7c09`, `x-species/TUnA_seed47/model` |

The model identities come from the existing benchmark's provenance. The new audit verifies the released datasets, not an unavailable private history of every checkpoint's training. Exact file hashes and equality checks are in [legacy-source-audit.json](literature/string-v12.5-review-v6/legacy-source-audit.json); the earlier independent overlap audit is [human-releases-exposure.json](benchmark-v5/provenance/human-releases-exposure.json).

**2. Reconstructing the historical preparation without overstating what is known.**

The original D-SCRIPT Methods describe STRING v11 binding interactions with a positive experimental-evidence score, proteins between 50 and 800 residues, and CD-HIT clustering at 40% similarity. A positive pair A–B is redundant with C–D when A/C and B/D share their respective clusters. Random protein pairs provide ten negatives per positive. This is interaction-pair redundancy reduction, not necessarily removal of an entire protein family. The paper does not establish a separate 700 or 900 score threshold. [D-SCRIPT paper](https://doi.org/10.1016/j.cels.2021.08.010), [archived Methods text](literature/string-v12.5-review-v6/dscript-paper.txt)

Several distinctions matter when implementing a successor:

| Question | Evidence and implication |
|---|---|
| Are all STRING functional associations eligible positives? | No. The historical account specifies experimentally supported binding interactions. Coexpression or text-mining association alone is not equivalent. |
| Does “40% clustering” mean retain one protein per cluster? | No. The described exclusion unit is a **pair of clusters**. Different interactions involving the same protein can survive. |
| Does it guarantee protein-disjoint TRAIN/DEV? | No. The actual files have complete validation-sequence overlap with TRAIN. |
| Does it guarantee no human/nonhuman homologs? | No. Neither the pair rule nor the released data establishes this. |
| Were the negative pairs experimentally verified non-interactions? | No. They are sampled background pairs; some can be unreported interactions. |
| Did TUnA apply a new negative sampler? | Its released cross-species preprocessing does not. |
| Can we reconstruct the exact raw STRING query and every CD-HIT parameter? | Not from the public construction information inspected. The original raw preparation program, exact coverage settings, representative tie-breaking, and original negative-sampling seed were not located. |

TUnA's paper describes nonhuman proteins with high similarity to human training proteins being removed. That wording is stronger than the original interaction-pair rule. The public files and our existing exact-overlap audit do not support treating this collection as universally protein/homology-disjoint. A new methods description should state the actual algorithm and measured overlaps instead of inheriting that assertion. [TUnA paper](https://doi.org/10.1093/bib/bbae359), [archived TUnA text](literature/string-v12.5-review-v6/tuna-paper.txt), [existing nonhuman exposure report](bechmark-v5-nonhuman/REPORT.md)

The negative universe deserves particular attention. In the released human sequence FASTA, 70,529 identifiers resolve to **15,631 distinct sequences**, all within the length range. The union of TRAIN and validation contains that same distinct-sequence universe; only 7,668 sequences occur in any positive row. This is strong evidence for a broad background-protein pool, not sampling confined to the positive graph. It does not prove the original sampler was uniform over sequence identities rather than identifiers. A new implementation must state its sampling unit explicitly. [Protein-universe audit](literature/string-v12.5-review-v6/legacy-protein-universe-audit.json)

**3. Defects and shortcuts present in the released human files.**

These are new measurements on the exact sequences used by both human releases. They do not rely on protein-name matching.

| Property | TRAIN | Validation | Combined |
|---|---:|---:|---:|
| Distinct sequences | 15,631 | 15,351 | 15,631 |
| Sequences in positive rows | 7,492 | 3,356 | 7,668 |
| Additional duplicate positive rows after unordered sequence-pair collapse | 30 | 0 | 37 |
| Additional duplicate negative rows after the same collapse | 356 | 3 | 434 |
| Distinct sequence pairs carrying both labels | 47 | 0 | 54 |
| Positive equal-sequence rows | 1 | 0 | 1 |
| Negative equal-sequence rows | 16 | 1 | 17 |

Combined duplicate counts also capture repetitions across splits. There are **89 distinct unordered sequence pairs shared between TRAIN and validation**: 82 validation rows have a same-label training match and seven have an opposite-label training match. All 52,725 validation rows have both sequence endpoints represented somewhere in TRAIN. These are observed defects to correct, not features to reproduce. [Detailed source audit](literature/string-v12.5-review-v6/legacy-source-audit.json)

To quantify the protein-level signal, I calculated the following score using only the released TRAIN rows:

\[
s(A,B)=\log\frac{d^+_{\mathrm{train}}(A)+1}{d^-_{\mathrm{train}}(A)+1}
       +\log\frac{d^+_{\mathrm{train}}(B)+1}{d^-_{\mathrm{train}}(B)+1}.
\]

Here degree counts interaction-row occurrences; an equal-sequence pair contributes once to its endpoint. There is no fitting, parameter search, PLM, target-species information, or validation-label input to the score.

| Human validation diagnostic | Rows | AP | AUROC |
|---|---:|---:|---:|
| TRAIN degree-ratio score, all released validation rows | 52,725 | 0.8360 | 0.9635 |
| Same score, excluding exact TRAIN-pair overlaps | 52,636 | 0.8361 | 0.9635 |
| Count of endpoints that appear in TRAIN positives, all validation rows | 52,725 | 0.2941 | 0.8739 |

The random-ranking AP reference is approximately 0.0909. These results show that this human validation task can be solved surprisingly well through endpoint sampling patterns. They do **not** show that a human identity lookup achieves these scores on mouse, fly, worm, yeast, or E. coli. Transfer of conserved protein properties is a plausible explanation for part of the cross-species advantage, but this experiment does not isolate it. An independent aggregation from the native sequence CSVs reproduced the degree-ratio metrics to within 1e-12. [Diagnostic script](literature/string-v12.5-review-v6/audit_degree_baseline.py), [complete results](literature/string-v12.5-review-v6/legacy-degree-baseline.json), [independent verification](literature/string-v12.5-review-v6/independent-degree-verification.json)

Consequently, a very high v6 human DEV score will require careful interpretation. It can demonstrate learning of the intended dataset while providing limited evidence of partner-specific recognition. Include this cheap TRAIN-degree diagnostic in the future DEV report; it does not require another neural-model run.

**4. HumanV11 is different from the paper's additional humanV12 model.**

The PLM-interact paper also describes a separate STRING v12 preparation: physical links, positive experimental scores, exclusion of positive homology scores, combined confidence at least 400, combined protein length at most 2,101, and MMseqs2 clustering at 40%. It reports 60,308 training positives and 15,124 testing positives, again with approximately ten negatives per positive. Those are not the humanV11/TUnA numbers above. [PLM-interact Methods](https://www.nature.com/articles/s41467-025-64512-w), [local article text](literature/plm-interact/article.txt)

In particular, a 400 confidence cutoff and a 2,101 combined-length rule should not be retrospectively presented as established humanV11 settings. A new model using them would be a humanV12-style successor. The recommended v6 below instead retains the requested humanV11-like length and sampling design, while defining the modern evidence filter precisely.

The paper, supplementary information, and author response were reviewed alongside code. The peer-review response acknowledges random unreported pairs and argues that sparsity limits contamination. That is an argument about false-negative frequency, not proof of unbiased negatives. Our degree diagnostic addresses a different failure mode. The statement that a low error rate cannot harm training also needs qualification: systematic label errors or sampling biases can matter even when relatively uncommon. [Author response](literature/plm-interact/peer-review.txt), [supplementary information](literature/plm-interact/supplementary-information.txt)

**5. What the latest STRING files actually contain.**

I downloaded only human-specific physical/full and sequence files for v12.5 and v12.0 for a bounded source comparison. No all-organism network download was needed. The v12.5 compressed files are approximately 9.62 MB and 6.76 MB respectively. The exact names and content hashes are recorded in [string-downloads.json](literature/string-v12.5-review-v6/string-downloads.json).

Use these versioned resources for the proposed preparation:

- [Human STRING v12.5 physical links with full evidence channels](https://stringdb-downloads.org/download/protein.physical.links.full.v12.5/9606.protein.physical.links.full.v12.5.txt.gz)
- [Human STRING v12.5 protein sequences](https://stringdb-downloads.org/download/protein.sequences.v12.5/9606.protein.sequences.v12.5.fa.gz)
- [Official download documentation](https://string-db.org/cgi/download)

The verified physical/full header is:

```text
protein1 protein2 homology experiments experiments_transferred database database_transferred textmining textmining_transferred combined_score
```

Parse by column name. The official FAQ's example using column 10 concerns the **functional/full** file; in this **physical/full** file, column 10 is `combined_score`, and direct `experiments` is column 4. Copying the FAQ's positional extraction to this file would select the wrong evidence. The simplified physical-links file lacks the channel information required for our filter. [STRING FAQ](https://version-12-5.string-db.org/cgi/info)

Direct experimental support and transferred experimental support are separate columns. Require `experiments > 0` for positive eligibility. Do not accept a transferred-only row as a directly observed human positive. Conversely, a row with direct evidence does not need to be discarded merely because it also has transferred support. The `homology` column is not a replacement for inspecting `experiments_transferred`.

STRING's physical network includes direct binding and co-complex proximity; it is not exclusively a database of binary binding experiments. Its scores represent confidence rather than binding affinity. Thus modern physical links plus direct experimental support are a defensible operational counterpart of the old binding collection, **not a demonstrated exact reconstruction of its original binding-mode selector**. [STRING network/evidence definitions](https://string-db.org/help/scores/)

The following numbers were calculated locally, collapsing AB/BA and checking that their evidence fields agree:

| Human source measure | STRING v12.0 | STRING v12.5 |
|---|---:|---:|
| Protein identifiers in the sequence file | 19,699 | 19,699 |
| Distinct physical identifier pairs, all channels | 738,805 | 648,245 |
| Physical identifier pairs with direct `experiments > 0` | 473,579 | 514,608 |
| Above, with both lengths 50–800 | 284,868 | 303,343 |
| Above, collapsed to distinct unordered sequence pairs | 279,899 | 296,306 |
| Distinct sequences in that eligible positive pool | 13,937 | 13,993 |
| Eligible identifier pairs also meeting combined score ≥400 | 73,854 | 209,349 |
| Eligible identifier pairs with transferred experimental support as well | 31,438 | 21,519 |

All these candidate counts precede 40% redundancy reduction, protected-test exclusions, sequence-quality checks, and splitting. The eligible v12.5 pool includes **46 equal-sequence pairs** despite having no self-identifier edges; aliases/paralogs make these different questions. The final recipe below handles them explicitly. [Reproducible STRING audit](literature/string-v12.5-review-v6/string-source-audit.json)

The human sequence gzip files for v12.0 and v12.5 have identical SHA-256 hashes. Eligible sequence-pair count rises by about **5.86%**, while the identifier-pair comparison contains 75,530 additions and 57,055 removals. Direct experimental scores change for 212,421 shared eligible identifier pairs. The latest release is therefore neither a simple append nor merely a larger proteome. The large change in the number passing 400 also cautions against interpreting equal numerical confidence cutoffs across releases as equal corpus selection. These are source observations; they do not establish which release's scores are better calibrated.

The 296,306 pre-clustering candidates must not be compared directly with the historical 38,344 training-positive rows to claim a particular dataset expansion factor. The processing stages and split denominators differ.

The modern 50–800-residue background universe contains **15,713 distinct sequences**; 13,993 already occur in the raw eligible positive pool, leaving 1,720 without such positive evidence before clustering. Thus keeping the same broad-background sampler does not guarantee reproducing the historical degree bias. Redundancy reduction will change these counts again. Report the final positive-endpoint coverage rather than deliberately discarding evidence to recreate the old shortcut. [Modern background-universe audit](literature/string-v12.5-review-v6/string-background-universe.json)

**6. Why one STRING retraining is justified, and what success would mean.**

The strongest local evidence for reconsidering the training distribution is the comparison between different releases within the same model family. Selected AP values from our completed, frozen benchmarks are:

| Model | Mouse AP | E. coli AP | Original human test AP | Human ILP-test AP |
|---|---:|---:|---:|---:|
| iPIN v5 ESM2, HIPPIE/ILP | 0.2091 | 0.0807 | 0.6917 | 0.6570 |
| Native PLM-interact humanV11 | 0.9084 | 0.7329 | 0.6477 | 0.5863 |
| Native PLM-interact Bernett | 0.2751 | 0.1572 | 0.6903 | 0.6377 |
| TUnA human seed 47 | 0.8832 | 0.6730 | 0.6578 | 0.5922 |

These columns represent different tasks and prevalences; compare models within a column, not AP values across columns. The human STRING releases also have known exposure in the human benchmark, so these human columns are descriptive results, not clean unseen-protein generalization estimates. [Nonhuman report](bechmark-v5-nonhuman/REPORT.md), [human benchmark report](benchmark-v5/REPORT.md)

My interpretation is that training-distribution compatibility is a much better-supported next direction than another speculative attention or readout change. However, these observations confound source coverage, positive definitions, protein lengths, negative sampling, homology, optimization, and checkpoint selection. They do not identify random negatives alone as the cause. Nor does a 1:10 class ratio alone explain a ranking gap; a constant score-prior adjustment cannot change AUROC or AP ranking.

The proposed run tests whether the native ESM2 model trained on a current, explicitly prepared STRING corpus can recover or improve the useful cross-species behavior of the STRING-trained releases. If it improves only those historical random-negative tests and loses performance on ILP negatives, the defensible conclusion is **task specialization**, not universal superiority. Retain v5 for its original evaluation setting.

**7. Recommended preparation for one production dataset.**

The default decisions below are concrete. They avoid an open-ended grid of negative samplers, thresholds, or model variants. Items described as changes are intentional departures from defects or ambiguities in the historical files.

| Decision | Recommended v6 choice | Relation to historical preparation |
|---|---|---|
| Source | Human, taxon 9606, pinned STRING v12.5 physical/full file | Updated release with inspectable evidence channels |
| Positive evidence | Physical row with direct `experiments > 0` | Closest documented modern operational counterpart; not proven identical binding semantics |
| Extra combined-score threshold | None beyond inclusion in the published file | Do not import the separate humanV12 ≥400 rule silently |
| Individual protein lengths | Inclusive 50–800 residues | Matches released humanV11/TUnA data |
| Maximum concatenated input | 1,603 tokens including CLS/EOS/EOS | Both chains fit fully; no truncation |
| Sequence identity unit | Exact normalized amino-acid sequence, with all identifiers retained as aliases | Prevent alias multiplication of pairs and sampling weights |
| Positive redundancy | One physical positive pair per unordered pair of 40% protein clusters | Retain interaction-pair logic, not one-protein-per-family deletion |
| Negative universe | All eligible distinct human sequences, including proteins absent from retained positives | Matches the observed broad background pool |
| Negative sampler | Uniform unordered sequence pairs without replacement, excluding forbidden pairs | Static sampled background negatives; no ILP or degree balancing |
| Class ratio | Exactly 10 negatives per positive within each split | Preserves intended historical ratio |
| TRAIN:DEV target | 8:1 by cluster-pair group | Approximately matches actual released files; realized counts reported |
| TRAIN/DEV separation | No exact pair or cluster-pair group shared | Stronger and explicitly defined cleanup; proteins may occur in both |
| Equal-sequence pairs | Exclude from both classes in the new training corpus | Avoid asymmetric self-pair handling; acknowledge removed positive biology |
| Data seed / model seed | 47 for deterministic preparation; 2 for the single neural run | Explicit new choices, not a claim to recover the old sampling RNG |

The “no additional threshold” choice does not mean every nonzero experimental score is strong evidence. Preserve the source scores and report their distributions. The goal here is a clearly specified humanV11-style experiment; a separate ≥400 model is not part of the initial proposal.

Implementation order matters:

1. **Freeze input identities.** Record download URLs, byte sizes, SHA-256 hashes, release date, headers, tool versions, and all source counts. Keep the full raw human physical edge inventory before length filtering, clustering, or train/dev splitting.
2. **Resolve sequences and aliases.** Uppercase and remove FASTA formatting whitespace. Preserve amino-acid symbols supported by the ESM2 tokenizer, including supported ambiguous symbols. Do not silently delete residues, replace an unrecognized symbol with another amino acid, select an unrelated isoform, or crop a chain. Log unsupported or missing sequences. Use sequence hashes for pair identity while keeping original STRING identifiers and evidence.
3. **Apply the fixed length and test-protection rules described below.** Build the background universe from eligible human sequences, not just endpoints of the surviving positive graph. Exclude equal-sequence pairs from both classes; the present raw candidate audit identifies 46 such positive sequence pairs before other filters. This limits the training target to distinguishable sequence pairs and should be recorded as a coverage limitation.
4. **Cluster the eligible human sequence universe at 40%.** Use a pinned CD-HIT executable to keep the historical tool family. Record the complete command and coverage settings. A concrete proposed starting command uses `-c 0.4 -n 2 -G 1 -g 1 -d 0`; CPU/memory flags are execution settings. The original full command is unknown, so this is a declared implementation choice, not claimed bitwise historical equivalence. Clustering uses human sequences only, never nonhuman labels or a joint cross-species clustering purge.
5. **Collapse positive interaction redundancy.** Define a cluster-pair key as the sorted two cluster identifiers. Keep one actual positive sequence pair per key, choosing the highest direct experimental score and a stable sequence-hash tie-break. Do not substitute cluster-representative sequences for experimentally observed partners. Do not automatically exclude interactions within one cluster: different homologous proteins can interact.
6. **Assign cluster-pair groups to TRAIN/DEV with a seeded stable 8:1 assignment.** A key's partition is fixed for either label. This keeps identical cluster-pair contexts from straddling TRAIN and DEV, while allowing shared proteins. Report the actual positive split counts; do not promise exact 8:1 counts from an indivisible grouping. Greedy cluster separation is not proof that every cross-cluster alignment falls below 40% identity.
7. **Sample ten distinct negatives per positive for each partition.** Draw uniformly from distinct sequences in the fixed broad universe. Canonicalize AB/BA, apply the same cluster-key split assignment, reject equal-sequence pairs, reject every known physical sequence pair in the raw inventory, reject all protected test pairs, and reject duplicates. Multiple negative pairs may occupy one cluster-pair group, but that group stays in one partition. Do not refresh negatives each epoch. Do not use a test score to modify this rule.
8. **Freeze manifests and export token arrays.** Emit exact split sizes, sequence coverage, length and degree distributions, score distributions, alias mappings, rejected-row reasons, positive-pair retention after each stage, and a negative-sampling seed/hash record. Require zero exact pair contradictions, zero TRAIN/DEV pair overlap, zero cross-split cluster-pair overlap, and exactly 1:10 classes before authorizing training.

The negative exclusion inventory deliberately includes **all published human physical links**, including pairs dropped by length, redundancy, or confidence choices. Otherwise a known positive that was discarded during dataset reduction could accidentally be relabeled negative. Mapping this inventory through sequence aliases is essential.

This conservative inventory includes some human STRING links supported by transferred evidence. They are used only to avoid labeling known/inferred physical links negative; transferred-only links do not become positive training examples. Therefore “direct-human-evidence positive set” is accurate, while “no transferred information affects any construction decision” would be inaccurate. No target test labels should be consulted to decide which background pairs are biologically negative.

Broad random background sampling remains biased and its negatives remain unverified. We are intentionally testing this distribution because it matches the comparator setting. Neither one-per-cluster-pair positives nor duplicate cleanup balances protein degree. Report that fact instead of calling the new dataset bias-free.

**8. Protect the existing benchmarks without silently turning this into v5 again.**

For the **five-species evaluation**, remove human candidate sequences exactly identical to any sequence in those frozen tests before constructing TRAIN or DEV. This is a label-independent exclusion and a deliberate improvement over the historical exact-overlap situation. Keep homologs below exact identity: blanket removal of every ≥40%-identity human homolog would change the intended conserved cross-species transfer question. Record both exact overlap and sequence-similarity strata in the eventual report. Exact exclusion does not establish homology-disjoint evaluation.

For the **original Bernett and ILP human tests**, block every exact unordered sequence pair from both tests from entering either TRAIN or DEV, irrespective of its label. This removes direct pair memorization and train/test label contradictions. I recommend retaining other interactions involving the same human proteins for this STRING transfer experiment. Consequently, these human tests would **not** be unseen-protein evaluations of v6, and results would not have the same protection claim as v5.

This is an important scope decision, not a technicality. If the scientific requirement is instead to preserve v5's exclusion of all human test proteins and their ≥40% homologs, that requirement must be applied before construction and the retained STRING corpus counted again. It defines a stricter, materially different experiment. It is not an additional default run in this proposal.

Blocklists should contain identities, not training labels. Keep test pair arrays and labels out of the training directory and trainer. Keep all test labels, lengths, and released row multiplicities unchanged for primary reporting. New STRING evidence contradicting a frozen test negative should be logged for a separately labeled sensitivity analysis; do not relabel the test to improve apparent accuracy.

The human and nonhuman benchmarks have already been inspected during development. They are useful historical comparisons, but they cannot now serve as wholly untouched confirmatory evidence for the research program. Use only the new human DEV set for selecting the checkpoint, stopping, and any calibration.

**9. Keep the native ESM2 model and adapt the imbalanced-data objective correctly.**

Initialize from the original pretrained `facebook/esm2_t33_650M_UR50D` language-model weights with a fresh PPI head. Fine-tune the entire model. Do not initialize this run from the best supervised v5, humanV11, or Bernett checkpoint: that would mix prior training histories with the proposed STRING preparation. The existing v5 checkpoints remain frozen comparison models.

Retain joint `CLS A EOS B EOS` input, standard attention, the native **CLS → ReLU → one linear logit** head, and the active MLM decoder. Use both orientations during training. There is no new ESMC run, chain-aware mechanism, additional MLP, MSA channel, or architecture search in this proposal.

The proposed objective is:

\[
\mathcal L = 10\,\operatorname{BCEWithLogits}(z,y;\mathrm{pos\_weight}=10)
             +\mathcal L_{\mathrm{MLM}}.
\]

Positive-class weight 10 compensates for the ten sampled negatives per positive; the outer factor 10 controls classification relative to MLM. They are separate settings. The released native trainer explicitly implements the former, while v5 correctly used positive-class weight 1 on its balanced data. Preserve example-mean BCE reduction; dividing by the sum of class weights would alter its magnitude relative to MLM. [Pinned native trainer](https://github.com/liudan111/PLM-interact/blob/ab2ae6ae1aa81accf4c1f7f6f341164baf97809a/PLMinteract/train_mlm.py), [v5 protocol](retrain-v5/PROTOCOL.md)

Use the already qualified v5 engine's globally correct distributed normalization and shared encoder forward, rather than transplanting older training-loop quirks. The public native trainer contains 22% masking, whereas the paper's chosen recommendation is 15%; v5 already explicitly uses 15%. Keep 15%, record the deviation from public-code defaults, and do not claim an exact replay of the unavailable humanV11 training trajectory. TUnA also differs in objective, frozen ESM2-150M embeddings, and model architecture; matching its pairs does not make these the same training procedure.

| Training setting | Proposed fixed value |
|---|---|
| Production runs | One ESM2-650M model, seed 2 |
| Hardware | One Arrhenius node with four GH200 GPUs |
| Physical pairs/update | 64; both orientations give 128 oriented examples |
| Learning rate | 2e-5 |
| Optimizer | AdamW; weight decay 0.01; global gradient norm clipping 1.0 |
| Schedule | 2,000 warmup updates, then linear decay over a horizon computed from the frozen corpus |
| Maximum horizon | Five complete physical-pair passes; do not copy v5's old absolute update count |
| Corruption | Existing deterministic 15% MLM scheme; same residue corruption across AB/BA |
| Precision | Existing qualified BF16 autocast with FP32 parameters/optimizer |
| Validation | Full DEV every quarter pass; clean inputs |
| Selection | Highest full-DEV AP from mean AB/BA raw logits; exact ties keep the earlier checkpoint |
| Early stopping | After at least two passes, stop after four consecutive quarter-pass evaluations without a new best AP |
| Evaluation diagnostics | AUROC, original-order AP, BCE/MLM loss, order gap, and the fixed degree baseline |

Five passes and full-DEV pooled scoring are deliberate v5-engine choices, not claims to reproduce the paper's ten-epoch training and sampled validation. Neither test species nor the two old human test sets should decide whether the horizon is extended. Changes to runtime packing must preserve the physical-example and masked-token loss normalizers.

A fixed 1:10 training sample and weighted loss do not make the output a calibrated probability of interaction among arbitrary proteome-wide pairs. If an operating threshold is needed, choose it on DEV and label the applicable prevalence. Ranking metrics can be reported without inventing a universal probability interpretation.

**10. Feasibility, cost, and resumability on Arrhenius.**

The proposed input ceiling is only 1,603 tokens. The existing v5 implementation already handled much longer inputs on four GH200 GPUs, so memory feasibility is well supported. Data parsing, clustering, and random-negative construction are CPU tasks; this design has no ILP solver or paired-MSA search. Reuse the qualified ESM2 SIF at [images/plm-interact](images/plm-interact/README.md). A new ESM2 runtime image is not justified by this proposal.

I inspected the actual v5 ESM2 event records. Across 2,406 logged updates whose longest concatenated input was at most 1,603 tokens, median update time was **1.80 seconds**, mean **1.90 seconds**, with 64 physical pairs/update on four GPUs. This is useful reference evidence, not a measurement of the future STRING length distribution, communication behavior, or complete job walltime. [Compute reference](literature/string-v12.5-review-v6/compute-reference.json)

Let P be the final retained positive count across TRAIN+DEV. With the proposed 8:1 split, ten negatives per positive, and five passes, the approximate maximum optimizer-update count is:

\[
U_{\max}\approx\frac{5\times11\times(8/9)P}{64}=0.764P.
\]

Illustrative budgets, before early stopping:

| Retained positive count P | Approximate TRAIN rows | Updates for five passes | Training-compute hours at 2–3 s/update |
|---|---:|---:|---:|
| 50,000 | 488,889 | 38,194 | 21–32 |
| 100,000 | 977,778 | 76,389 | 42–64 |
| 150,000 | 1,466,667 | 114,583 | 64–95 |
| 296,306, raw candidate ceiling before exclusions | 2,897,214 | 226,345 | 126–189 |

These are conditional calculations, not forecasts of the final dataset size. Add full validation, checkpoint I/O, launch/requeue overhead, and queue waiting separately; allocation GPU-hours are four times allocated wall-hours. The last line is an upper planning scenario before mandatory reductions, not the recommended dataset size. Representative one-node qualification timings on the final length distribution should replace these estimates before submission.

One bounded qualification is sufficient: verify weighted BCE/MLM against a reference calculation, run a small representative and maximum-length batch, and confirm interrupted/resumed four-GPU execution preserves the training state. This is execution verification, not another learning-rate or model-selection screen. CPU preparation can start with 16–32 cores and a conservative memory allocation; record actual memory/elapsed time rather than reserving GPUs for it.

Retain the existing atomic checkpoint system: model and MLM decoder, optimizer, exact scheduler/update position, per-rank RNG, sampler cursor/cycle, corruption identities, examples seen, pending validation, and best-checkpoint metadata. Checkpoint at early health confirmation, roughly every 15 minutes, validation boundaries, and scheduler stop/requeue signals. Keep best, latest, and a valid fallback; do not retain every optimizer snapshot. A rough 50–100 GB working allowance for checkpoint generations and exports is sensible until the exact format is measured.

Resume the same frozen data, code, image, four-GPU world size, and fixed horizon. A resumed job must not resample negatives or reset its learning-rate schedule. An interrupted validation must restart from a committed checkpoint and cannot publish partial metrics. Reuse the production launcher pattern with walltime warnings and safe requeue. The existing v5 configurations contain their historical absolute run root; any v6 copy must be rebased and resealed, not executed unchanged.

**11. Finish this round with one training run and one frozen evaluation.**

The proposed work products are a `data-preparation-v6` release, one `retrain-v6` run, and a separate `benchmark-v6` evaluation after DEV selection. Those production folders are proposed future work, not created by this investigation.

The final benchmark should use the existing five-species files and frozen comparator predictions wherever their input and scoring identities remain valid. Compare against humanV11, TUnA human seed 47, and v5 ESM2 at minimum; reuse the remaining completed competitors for context. Report AP and AUROC for every species, an unweighted species macro-average, paired uncertainty intervals, and the same duplicate/conflict sensitivity already used in the project. Do not choose which species to emphasize after seeing the result.

Use the established common scoring convention for the main comparison: mean AB/BA raw logits for concatenation models. Retain original-order values separately for correspondence with the native paper. Existing humanV11 predictions used qualified FP32; no precision change should be introduced simply to save a small amount of inference time. Common exact-unexposed subsets and homology-stratified results should be clearly separated from full historical-test results.

Predeclare the primary target as improvement in **macro AP across all five species relative to humanV11**, with species-wise results and macro AUROC accompanying it. Report the magnitude and uncertainty of the difference. A tiny positive average driven by one organism is weaker evidence than consistent gains. Failure to exceed humanV11 remains a valid completed outcome; it should not trigger more production variants automatically.

Evaluate the two old human tests as secondary distribution/retention checks, with the exposure qualifications above. Keep their shared positives in mind: they are not two independent replications. A broad statement that v6 is a better PPI model would require evidence beyond these familiar, source-dependent tests.

**My assessment is that this is worth one controlled attempt.** The native architecture has already worked strongly when trained on the relevant STRING collection, and the newer source contains additional directly supported candidate interactions. The largest motivation is matching a demonstrably different training distribution, not an assumption that a newer version number guarantees better biology. The newly measured degree shortcut makes transparent preparation and restrained interpretation essential.

**Evidence and reproducibility record.** The investigation is reproducible from the following artifacts:

- [Source download provenance](literature/string-v12.5-review-v6/source-downloads.json), [pinned code sources](literature/string-v12.5-review-v6/code-sources.json), and [local input hashes](literature/string-v12.5-review-v6/local-inputs.json).
- [Human STRING download URLs and SHA-256 hashes](literature/string-v12.5-review-v6/string-downloads.json), [HTTP metadata](literature/string-v12.5-review-v6/string-file-heads.json), and [count/evidence comparison](literature/string-v12.5-review-v6/string-source-audit.json).
- [Read-only source audit implementation](literature/string-v12.5-review-v6/audit_sources.py), [released-data equality and cleanup counts](literature/string-v12.5-review-v6/legacy-source-audit.json), and [protein-universe audit](literature/string-v12.5-review-v6/legacy-protein-universe-audit.json).
- [TRAIN-degree diagnostic implementation](literature/string-v12.5-review-v6/audit_degree_baseline.py) and [its complete numerical results](literature/string-v12.5-review-v6/legacy-degree-baseline.json).
- The completed [nonhuman benchmark](bechmark-v5-nonhuman/REPORT.md), [human benchmark](benchmark-v5/REPORT.md), and [v5 training contract](retrain-v5/PROTOCOL.md) provide the existing-model evidence. The nonhuman benchmark folder is currently named `bechmark-v5-nonhuman`.

The download provenance records failed direct HTML fetches as failures; official release/schema information was separately verified through the web research tool and the successfully downloaded data headers. Missing historical construction details remain explicitly unresolved rather than filled in with assumptions.
