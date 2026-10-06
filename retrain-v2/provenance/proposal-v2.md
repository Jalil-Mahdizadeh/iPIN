**Improving PLM-interact: second proposal after the v1 benchmark**

Prepared 29 September 2026. Evidence cutoff for the new local audit: 10:22 UTC. Primary target: human PPI ranking on unseen proteins, with average precision (AP) as the main endpoint and AUROC as a secondary endpoint. Cross-species generalization is a separate confirmation task.

**Recommendation.** My strongest new architectural hypothesis is to make the model distinguish within-protein sequence organization from between-protein interaction evidence, and give its classifier access to residue-level evidence beyond the single CLS token. My strongest data hypothesis remains improving the evidence and supervision attached to each pair. The v1 experiments did not implement that data rebuild. Before combining these ideas, establish controlled training baselines and test whether class weighting, masking, and adaptation strength explain part of the gap to the released checkpoint.

There is currently **no demonstrated recipe for outperforming native PLM-interact** in this project. The proposal below ranks plausible interventions, identifies their failure modes, and specifies what would count as convincing improvement. More sequence coverage and exact prediction symmetry are useful properties, but our results do not justify treating them as reliable routes to higher AP.

**What the current results actually establish**

The frozen [benchmark-v1 report](benchmark-v1/REPORT.md) compares three checkpoints on the same 52,048 Bernett test pairs, using full sequences, BF16 computation, and the mean of A–B/B–A logits for every model. Reference update 4,000 and symmetric update 7,000 were selected by their existing validation rules before test inference.

| Checkpoint | Pooled validation AP | Pooled test AP | Test AUROC | Test Brier ↓ |
| --- | ---: | ---: | ---: | ---: |
| Released native Bernett model | 0.642142 | **0.690319** | **0.699467** | 0.234875 |
| v1 reference, update 4,000 | **0.650275** | 0.675669 | 0.685761 | **0.224792** |
| v1 symmetric, update 7,000 | 0.647411 | 0.678164 | 0.687280 | 0.228095 |

Reference-minus-native AP is −0.014650, with paired protein-bootstrap 95% interval [−0.024693, −0.005176]. Symmetric-minus-native is −0.012155 [−0.018579, −0.005150]. Symmetric-minus-reference is only +0.002495 [−0.006472, +0.011209]. These approximate intervals condition on the observed test graph; they do not include training-seed or model-selection uncertainty.

Three conclusions matter for the next round:

1. **Selection on this validation set did not predict which training approach would win on test.** This is an observed ranking reversal, not proof of its cause. Sampling variation, differences between protein families, repeated model selection, and training choices remain competing explanations.
2. **The long-sequence intervention was not isolated.** Both v1 arms use full-length training, while their comparison with native also changes other training details. On the 3,392 long test pairs, symmetric AP is 0.641743 versus native 0.644194; their difference has a wide interval spanning zero.
3. **Lower Brier is a distinct benefit.** It indicates lower squared probability error on these benchmark labels. It does not establish better ranking, or calibrated probabilities of physical interaction in a real proteome.

The latest validations available at the audit cutoff were update 9,000. Pooled AP was 0.641374 for reference and 0.641353 for symmetric; the selected updates remained 4,000 and 7,000. Both five-epoch runs were still active. Therefore, the benchmark remains an interim comparison, not a final statement about every checkpoint the runs might produce. [Captured training observations](literature/plm-interact/improvement-review-v2/existing-data-audit.json).

**New data checks change the diagnosis**

I ran a CPU audit of the existing arrays and saved predictions in the supplied SIF. It checked row and label alignment before recomputing metrics; it fitted no models and selected no new checkpoints.

| Prepared split | Pairs | Unique protein sequences | Median combined residues | Pairs above 2,193 residues |
| --- | ---: | ---: | ---: | ---: |
| Training | 163,085 | 4,285 | 1,386 | 20.00% |
| Validation | 59,258 | 3,710 | 969 | 8.33% |
| Test | 52,048 | 3,022 | 917 | 6.52% |

Training therefore allocates substantially more pair exposure to long inputs than either evaluation split. This is a reason to investigate coverage, sampling, and compute allocation separately. It is **not** a reason to tune the training mixture to the already-observed test distribution.

Length mix is also insufficient to explain the ranking reversal: among short pairs alone, reference beats native on validation AP, 0.655686 versus 0.647409, but loses on test, 0.680836 versus 0.693226. This points toward evaluating multiple independent protein groups rather than relying on a single aggregate validation score. [Audit script](literature/plm-interact/improvement-review-v2/audit_existing_results.py), [complete audit](literature/plm-interact/improvement-review-v2/existing-data-audit.json).

An important correction to a simplistic negative-sampling diagnosis: **Bernett negatives already preserve positive-network protein degrees in expectation.** They are sampled using a degree-weighted protein multiset, rather than uniformly from all proteins. Our within-split positive/negative degree correlations are correspondingly high: Spearman 0.972 in training and approximately 0.942 in validation and test. This does not eliminate all sampling artifacts, but it makes “introduce degree matching” an inadequate new contribution. Their labels still represent sampled unreported interactions, not confirmed noninteractions. [Bernett et al., 2024, dataset construction](https://academic.oup.com/bib/article/25/2/bbae076/7621029).

**Reassessment of the original paper and implementation**

The central contribution remains credible: joint sequence encoding and end-to-end ESM-2 adaptation produce useful PPI predictions. Our [native reproduction](plm-interact-reproducability/REPORT.md) supports the released checkpoint results. However, reproducing inference and reconstructing the training procedure are different accomplishments.

I revisited the [paper](literature/plm-interact/article.pdf), [supplement](literature/plm-interact/supplementary-information.pdf), [peer-review exchange](literature/plm-interact/peer-review.pdf), [reporting summary](literature/plm-interact/reporting-summary.pdf), deposited [source tables](literature/plm-interact/source-data-small-tables.json), and [archived trainer](plm-interact-reproducability/provenance/publication-code/PLMinteract/train_mlm.py). The following points directly affect the next design:

| Finding | Consequence |
| --- | --- |
| The classifier reads ReLU(CLS) through one linear layer, although the encoder computes contextual representations for every residue. | A richer readout is a modest, testable architectural intervention; it need not require a larger backbone. |
| Concatenation gives cross-chain residue pairs ordinary ESM-2 rotary positional relationships. | Chain order and chain length introduce positional structure that does not correspond to covalent proximity between proteins. This is a plausible modeling limitation, not yet a demonstrated cause of our error. |
| The released trainer classifies masked inputs; clean inputs are used for inference. | Separate the effects of sequence corruption and auxiliary MLM supervision. |
| Methods/code use MLM:classification weights 1:10; Results prose reverses the description. The script also hardcodes positive-class weight 10. | Loss labels alone do not define the effective objective. Verify actual reductions and positive weighting. The exact Bernett checkpoint training configuration remains unverified. |
| v1 uses positive weight 1 and per-pair MLM normalization; the published script uses masked-token averaging. Other scheduling, batching and masking details also differ. | v1 reference is a corrected retraining baseline, not an exact reconstruction of native training. Native-versus-v1 differences cannot be assigned to length alone. |
| Final loss-ratio selection used the five cross-species test sets; Reviewer 3 raised this explicitly. | Those familiar tests support historical comparison, not an untouched confirmation of new tuning choices. |
| Source-data mouse AP is 0.907840 without masking and 0.903963 with 15% masking. Supplementary Figure 2 uses a classification-outcome significance test. | MLM is not uniformly beneficial, and McNemar significance does not establish an AP improvement. |
| The selected native Bernett epoch/update and its full training history are unavailable in the checked releases. Supplementary Figure 10 concerns the mutation model. | Do not compare our update numbers with the mutation model's epoch 19 or assume the same convergence schedule. |

The paper's strict Bernett ranking result is also much less decisive than its cross-species headline: deposited native AP/AUROC are 0.686620/0.697756, versus TUnA 0.691908/0.703356. These are historical score-file comparisons, not a newly matched inference experiment or a significance claim. The approximately 0.69 AP level is a useful baseline to challenge, not evidence that the task is nearly solved.

**What relevant newer research contributes**

| Primary source | Useful evidence | Limit on the inference we should make |
| --- | --- | --- |
| [PPLM, Liu et al., Nature Communications, March 2026](https://www.nature.com/articles/s41467-026-70457-5) | Distinguishes within-chain positional attention from cross-chain attention; uses paired pretraining and richer pooled features. | Its PPI comparison is cross-species and uses an ensemble of five selected models. It does not establish a matched Bernett win over our native checkpoint or isolate architecture from additional data. |
| [MINT, Ullanat et al., Nature Communications, January 2026](https://www.nature.com/articles/s41467-025-67971-3) | Adds interaction-aware attention and learns from 95.8 million STRING-derived training pairs. Reports approximately 0.69 AUPRC on Bernett with downstream predictors. | That rounded result is not an obvious improvement over native. Its scale does not justify assuming that more paired pretraining will solve our problem. |
| [Chatterjee et al., Bioinformatics, 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12080959/) | Studies topology-informed negative sampling and evaluates it against alternative negatives. | A topology-derived nonedge is still an inferred negative. Its selected evaluation distribution differs from Bernett; graph-based sampling must use training information only. |
| [TUnA, Ko et al., Briefings in Bioinformatics, 2024](https://escholarship.org/content/qt9q42r5vh/qt9q42r5vh.pdf) | Combines ESM-2 representations, transformer processing, and uncertainty estimation. | Uncertainty-aware filtering can improve accepted predictions while reducing coverage. It is not automatically an improvement in full-set AP. |
| [PRING, Zheng et al., NeurIPS 2025](https://papers.nips.cc/paper_files/paper/2025/file/86a4ea51ccac6c830230f281a23e74c8-Paper-Datasets_and_Benchmarks_Track.pdf) | Evaluates network reconstruction and biological organization beyond isolated pair classification. | It uses released PLM-interact weights while retraining many comparators. It motivates additional endpoints, not an automatically fair architecture ranking. |
| [Szymborski and Emad, Nature Machine Intelligence, February 2026](https://www.nature.com/articles/s42256-025-01176-7) | Controlled experiments show that PLM pretraining exposure can inflate downstream PPI evaluation. | This is not a measurement of contamination in our checkpoint. Shared ESM-2 initialization helps comparative fairness but does not establish complete pretraining independence. |

In particular, a released interaction-pretrained encoder may already have encountered a held-out pair as a positive during pretraining, even if its downstream classifier did not. Before importing MINT/PPLM weights, audit pair, protein, homology and date overlap across **all** pretraining and fine-tuning sources. Unverifiable exposure makes the comparison exploratory. Architecture ideas can instead be tested from the same ESM-2 initialization as our controls.

**Ranked improvement candidates**

| Priority | Intervention | Why it earns a place | Main risk |
| --- | --- | --- | --- |
| 1: first inexpensive training tests | Clean classification objective and controlled encoder adaptation | Directly tests a train/inference mismatch and unresolved training confounds. | Removing a useful regularizer can reduce generalization. |
| 2: strongest new architecture hypothesis | Residue-aware readout, then chain-aware attention | Targets how interaction evidence is represented and extracted. | Added flexibility can overfit; custom attention may be expensive. |
| 3: highest-value data investment | Evidence-aware positives and carefully controlled uncertain negatives | Addresses label meaning and assay bias, which v1 did not change. | Data filtering can reduce diversity or change the prediction task. |
| 4: conditional extension | Experimental interface supervision for a limited auxiliary task | Supplies spatially relevant supervision missing from binary labels. | Structural coverage and interface annotations are biased and incomplete. |

**1. Give PPI classification the intact sequences; control how far the encoder moves**

The immediate hypothesis is that recovering residues and ranking interactions do not always need the same input corruption or representation changes. In v1, both losses act on the same masked sequences. A drop in MLM loss does not show that the model increasingly depends on the correct partner.

Test three objective variants against the same control: unmasked BCE alone; masked BCE alone; and unmasked BCE plus an auxiliary masked pass. These comparisons separate the effect of corruption from the effect of MLM. The current training pipeline enables masking through `mlm_weight > 0`; this must become an independent setting to implement the masked-BCE-only control correctly. Normalize losses explicitly and log occasional backbone gradient magnitudes and alignment between the two objectives. Select any auxiliary weight on development data, rather than transferring “1:10” without checking its effective gradient contribution.

For the first decoupled-MLM test, keep the same auxiliary pair population as the control. Restricting MLM to evidence-supported positives is a subsequent data ablation; otherwise, the objective and data changes become inseparable. Partner-shuffled and partner-removed diagnostic inputs can test whether any MLM improvement actually uses partner information. Treat these as diagnostic controls, not biological negative labels.

A second, separate test should reduce adaptation strength: a lower backbone learning rate, or gradual unfreezing followed by end-to-end training. The early validation peak and later probability-error deterioration justify this experiment, but do not prove catastrophic forgetting. Keep the readout and training data fixed when testing it.

For the historical Bernett track, conservative continuation from the released native checkpoint is also a sensible practical control. Its existing PPI knowledge may be valuable. Record that this is native-model adaptation, and audit its previous supervised exposure before using any newly constructed holdout. For the clean architecture comparison, start all arms from the same pretrained ESM-2 weights.

**Promote this idea if:** it improves AP consistently across development protein groups and seeds, with stable numerical behavior. **Reject it if:** it mainly improves Brier or training loss while ranking stays flat or worsens. Positive-class weighting can shift probabilities without improving their order; it is an empirical control, not a guaranteed ranking fix.

**2. Extract interaction evidence beyond CLS, and correct the treatment of chain boundaries**

Start with a small readout experiment. Pool contextual residue representations separately for the two chains, combine their sum, absolute difference and elementwise product, and feed these features to a modest classifier alongside the CLS representation. Compare mean pooling with a small learned pooling module. A residual branch initialized to make no contribution provides a useful preservation control when adapting native weights.

This tests whether the current classifier discards useful residue evidence. It does not assume that adding layers or parameters is intrinsically better. Compare with a parameter-matched CLS-only MLP so that a gain cannot be explained merely as “a bigger head.” Retain A–B/B–A pooling for every baseline: a commutative readout alone does not guarantee invariance when its contextual encoder remains order-dependent.

Then test chain-aware attention: retain the pretrained positional treatment within each protein, while allowing between-protein attention to learn interactions without a fictitious sequence-distance relationship across the separator. PPLM supplies a concrete precedent for this distinction; our hypothesis is that it will improve transfer to unfamiliar chain lengths and families. Compare an attention-only change and a readout-only change before combining them. [PPLM methods](https://www.nature.com/articles/s41467-026-70457-5).

Keep the 650M backbone initially. A larger PLM would introduce another expensive confound. Do not materialize all layer/head attention matrices for our longest sequences: prototype a memory-efficient implementation, preserve joint attention normalization, and measure its overhead before a full run. Attention weights or saliency maps alone are not validated contact maps.

**Promote this idea if:** improvements survive protein-disjoint validation, the parameter-matched head control, and matched inference. **Reject it if:** gains come only from familiar families, extra ensemble members, or reduced evaluation coverage. Symmetry remains a common requirement; it is not the claimed novelty of this experiment.

**3. Improve what the labels mean, rather than simply adding more pairs**

The first proposal's broad data recommendation remains untested. The next data version should retain provenance for every pair: source release, publication, assay, interaction type, organism, sequence/isoform mapping, evidence multiplicity, and observation date. Map existing Bernett pairs back to evidence where possible and quantify unmapped or ambiguous cases before deciding how much new data is useful.

Direct binary binding, co-complex association, and functional association should not be silently merged. Retain the Bernett endpoint for the historical comparison. For an expanded corpus, use source-aware weighting or separate supervision heads so that direct-binding labels do not redefine every historical positive. Deduplicate evidence shared across databases; repeated publications and assays should not create repeated independent examples.

Systematic human binary-interaction resources such as HuRI offer useful evidence and an assay-distinct validation opportunity. Their assay limitations matter, and many entries may already overlap HIPPIE or other training sources. “Not detected in a screen” is not a universal noninteraction label. [HuRI study](https://www.nature.com/articles/s41586-020-2188-x), [authors' assay limitations](https://interactome-atlas.org/faq/). IntAct's [curation manual](https://raw.githubusercontent.com/intact-portal/intact-portal-documentation/master/assets/intact-curation-manual.pdf) provides an evidence vocabulary for distinguishing interaction types.

Preserve Bernett-style degree control. Add only a limited, separately tagged set of more challenging candidates, matched on relevant nuisance properties such as protein length and study coverage. High-scoring unknown pairs should receive reduced negative confidence or an unlabeled treatment, rather than automatically becoming certain negatives. Small curated noninteraction resources can anchor a diagnostic panel, with their experimental context retained. [Negatome resource](https://mips.helmholtz-muenchen.de/proj/ppi/negatome/).

Topology-informed sampling is an optional ablation, with the graph built exclusively from the permitted training partition. Building it from a full database that includes held-out edges would leak information. Do not use subcellular separation as the sole negative rule, or assume every in-batch partner is negative: a protein can have multiple true partners. Positive–unlabeled learning is worth testing only with an explicit sensitivity analysis for the class prior and observation mechanism; selective experimental reporting makes its assumptions difficult.

Compare old and new training data at matched pair exposure before scaling up. Evaluate on both historical sampled negatives and an independently curated assay/evidence panel. A gain solely on the new sampler's own test distribution is weak evidence. Report gains and losses by assay, evidence strength, protein family and length; do not optimize these groups using the historical test labels.

**Promote this idea if:** its benefit transfers across evidence sources and persists after leakage and nuisance-feature controls. **Reject or revise it if:** the improvement disappears outside the sampling rule, depends on database identifiers, or sacrifices most low-study-coverage proteins. New usable pair counts and label-error rates remain unknown until curation is performed.

**4. Add a small amount of experimentally grounded interface supervision**

Binary labels provide little information about where a model should find interaction evidence. A later auxiliary task could predict partner-conditioned interface residues on a curated set of experimentally resolved human or established model-organism complexes. Keep sequence input at deployment; structural annotations would supervise training only.

Use biological assemblies and defensible sequence mappings. Exclude unresolved positions from the auxiliary loss, separate direct contacts from co-complex membership, and split by homologous proteins/complexes rather than PDB entry alone. A compact residue-level target is a more practical first experiment than storing every residue-pair attention map. Interface-rich crops can support this auxiliary task, but arbitrary crops must not inherit a positive full-protein PPI label when the interacting region may have been removed.

This is conditional on obtaining an adequately independent corpus. It is not a recommendation to generate millions of predicted complexes and treat their confidence scores as truth. The original supplement's selected structural examples do not establish a general interface-prediction capability.

**Promote this idea if:** it improves both PPI ranking and a separately held-out interface endpoint, including proteins without close structural homologues. **Reject it if:** it only helps well-structured protein families or improves contact localization without improving the PPI endpoint. This is a higher-risk research extension than priorities 1–3.

**A controlled experiment sequence**

Run one intervention family at a time. Retain the current five-epoch runs as evidence and use their prespecified final selection rules. The following are proposed future experiments, not claims about experiments already completed.

| Stage | Comparison | Question answered |
| --- | --- | --- |
| Baseline recovery | Corrected reference with full versus ≤2,193-residue training coverage; separately test positive weight 1 versus 10. Complete the 2×2 only if needed to resolve an interaction. | How much of the native gap relates to coverage or objective weighting? |
| Objective | Matched masked/unmasked classification and MLM controls described above. | Is corruption useful, and does MLM add transferable signal? |
| Readout | CLS linear, parameter-matched CLS MLP, and residue-aware readout. | Is residue access useful beyond head capacity? |
| Attention | Current versus chain-aware attention with the same selected readout. | Does chain treatment add value? |
| Data | Original versus evidence-curated supervision with the same model and pair exposure. | Does training-signal quality help independently of architecture? |
| Confirmation | Frozen best candidate and control across three seeds; then the locked external evaluation. | Is the benefit stable and transferable? |

For the coverage comparison, match optimizer updates, global pair exposure, loss reductions, learning-rate schedule, seed and validation rules. Report both effective epochs and GPU-hours: repeating the shorter subset is different from seeing additional long pairs. Existing runs are reusable controls only where these conditions actually match. Document any remaining mismatch instead of calling the comparison exact native retraining.

Do not recreate suspected public-code defects to achieve “faithfulness.” Recover documented modeling choices in a validated trainer and acknowledge that unavailable native run metadata prevents exact reconstruction.

**Evaluation that can support an outperformance claim**

Use two explicitly different claims:

- **Historical improvement:** higher AP than the released Bernett model under identical evaluation. This is measurable, but Bernett test results have already influenced research decisions and are now exploratory evidence.
- **Improved generalization:** a frozen candidate outperforms matched baselines on a new, independently held-out protein/evidence distribution. This requires additional data curation and exposure auditing.

For development, add at least three protein/homology-group validation partitions, with matched controls trained afresh for each partition. Use mean performance and a prespecified tolerance for deterioration on any partition. Where these partitions are carved from historical training data, compare models initialized from ESM-2; the released native checkpoint has already seen those training labels and is not a clean fold baseline. Its original validation endpoint remains useful as a historical reference.

Reserve a new external test set before candidate optimization, preferably combining an assay/source distinction with a publication-time distinction. Publication chronology alone is insufficient: older databases may already contain the same interaction, and pairwise novelty is weaker than both-protein novelty. Audit identifiers, exact sequences, pair overlap, homology, evidence reuse and all known checkpoint training sources. Keep both-proteins-unseen and one-protein-unseen tasks separate. Do not promise a particular holdout size before examining the available evidence.

Use AP from raw scores with explicit prevalence, AUROC, Brier, and validation-selected operating points. Include per-protein partner retrieval or precision at a fixed experimental budget where meaningful. Report full coverage alongside any uncertainty-based abstention. A 50:50 benchmark does not represent proteome-wide prevalence. Calibration must use representative development data; a strictly monotone calibration transform cannot repair ranking.

Retain simple sequence/length/composition and training-only similarity controls to test whether improvements rely on nuisance information. Use paired protein- or homology-group uncertainty estimates, report training-seed variation separately, and avoid treating shared-protein pairs as independent replicates. Freeze all selected models and inference settings before the external test is opened.

As a planning target, seek **at least +0.01 absolute AP** over the matched native baseline, with a paired interval excluding zero, consistent seed results, and no material AUROC deterioration under a tolerance fixed on development data. This is a proposed practical threshold, not a forecast. Recompute native under the final precision protocol: 0.690319 is specifically the current pooled BF16 baseline.

Precision needs attention. Our full native original-order comparison found only a −0.000281 AP change between earlier FP32 and BF16 inference, but individual probability differences reached 0.110910. The next confirmation should use common efficient FP32 inference where feasible, or qualify a common mixed-precision protocol before evaluating outcomes. The measured native AP change is much smaller than the approximately 0.012–0.015 gap; a complete matched FP32 comparison would still be needed to quantify differential precision effects across models. [Numerical audit](benchmark-v1/provenance/precision-outlier-audit.json).

If a claim extends to the original paper's Figure 2, train a separately specified human cross-species model and evaluate that task explicitly. Improving the Bernett/Figure 4 model alone would not establish improvement across all main paper results.

**Arrhenius feasibility and a bounded budget**

The present runs demonstrate that end-to-end 650M PPI fine-tuning is feasible on one four-GH200 node in the existing ARM64 SIF. This is stronger evidence than the earlier synthetic profiling. At the audit cutoff, approximately 9,100 updates required 14.5 hours per run. Linear projection to 12,745 updates gives approximately 20.2–20.3 hours; allow **20–25 wall-hours, or 80–100 GPU-hours per baseline-sized full run**, as a planning range. This is an extrapolation, not a completed runtime measurement.

Recorded rank-zero peak allocated GPU memory was approximately 21.3 GiB. That figure is not an all-rank reserved-memory bound, especially for a new attention implementation. Training includes inputs up to 16,322 tokens, and evaluation has successfully handled a 39,391-token pair. These facts support reusing the current batching/checkpointing infrastructure; they do not guarantee that dense custom attention will fit.

| Work package | Planning scale at the current Bernett size | Qualification |
| --- | --- | --- |
| Data/provenance audits and saved-score diagnostics | Primarily CPU | Runtime grows with external data and homology searches. |
| Initial training tranche | Four baseline-cost runs: approximately 320–400 GPU-hours | Choose the most informative controls first; this is not the entire ablation matrix. |
| One candidate plus one control, each with three seeds | Six baseline-cost runs: approximately 480–600 GPU-hours | Development partitions or larger datasets add further runs. |
| Chain-aware attention / separate MLM pass | Profile first | May substantially increase runtime; the baseline estimate does not cover it automatically. |
| New large paired-pretraining campaign | Defer | MINT-scale training is far beyond a small PPI fine-tuning campaign. |

A comparable first tranche plus independent three-seed confirmation therefore needs roughly **800–1,000 GPU-hours**, before extra validation folds, attention overhead, or data expansion. Stage the spending: a failed controlled screen should prevent a larger combined run.

PPLM reports paired pretraining on four A100s for 50,000 steps, showing that a smaller interaction-pretraining project can be technically conceivable; it gives no measured Arrhenius runtime for our inputs. MINT reports four million training iterations. Neither budget should be inferred from the cost of our current supervised run. [PPLM training methods](https://www.nature.com/articles/s41467-026-70457-5), [MINT training methods](https://www.nature.com/articles/s41467-025-67971-3).

Use the single-GPU interactive allocation for compatibility and memory qualification, and four-GPU SLURM jobs for full runs. The live `gpu` partition allows up to three days, with requeue preemption enabled. The existing account works, but **remaining project allocation was not established**: `projinfo` reports consumption and explicitly warns that Arrhenius GPU accounting may be unreliable. Available disk headroom at capture was approximately 7.2 TiB below the shared soft quota; it is shared project capacity, not a private reservation. [Live resource snapshot](literature/plm-interact/improvement-review-v2/arrhenius-snapshot.json), [official NAISS job guidance](https://hpc.pages.naiss.se/training/NAISS_Slurm/jobscripts/).

Retain atomic resumable checkpoints containing model, optimizer, schedule, RNG, sampler position, configuration and data hashes. A current full checkpoint is approximately 7.82 GB; retaining two recent checkpoints plus the best costs approximately 23.5 GB per run before logs and immutable benchmark snapshots. Validate resume equivalence when changing the optimizer, objective or distributed implementation. Reuse the SIF for established components; qualify new ARM/CUDA dependencies separately rather than assuming an external repository will run unchanged.

**Ideas to deprioritize**

Increasing sequence length alone, repeating symmetric-loss training, choosing later checkpoints because they trained longer, and enlarging ESM-2 without a controlled mechanism test have weak support from the current evidence. Reweighting the test set or tuning ensembles on its labels would improve the appearance of results without establishing transfer. A native-inclusive ensemble can be a development-selected practical baseline, but it must be reported separately from improvement in one model.

Retrieval-oriented systems such as [RaftPPI's author implementation, ICLR 2026](https://github.com/AndyJZhao/RaftPPI) suggest a useful future route for proteome-scale speed. Speed is a separate objective from exceeding native AP on the current complete-pair benchmark. Likewise, uncertainty estimation and calibration deserve evaluation for usability, but they should not substitute for the ranking improvement requested here.

**Review artifacts.** New work supporting this proposal consists of the [CPU audit and captured inputs](literature/plm-interact/improvement-review-v2/existing-data-audit.json), [training-log snapshots and audit script](literature/plm-interact/improvement-review-v2/), [resource snapshot](literature/plm-interact/improvement-review-v2/arrhenius-snapshot.json), and [locally retained research sources](literature/plm-interact/improvement-review-v2/sources/). The original [v1 proposal](improvment-proposal-v1.md), native reproduction, and frozen benchmark remain the provenance for the earlier experimental results.
