**Third proposal to improve PLM-interact — evidence, hypotheses and a staged research plan**

Prepared 30 September 2026. Primary task: human PPI ranking on the complete Bernett benchmark, followed by independently reserved human interaction data. This is a research proposal, not a claim that the proposed model already outperforms native PLM-interact. It draws on the completed v1/v2 experiments, a new audit of saved predictions and training logs, renewed examination of the original article and associated information, and a broad search of primary literature available by this date.

**My recommendation is to preserve useful representations while learning better interaction evidence.** Start from clean BCE as our strongest reproducible training baseline. Test gentler encoder adaptation and a small residue-aware residual readout, with appropriate controls. In parallel, test conservative adaptation of the released native checkpoint as a practical route to improving that model. Invest in evidence provenance and protein-balanced development before committing to substantial data expansion. Reserve chain-aware attention and structural auxiliary learning for mechanisms that survive these cheaper tests.

The strongest reason for this ordering is our own evidence: removing masking and MLM recovered the native model's aggregate ranking, but extending training substantially degraded validation performance. Richer readout, lower learning rate, evidence-curated training, and chain-aware attention have **not** yet received production tests. V1 and v2 therefore do not establish that architectural improvement has failed. They establish that more sequence coverage, a different symmetry loss, or a larger positive weight is insufficient by itself under the configurations tested.

The proposal adds several concrete directions beyond v2: a native-preserving adaptation branch; analysis of complementary predictions and protein-level ranking; supervision aimed at distinguishing partners of the same protein; an explicit clean-objective/coverage interaction; and a bounded structural pretraining experiment informed by newer work. It also changes execution priorities: fewer uncontrolled full runs, more matched development comparisons, and a clear separation between improving the released checkpoint and demonstrating a generalizable training method.

**What our completed experiments actually show.** The table below uses one validation-selected checkpoint per run. All seven models share complete sequences, mean A–B/B–A logits, BF16 computation with FP32 parameters, and the same 52,048 test pairs. AP is average precision calculated from raw ranking scores. Test prevalence is exactly 50%.

| Model | Selected update | Validation AP | Test AP | Test AUROC | Test Brier ↓ |
| --- | ---: | ---: | ---: | ---: | ---: |
| Released native | Unreported | 0.642142 | 0.690319 | 0.699467 | 0.234875 |
| V1 reference | 4,000 | 0.650275 | 0.675669 | 0.685761 | 0.224792 |
| V1 symmetric | 7,000 | 0.647411 | 0.678164 | 0.687280 | 0.228095 |
| V2 reference | 4,000 | 0.653895 | 0.669386 | 0.681621 | 0.225789 |
| V2 capped | 8,000 | 0.651579 | 0.685180 | 0.691005 | 0.231953 |
| V2 positive weight 10 | 4,000 | 0.654795 | 0.672726 | 0.690719 | 0.392505 |
| V2 clean BCE | 4,000 | 0.652156 | 0.690596 | 0.699727 | 0.219897 |

The authoritative numerical record is [benchmark-v2/REPORT.md](benchmark-v2/REPORT.md), with [paired comparisons](benchmark-v2/results/paired-differences.csv) and the [frozen evaluation protocol](benchmark-v2/PROTOCOL.md). The four v2 production runs completed; v1 was stopped earlier at the user's request. Historical v1 checkpoints are useful comparators, but v1/v2 differ in their training stream and checkpoint-selection rules.

| Comparison | Test AP difference | Conditional 95% interval | Supported interpretation |
| --- | ---: | --- | --- |
| Clean BCE − native | +0.000277 | [−0.007765, +0.008157] | No demonstrated ranking superiority; no formal equivalence claim either |
| Clean BCE − v2 reference | +0.021209 | [+0.010962, +0.031347] | Strongest encouraging intervention in this historical, one-seed comparison |
| Capped − v2 reference | +0.015794 | [+0.006659, +0.024670] | Coverage policy mattered under the masked/MLM objective |
| Positive10 − v2 reference | +0.003340 | [−0.003897, +0.009671] | No convincing AP improvement from positive weighting |
| Clean BCE − v1 reference | +0.014926 | [+0.007152, +0.022780] | Improved over our earlier selected reference |
| Clean BCE − v1 symmetric | +0.012431 | [+0.004342, +0.019653] | Improved over our earlier selected symmetric model |

These intervals resample proteins jointly across models, conditional on the observed test graph and selected checkpoints. They do not include training-seed, hyperparameter-selection, or remote-homology uncertainty. The historical test has already influenced our research decisions, including this proposal. These comparisons are exploratory evidence; further experimentation cannot restore its status as an untouched test.

**Four corrections to our earlier interpretation matter.** First, clean BCE did help relative to our matched reference on test. The earlier validation-only report was written before that test evaluation. Its observation about late deterioration remains correct, but must not be interpreted as failure of the selected clean checkpoint. Second, the capped model improved test AP relative to full coverage; that does not establish that long sequences are intrinsically harmful. Capping changes the proteins and pairs seen, their repetition rate, and computational cost. Third, native receives the same symmetric inference pooling as our models. V1 tested a different training loss, not symmetry versus an asymmetric native comparator. Fourth, all four v2 models retained the original encoder/readout architecture. Their results say little about the untrained architectural proposals.

Clean BCE also reduced Brier error relative to native by 0.014978, with interval [−0.019017, −0.011291] for candidate minus native. This is useful probability accuracy against these benchmark labels and prevalence. It does not prove better ranking, universal calibration, or valid probabilities for all human protein pairs. Positive10 predicts every test pair positive at a 0.5 threshold; its F1 of 0.666667 at that threshold is the all-positive baseline.

**What the learning curves suggest, and what they cannot identify.** Clean BCE's validation AP fell from 0.652156 at update 4,000 to 0.611196 at update 12,745. A new CPU audit of periodically logged training batches found mean training BCE falling from 0.6277 during updates 3,000–4,000 to 0.1958 during the final approximately 1,000 updates. This combination is consistent with excessive fitting to the training distribution. It does not distinguish memorization, representation distortion, label noise, or distribution shift.

Every logged batch in those two windows exceeded the gradient clipping threshold in all four v2 runs. Clean BCE's mean pre-clipping norm rose from 8.26 to 69.64. These values include the prescribed classification-loss scale and are not evidence of numerical failure. They do mean that a change in objective scaling, clipping, or auxiliary losses can alter optimization in ways that “same learning rate” alone does not control. Log encoder/head gradient norms and actual parameter-update norms in future ablations. The records are in [training-window-audit.json](literature/plm-interact/improvement-review-v3/training-window-audit.json); these are sampled batch summaries, not full training-set losses.

**New evidence from the saved predictions.** I calculated equal-protein macro AP over incident candidate pairs, requiring at least two labeled positives and two labeled negatives for each eligible protein. This includes 3,049 validation proteins and 2,485 test proteins. It is a descriptive partner-ranking diagnostic, not a replacement for the primary benchmark or a new independent test.

| Model | Validation macro AP | Test macro AP |
| --- | ---: | ---: |
| Native | 0.673090 | 0.709882 |
| V1 reference | 0.666138 | 0.688194 |
| V1 symmetric | 0.672380 | 0.700804 |
| V2 reference | 0.669943 | 0.686541 |
| V2 capped | 0.675718 | 0.701894 |
| V2 Positive10 | 0.673903 | 0.695257 |
| V2 clean BCE | 0.673464 | 0.701567 |

Clean BCE ties native in aggregate AP while trailing it by 0.008315 in this test macro AP point estimate. Shared edges make protein outcomes dependent, and small candidate lists have different AP baselines; I have not attached a significance claim to this difference. Nevertheless, it is a reason to inspect whether improvements reach many proteins rather than mainly changing rankings around a few well-represented endpoints.

Native and clean BCE also make meaningfully different rankings. Their score Spearman correlations are 0.7912 on validation and 0.8068 on test. Their top 1,000 predictions overlap in 632 validation pairs and 624 test pairs. On validation, precision at this fixed budget is 0.890 for native and 0.909 for clean BCE; on test both are 0.899. Each contributes 323 labeled positives absent from the other's test top 1,000. This motivates a development-only ensemble experiment, but it does not show that an ensemble would improve AP or precision at the same budget. Taking their union increases the number of predictions.

For that pilot, predeclare an equal-probability blend as the simplest comparator. If tuning is justified, restrict it to a small fixed mixing grid on permitted development data, retain the individual-model endpoints, and keep any calibration fitting separate from mixture assessment. Each member first applies its usual orientation pooling. Freeze the complete mixture before another evaluation; do not choose its weight, calibration or members from historical test labels. Two members require approximately twice the encoder inference of one under this protocol, and an ensemble gain is not a single-model architectural gain.

No blends were fitted, no new checkpoint was selected, and no inference was rerun for this audit. [The script](literature/plm-interact/improvement-review-v3/audit_saved_predictions.py), [complete audit](literature/plm-interact/improvement-review-v3/saved-prediction-audit.json), and [per-protein results](literature/plm-interact/improvement-review-v3/protein-macro-metrics.csv) preserve the calculations and input hashes.

**Why validation has not been a reliable sole guide.** All six locally trained selections exceed native's historical validation AP, but none establishes test superiority. Positive10 has the highest v2 validation AP; clean BCE wins the v2 test comparison. This is evidence to broaden development, not proof that the validation labels are wrong. The number of models is small, they share training histories, and their validation maxima were selected after repeated evaluation.

The data also differ appreciably in length: median combined length is 1,386 residues in training, 969 in validation, and 917 in test. Approximately 20.0%, 8.33%, and 6.52% respectively exceed 2,193 residues. Full training contains 163,085 pairs but only 4,285 distinct sequences. Many pairs reuse the same endpoints, so the effective diversity is far smaller than the pair count suggests. These facts justify family, protein, evidence-source and length diagnostics. They do not justify fitting our training sampler to the observed test distribution. [Existing data audit](literature/plm-interact/improvement-review-v2/existing-data-audit.json).

**What I retain from the original paper.** Jointly encoding a pair and adapting ESM-2 remains a credible foundation. Our checkpoint reproduction supports its reported predictive performance. The original article does not establish that its particular masking recipe, CLS readout, length restriction or optimization schedule is optimal for the Bernett task. The five-epoch Bernett training horizon is described in the peer-review response; the selected native update is still unavailable. Supplementary epoch 19 refers to mutation-effect training, not this checkpoint. [Article](literature/plm-interact/article.pdf), [supplement](literature/plm-interact/supplementary-information.pdf), [reporting summary](literature/plm-interact/reporting-summary.pdf), [peer review](literature/plm-interact/peer-review.pdf).

Several original limitations remain relevant. Hyperparameter comparison used the five test species; changing its description to a technical benchmark does not make those results independent model selection. McNemar comparisons concern thresholded outcomes, not AP differences. Similar score distributions after swapping chains do not prove identical pair predictions. Selected predicted complexes, including low interface-confidence examples, do not demonstrate contact learning. The source data also do not support a universal masking benefit or uniform superiority in every identity bin. The earlier [source-data review](improvment-proposal-v1.md) documents those numerical checks.

We should therefore retain the original model's useful behavior without treating its explanations as established mechanisms. Conversely, incomplete native training metadata prevents attributing our retraining gap to one supposed implementation defect. The public positive-class weight, masking choices, MLM reduction and selected checkpoint cannot all be assumed to match the deposited Bernett run.

**What the broader literature changes.** I searched paired protein encoders, strict PPI benchmarks, residue readouts, representation-preserving fine-tuning, negative-label construction, interaction datasets, and efficient training. The following are the most decision-relevant findings; the [source inventory](literature/plm-interact/improvement-review-v3/SOURCES.md) records reading scope and publication status.

| Primary source | Finding relevant to us | Decision and limitation |
| --- | --- | --- |
| [Reim et al., Bioinformatics 2025](https://doi.org/10.1093/bioinformatics/btaf192) | Many ESM-2 embedding classifiers had similar performance despite architectural differences. | Include cheap embedding/head controls. Their approximately 0.65 **accuracy** is neither an AP ceiling nor a theorem about end-to-end PLM-interact. |
| [PPLM, Nature Communications 2026](https://www.nature.com/articles/s41467-026-70457-5) | Paired pretraining, different intra-/inter-chain positional treatment, and richer readout provide architectural precedents. | Supports a controlled chain/readout experiment. Its reported PPI pipeline includes a five-model ensemble and different data; it is not a matched single-model Bernett result. |
| [MINT, Nature Communications 2026](https://www.nature.com/articles/s41467-025-67971-3) | Large-scale interaction pretraining reaches reported Bernett AUPRC around 0.69 with downstream predictors. | Strong motivation for relational pretraining, weak evidence that more pretraining alone clears our current level. Audit pair exposure before checkpoint reuse. |
| [SMP, Nature Communications 2026](https://www.nature.com/articles/s41467-026-73885-5) | Monomer-derived pseudo-dimer pretraining improves its PPITrans baseline. Reported HIPPIE AP is 0.728. | A new auxiliary-learning hypothesis, **not** a demonstrated win on our test: its Methods list 35,496 HIPPIE test pairs, versus our 52,048. |
| [Bernett et al., September 2026 preprint](https://arxiv.org/abs/2609.10193) | Residual sampling shortcuts can persist after protein separation; apparently high-confidence negatives can introduce additional bias. | Audit before resampling. Preprint evidence; not a measurement of those biases in our current models. |
| [Ahmadian et al., Briefings in Bioinformatics 2026](https://doi.org/10.1093/bib/bbag376) | Entity-balanced evaluation/training targets degree-ratio shortcuts in multi-input tasks. | Motivates a training-sampling control. Its empirical applications are not our strict human PPI setting. |
| [Szymborski and Emad, Nature Machine Intelligence 2026](https://www.nature.com/articles/s42256-025-01176-7) | Controlled experiments show that upstream PLM exposure can inflate downstream PPI estimates. | Disclose pretraining exposure separately from supervised pair leakage; do not allege a measured contamination amount here. |
| [PRING, NeurIPS 2025](https://arxiv.org/abs/2507.05101) and [RaftPPI, ICLR 2026](https://openreview.net/pdf?id=Dp1RM3gPg8) | Network evaluation and efficient residue-aware retrieval address limitations of isolated pair scoring. | Add partner/retrieval diagnostics; retrieval speed is a separate objective from beating native AP. |
| [Kumar et al., ICLR 2022](https://arxiv.org/abs/2202.10054) and [Li et al., ICML 2018](https://proceedings.mlr.press/v80/li18a.html) | Head-first training and regularization toward initial weights can preserve transfer behavior. | Test the mechanism in PPI; these papers do not establish a PPI performance gain. |
| [ESME, iScience 2025](https://pmc.ncbi.nlm.nih.gov/articles/PMC12481099/) | Efficient attention, packing and adapters can reduce resource costs; adapter benefits vary by task. | Engineering options, not automatic accuracy improvements or guaranteed speedups over our already efficient implementation. |

I checked the SMP Methods and author evaluation code rather than comparing headlines. Its evaluation uses average precision, but its data sizes differ: 68,658 training, 39,451 validation and 35,496 test pairs. Exact sequence/pair coverage would have to be reconciled before a direct comparator run. [Retained article text](literature/plm-interact/improvement-review-v3/sources/smp-2026.txt), [pinned author evaluation code](literature/plm-interact/improvement-review-v3/sources/smp-ppi-evaluate.py).

The new shortcut literature also needs critical interpretation. A simple classifier performing above chance is a diagnostic, not proof that every signal it uses is spurious. Mean embeddings can retain legitimate biological information, and functional relatedness can be relevant to some PPI endpoints. We should define which nuisance correlations are unwanted for the stated task rather than optimize every cheap baseline down to chance.

**Priority 1: reduce destructive adaptation, and add information beyond CLS with a controlled residual head.** This is my strongest near-term model recommendation. It has a direct connection to the late deterioration, requires modest code changes, and can be separated into testable components.

For the clean ESM-initialized branch, compare the existing clean-BCE recipe at backbone learning rates 2e-5 and 5e-6, holding every other choice fixed. This is a specific adaptation-strength test, not a broad optimizer search. The prepared v2 lower-LR template currently inherits the masked/MLM reference; it must explicitly be combined with clean BCE and qualified before being called the proposed experiment. Do not silently compare two different objectives.

Then compare the native CLS-linear head, the existing parameter-matched CLS-only residual MLP, and the existing residue-aware residual MLP under the selected adaptation setting. Let \(c\) denote CLS and \(a,b\) the contextual residue means of the two proteins, excluding special and padding tokens. The proposed residual features are

\[
z=[c,\ a+b,\ |a-b|,\ a\odot b],\qquad
f_{\theta,\phi}(A,B)=f_{\mathrm{CLS},\theta}(A,B)+r_\phi(z).
\]

Initialize the residual output layer to zero. Preserve A–B/B–A pooling for all models; symmetric summary features do not make the underlying concatenated encoder order invariant. The implemented v2 residue and CLS-MLP controls each add 655,489 parameters. Their correctness was qualified, but their predictive value was never tested in production. [V2 implementation contract](retrain-v2/PROTOCOL.md), [qualification](retrain-v2/QUALIFICATION.md).

The mechanistic question is whether contextual residues contain transferable pair information that a single CLS vector fails to expose. If the residue head cannot beat an equally sized CLS MLP, do not describe a gain as evidence for residue interaction modeling. If neither beats the simple head, keep the simple head.

For a later readout refinement, use a small set of learned residue summaries or partner-conditioned pooling. A low-rank compatibility score between summaries can express local matching without retaining all layer/head attention matrices. Normalize aggregation for the number of candidate residue pairs; an unnormalized sum or maximum can acquire a length-dependent score distribution. Test shuffled/removed-partner diagnostics and fixed length/composition controls. These are sensitivity tests, not biological negative labels or proofs of physical contact.

**The practical native-initialized branch should run separately.** Load the released checkpoint, first verify exact initial score preservation, and train the zero-initialized residual head with the encoder frozen. Compare residue features to the matched CLS-only head on the same cached representations. Only if useful, unfreeze with a small backbone learning rate and a penalty toward the initial encoder:

\[
\mathcal L=\mathcal L_{\mathrm{clean\ BCE}}
 +\lambda\sum_\ell
 \frac{\|\theta_\ell-\theta_{\ell,0}\|_2^2}{|\theta_\ell|}.
\]

The penalty strength needs a bounded development comparison, including zero penalty; the equation defines the proposed normalization, not a validated coefficient. Head-first fitting and weight anchoring address different questions and should be logged as separate stages. Once the encoder changes, the residual construction no longer guarantees native predictions; preservation must be measured.

This branch may be our shortest route to a better usable checkpoint because it starts with the behavior we are trying to surpass. It is **native adaptation**, not a from-ESM replication. Native has already seen the parent training labels, so it cannot be used as a clean baseline or teacher on internal folds drawn from that training set. Develop this branch using permitted historical validation and a separately reserved external development panel; confirm only on an exposure-audited external test.

A low-rank adapter is a subsequent alternative if full adaptation still erodes performance. Its purpose here would be regularization, not solving a demonstrated memory bottleneck. It still incurs encoder activation/backpropagation costs. EMA is another inexpensive-in-code alternative, but must preserve its state in checkpoints and be selected on development data. Avoid stacking lower LR, adapters, weight anchoring, EMA and a new head in one first experiment.

**Falsification:** reject the architecture claim if a matched CLS head performs as well; reject the adaptation claim if gains vanish across folds/seeds or merely change probabilities. Examine single-protein embedding drift and held-out training-only language-model diagnostics to distinguish representation change from ordinary fitting, while recognizing that these diagnostics do not themselves measure PPI quality.

**Priority 2: resolve the clean-objective/coverage interaction before concluding anything about long proteins.** We have three cells of a useful factorial comparison:

| Training coverage | Masked classification + MLM | Clean classification, no MLM |
| --- | --- | --- |
| Full | V2 reference completed | V2 clean BCE completed |
| ≤2,193 combined residues | V2 capped completed | **Missing experiment** |

The missing clean/capped arm is a high-information run. It asks whether clean training's benefit and capped training's benefit coexist, conflict, or reflect the same underlying problem. Their gains must not be added arithmetically to forecast a combined AP.

To complete this historical comparison, retain seed 2, initialization, stream policy, 12,745 updates, global physical-pair exposures, learning-rate schedule, loss scale and validation rule. Reuse the completed controls only if their exact contract remains compatible. Evaluate every validation/test pair at full length. Report the different effective epochs: the full and capped v2 runs used approximately 5.00 and 6.25 passes respectively. This comparison controls exposures, not unique-pair diversity or compute.

If the interaction is favorable on development, test a softer coverage policy that retains long pairs at nonzero probability instead of permanently deleting them. Define length strata from training data, bound their sampling weights, and report both exposures and token/attention cost. At fixed update count, extra long-pair exposure must displace something else. At fixed compute, shorter examples allow more updates. These are different interventions and require separate interpretations.

The long-pair test AP point estimates are native 0.644194, clean BCE 0.633929 and v2 reference 0.623986. Thus full coverage alone did not recover native's long-pair behavior. Avoid choosing a length policy from those test outcomes; use the development folds and explicitly monitor long-pair performance with uncertainty.

For very long proteins, a later multiple-instance design could encode several domain/window pairs and aggregate evidence at the whole-pair level. Do not assign the original positive label independently to every arbitrary crop: the relevant interface may be absent. This is a potentially valuable efficiency/modeling experiment, but it changes the model much more than a sampler and needs its own coverage audit.

**Falsification:** drop a coverage policy if its improvement exists only in a favorable length subset, depends on discarding evaluation pairs, or fails on source/family development splits. A cheaper run with indistinguishable accuracy may still be worthwhile, but that is an efficiency result.

**Priority 3: make supervision reward partner discrimination across proteins, while retaining the primary pair objective.** Repeated endpoint exposure and the macro-AP audit motivate this direction. They do not prove that hub shortcuts caused our current errors.

First add controls trained only on permitted training information: sequence length/composition; independent pooled ESM embeddings with a small symmetric classifier; and an additive model \(u(A)+u(B)\) without a pair-specific interaction term. Compare with the cross-encoder on the same development splits. For unseen proteins, do not use their validation/test positive degrees as model features. A degree statistic calculated with evaluation labels is an oracle diagnostic, not a deployable baseline.

Next compare uniform pair sampling with a bounded mixture that samples proteins more evenly and then samples their observed partners. Log the resulting label, degree, length and evidence distributions. Uncorrected resampling deliberately changes the empirical objective; inverse-probability weighting would restore the old risk and answer a different question. Do not claim both simultaneously.

If the diagnostic suggests a genuine partner-ranking weakness, add a small auxiliary ranking loss comparing a labeled partner and a tagged negative/unlabeled-derived partner of the same query:

\[
\mathcal L_{\mathrm{rank}}
=\operatorname{softplus}\!\left[-\left(f(A,B^+)-f(A,B^-)\right)\right],
\quad
\mathcal L=\mathcal L_{\mathrm{BCE}}+\gamma\mathcal L_{\mathrm{rank}}.
\]

Use multiple supported positive partners where available, cap per-protein contribution, and retain confidence information for negative-derived examples. This loss cancels an additive query-only offset, but **does not** remove partner-only biases. It is a ranking surrogate, not direct optimization of average precision. Keep BCE as the anchor and require improvement in the unchanged global AP endpoint.

Our Bernett negatives already preserve degree distributions in expectation. Simply proposing degree matching again adds little. The new experiment concerns remaining endpoint dominance, partner discrimination and candidate difficulty. It must be checked against the existing degree control, rather than advertised as repairing an unbalanced random-negative dataset. [Bernett et al., 2024](https://academic.oup.com/bib/article/25/2/bbae076/7621029).

Do not treat every off-diagonal batch pair or every high-scoring unknown as a true negative. Proteins have multiple partners, and the most plausible unknowns may contain the most missed positives. Prefer modest, provenance-tagged challenging pools, sampling sensitivity analyses, and explicit exclusion of permitted known positives. All graph construction, similarity statistics and mining must remain within the authorized training partition.

**Falsification:** reject this branch if it improves macro AP only by sacrificing the primary overall AP beyond a prespecified tolerance, or if gains disappear after nuisance controls. Sampling effects and ranking-loss effects require separate ablations before combination.

**Priority 4: improve evidence meaning and diversity, not just the number of rows.** None of the completed v1/v2 runs trained on a newly curated interaction corpus. This remains a major untested opportunity, but also the work package most likely to consume research time before any GPU run.

Keep an explicit distinction between the historical Bernett endpoint and direct physical binding. Use a shared sequence encoder with separate evidence-type heads, or carefully controlled evidence weights, for direct binary interactions, co-complex observations and broader associations. Include a Bernett-compatible head so that the historical endpoint is not silently redefined. Missing supervision for one head is not a negative label for that head.

There is also an information limit: an observation can depend on cellular context, cofactors or additional complex members absent from a two-sequence input. A sequence-only predictor estimates a propensity under its training evidence distribution. Adding biological context could be valuable, but would define a different input setting and require matched context-enabled baselines.

HuRI offers systematic binary-interaction evidence; BioPlex offers complementary cell-context and complex-association evidence. IntAct/IMEx supplies experimental and interaction-type metadata. These resources can overlap each other and HIPPIE, so “another database” is not automatically new information. [HuRI](https://www.nature.com/articles/s41586-020-2188-x), [BioPlex](https://doi.org/10.1016/j.cell.2021.04.011), [IntAct metadata protocol](https://doi.org/10.1002/cpz1.70018).

The first deliverable should be an evidence ledger, with one row per supporting experiment linked to its unordered sequence pair: versioned identifiers and sequences, isoforms, publication, assay, interaction type, source release, first evidence date, complex-expansion method, and conflicting observations. Collapse duplicated evidence copied across databases. Retain multiple experiments as evidence rather than automatically repeating their pair in every batch.

Reserve external protein/publication groups before choosing training additions. Audit both endpoints, homologues, domains, duplicate experiments and all known upstream checkpoint exposures. Exclude withheld groups from auxiliary structural and interaction pretraining too. A newly downloaded record can describe an old, already-seen interaction; database update date is not first experimental evidence.

Compare a small curated addition against equal exposure to the old data before scaling it. Balance source contributions so that one prolific assay or family does not dominate. Weighting solely by the number of publications can amplify study bias. Unknown metadata must remain unknown rather than being filled with an inferred “high-confidence” label.

Use experimentally supported noninteraction evidence only within its assay/context. Negatome is useful as a small diagnostic resource, not an exhaustive list of proteins that can never interact. Ordinary sampled unknowns should retain an uncertain-negative flag. [Negatome](https://mips.helmholtz-muenchen.de/proj/ppi/negatome/).

Positive–unlabeled learning is a conditional experiment, not the default replacement for BCE. Non-negative PU estimators address a real estimation problem, but their class-prior and positive-sampling assumptions do not follow from having an incomplete, selectively studied interactome. Require a prior/observation-model sensitivity analysis before promoting it. [Kiryo et al., 2017](https://papers.nips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html).

**Falsification:** reject the data package if it helps only the new source's own sampling distribution, relies on duplicate evidence or held-out exposure, or turns a binary-binding model into a co-complex classifier without reporting that change. Usable new pair counts, independent holdout size and source coverage remain unknown until curation is done.

**Priority 5: introduce a real interaction-learning signal before undertaking a larger encoder redesign.** Binary pair labels provide little direct supervision about which residues matter. A bounded auxiliary task using experimentally supported human/model-organism interfaces is more defensible than assuming that attention will automatically become a contact map.

Start with a compact partner-conditioned residue interface task, masked to resolved and mapped residues from biological assemblies. Separate related proteins/complexes across development boundaries. Report homomeric and heteromeric strata, and compare interface supervision against an auxiliary monomer task with comparable compute. This tests whether the gain comes from relational information rather than simply more training.

The new SMP result motivates an additional, higher-risk proxy task using monomer-derived fragment relationships. My preferred experiment would treat these as auxiliary structural relationships, followed by real PPI supervision, rather than append every fragment pair as a certain biological interaction. Covalent continuity, shared sequence origin and artificial boundaries can create easy shortcuts. Exclude homologues of evaluation proteins, include suitable origin/length controls, and require transfer to real held-out pairs. This is our proposed adaptation of the idea, not a reproduced SMP result.

Do not generate a large pseudo-labeled corpus from predicted complex confidence alone. A confident fold is not automatically a validated interaction; structural data also represent a selected population. Reim et al.'s contact-map analysis is a further reason to require explicit contact validation before an interpretability claim. Its small structural comparison and embedding-based model scope do not prove that all sequence models are incapable of learning interfaces.

Chain-aware attention is the next architectural experiment if readout or auxiliary evidence supports it. The qualified v2 implementation retains within-chain rotary scores, removes the artificial cross-chain sequence distance, and uses one joint softmax. It is a specific intervention inspired by paired encoders, not a complete PPLM reproduction. Compare it alone against the same objective/readout before combination.

**Falsification:** stop if auxiliary-task accuracy improves without PPI transfer, if benefits rely on similar held-out structures, or if runtime grows without a commensurate development benefit. A contact map or attention visualization is not sufficient evidence for outperforming native.

**The broader brainstorming pool, with explicit triage.** The table keeps plausible alternatives visible without turning every idea into a job. “Now” means part of the first decision sequence; “conditional” means evidence is needed first. Most are hypotheses, and the priorities reflect expected information gained per unit of research effort, not estimated probabilities of success.

| Idea | Proposed value | Decision / decisive concern |
| --- | --- | --- |
| Clean BCE baseline | Best local selected test model; simpler supervised signal | Now; retain as a reference, not a proven universal objective |
| Lower backbone LR | Reduce excessive adaptation | Now; same objective, schedule and readout control |
| Head-first native adaptation | Start from the behavior we want to improve | Now in the practical branch; exposure restrictions apply |
| Residual residue readout | Expose contextual residue evidence beyond CLS | Now; require matched CLS-MLP control |
| Clean BCE × capped coverage | Resolve an unmeasured interaction | High-information optional historical run |
| Three protein-group development folds | Reduce dependence on one validation distribution | Now; fresh fold-specific controls |
| Frozen independent-embedding classifier | Test whether complex joint modeling earns its cost | Now as a cheap diagnostic |
| Native + clean prediction ensemble | Exploit observed complementary rankings | Small development-only pilot; separate ensemble claim |
| Weight anchoring toward initialization | Preserve useful representations during adaptation | Conditional on native/head-first results |
| Low-rank adapters | Constrain the degrees of adaptation | Conditional; not automatically faster or more accurate |
| EMA / selected trajectory averaging | Reduce checkpoint sensitivity | Conditional; one recipe, development selection, resumable state |
| Masked BCE without MLM | Separate corruption from auxiliary reconstruction | Conditional causal follow-up; prepared but untrained |
| Clean BCE plus separate MLM | Retain language regularization without corrupting classification | Conditional; extra forward/backward cost |
| Protein-balanced sampling mixture | Spread supervision across endpoints | Conditional on endpoint diagnostics |
| Same-query ranking auxiliary | Reward distinguishing partners of a protein | Conditional; uncertain-negative and partner-bias controls |
| Rotating tagged negative pools | Reduce memorization of one sampled nonedge set | Conditional; training-only construction, fixed evaluation |
| Evidence-type heads | Separate binary binding and co-complex supervision | Prioritize after metadata mapping |
| Source-balanced training | Avoid dominance by one assay/database | Conditional on source diversity audit |
| Positive–unlabeled risk | Acknowledge unobserved positives | Conditional on defensible assumptions and sensitivity |
| Partner-conditioned residue pooling | Learn which parts matter for the supplied partner | After simple residue readout earns its cost |
| Chain-aware positional attention | Remove an artificial cross-chain distance prior | After cheaper controls; already qualified variant available |
| Late interaction between cached protein features | Better scaling and a different inductive bias | Useful baseline/alternative; hold encoder adaptation constant |
| Domain/window multiple-instance learning | Preserve relevant local evidence for long proteins | Later; whole-pair supervision and full coverage |
| Experimental interface auxiliary | Add localized relational supervision | Bounded later experiment; mapping/exposure audit first |
| Monomer-derived structural proxy | Obtain additional interaction-related training signal | Exploratory; artificial-fragment shortcuts are a major risk |
| PPLM/MINT checkpoint probe | Test whether interaction pretraining supplies missing information | Optional comparator, with exposure and software audits |
| Monomer-structure features | Add geometry without predicting every complex | Later; include missingness/confidence and sequence-only controls |
| Ensemble distillation | Retain complementary behavior in one deployable model | Only after a real ensemble gain; out-of-fold teachers where needed |
| Native-to-adapted weight interpolation | Potential robustness at single-model inference cost | Optional for a shared initialization/architecture; not arbitrary weight averaging |
| Bigger backbone | More representational capacity | Defer: weak evidence that size is our current bottleneck |
| Network/GNN input features | Add graph context | Separate task; strict unseen-protein setting limits usable graph information |
| Text/GO-augmented embeddings | Add functional context | Later; annotations can contain the target interaction or source-study leakage |
| Repeating symmetric-loss training alone | Enforce a reasonable inductive bias | Low priority; common pooled inference already supplies prediction symmetry |
| Positive weight 10 alone | Shift class preference | Low priority after observed results; does not solve ranking |
| More epochs alone | More optimization | Reject as the next default; selected/final curves argue against it |
| Calibration or uncertainty filtering | Improve use of probabilities/abstention | Useful secondary work; cannot substitute for full-coverage AP superiority |

Weight interpolation has precedent in [WiSE-FT](https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html), but our independently trained native and clean checkpoints are not known to lie in a compatible weight-space basin. Test prediction blending first. If blending helps, distillation may be worth its extra work; native-trained teachers must never supply leaked targets to a purportedly clean internal fold.

**A concrete development sequence.** My recommended core campaign consists of an adaptation screen, a readout screen, then a limited confirmation. It does not launch the entire brainstorming table. The native-adaptation route is a separate, bounded practical branch because its initialization has different exposure permissions.

| Stage | Runs / work | Fixed comparison | Decision produced |
| --- | --- | --- | --- |
| 0: evidence and cheap diagnostics | Metadata/exposure audit; fixed cheap baselines; optional cached head probes and blend pilot | Freeze all development/test roles and model-access permissions | Establish whether we can make a valid claim and which mechanism is worth training |
| 1: adaptation | Clean CLS at 2e-5 and 5e-6 on each of 3 existing development folds: **6 new runs** | Same ESM initialization, data, clean objective, schedule, exposures and selection | Does gentler adaptation improve transferable ranking? |
| 2: readout | Under the selected adaptation recipe, CLS-MLP and residue-MLP on each fold: **6 new runs** | Reuse matching Stage-1 CLS-linear controls | Is extra residue access useful beyond head capacity? |
| 3: confirmation | One frozen candidate and its matched control, seeds 2, 17, 42: **6 full official runs** | Same permitted corpus and final protocol | Estimate seed variability and compare the frozen method on reserved evaluation |
| Optional historical interaction | One clean/capped run at seed 2 | Complete the existing coverage/objective table without changing its contract | Decide whether coverage and clean supervision should be combined |
| Optional practical branch | Frozen native head controls, then at most two bounded continuation pilots | Same native initialization and permitted external development panel | Improve the released checkpoint without claiming clean-fold training superiority |

The internal fold row counts are 73,691/17,490, 71,698/18,285 and 71,986/18,515 train/validation pairs. Around 72,000 crossing pairs per fold are excluded. Keep those exclusions; moving crossing pairs back into training would invalidate the both-proteins-unseen evaluation. These folds use detected homology components at 40% identity and 80% bidirectional coverage. They reduce detected close homology, not all shared domains or remote relationships. [Data card](retrain-v2/DATA_CARD.md).

Use the already prepared 6,000-update/1,000-warmup development horizon as the starting matched contract, with 64 physical pairs per global update. Existing official v2 checkpoints cannot replace these newly trained fold controls. A smaller fold changes effective epochs; record that explicitly. If the low-LR candidate is still improving at the final scheduled evaluation, it is inconclusive rather than a demonstrated failure. Any extension must be prespecified for the matched comparison, preserve resume state, and appear in the budget/trial ledger.

Stage 1 establishes an optimization baseline; Stage 2 establishes a readout mechanism. Do not infer that a combination helps just because each ingredient looked promising elsewhere. If a component is only positive on one fold, investigate source/length/family composition before scaling it. Run clean/capped, masking/MLM decomposition or protein-balanced training as subsequent controlled experiments when the specific ambiguity matters, rather than confounding the initial adaptation/readout screen.

For development promotion, retain v2's provisional rule: mean paired fold AP gain at least 0.005, positive on at least two of three folds, no fold AP decrease exceeding 0.01, and no mean AUROC decrease exceeding 0.005. Add a prospective macro-AP guardrail of no mean decrease exceeding 0.005, with its eligible-protein rule frozen. These thresholds are decision rules, not significance tests; they are not retroactively applied to proclaim old models winners. Record every failed arm and checkpoint-selection opportunity.

If no candidate passes, stop the confirmation spend. Move effort to the evidence/partner-supervision branch, or retain native as the best established model. If a candidate passes, freeze its complete recipe before seeds or external labels are inspected. Do not select the best of three test seeds; report every seed and the mean, with the fixed released native checkpoint and the three-seed matched architecture control both visible.

**How to make the next result scientifically stronger.** Keep three evaluation roles explicit:

| Evaluation role | Permitted use | Interpretation |
| --- | --- | --- |
| Historical Bernett test, 52,048 pairs | Report the eventual frozen candidate once under the shared protocol | Historical comparison; already used in research decisions |
| Internal training-derived folds | Tune clean ESM-initialized candidates and compare their matched controls | Development evidence; no native warm starts or native teachers |
| Reserved external human evidence | Freeze protein/evidence partitions before tuning; use separate external development and test groups | Independent confirmation only after exposure auditing |

The external panel should include both-proteins-unseen and one-protein-unseen cases as separate tasks, with assay/source and first-evidence-date annotations. Human binary-interaction and co-complex strata should be reported separately. Newer source data are not automatically a valid time holdout; older studies, database propagation and pretrained checkpoints may already expose those labels.

The current three folds lose considerable data, but reconstructing them solely to improve scores would introduce another tuning choice. If better retention is needed for a new corpus, use similarity-aware split optimization with explicit constraints and an independent audit. DataSAIL and its 2026 addendum provide relevant methods and comparisons; minimizing aggregate similarity is not identical to enforcing a hard maximum identity threshold. Do not silently replace the historical benchmark. [DataSAIL](https://www.nature.com/articles/s41467-025-58606-8), [addendum](https://www.nature.com/articles/s41467-025-67495-w). [SpanSeq](https://academic.oup.com/nargab/article/6/3/lqae106/7734174) also supports controlling similarity during development, not only at the final test.

Unsupervised exposure to evaluation sequences through ESM-2, supervised exposure to PPI pairs, and homology overlap are distinct audit columns. Using the same ESM initialization in candidate/control helps make their incremental comparison interpretable; it does not establish generalization to sequences absent from pretraining. For released interaction-pretrained checkpoints, unknown exposure should remain a limitation, not be declared clean because downstream train/test files are disjoint.

Retain pooled full-coverage AP as the primary endpoint. Report AUROC, macro partner AP/AUROC with eligible counts, precision/recall at frozen development operating points, Brier/reliability, length/evidence strata, and complete prediction coverage. Macro AP here refers to sampled partner lists; a proteome-wide retrieval claim needs an explicitly defined candidate universe. No confidence-based abstention should silently remove difficult pairs from the primary calculation.

For prevalence sensitivity, report the consequences of a prespecified alternative candidate distribution separately. A balanced PPI benchmark does not identify real-world interaction prevalence. The relationship
\[
\operatorname{precision}=
\frac{\pi\,\mathrm{TPR}}{\pi\,\mathrm{TPR}+(1-\pi)\,\mathrm{FPR}}
\]
illustrates why precision falls when positives become rare, assuming the conditional rates transfer. It is not a license to estimate \(\pi\) from our 1:1 test or claim that a reweighted curve equals deployment performance.

Use paired endpoint/homology-group resampling suitable for the final graph, supplemented by assay/publication grouping sensitivity where available. Report seed variation separately; shared edges and overlapping source evidence are not independent observations. The existing 1,000-resample intervals are useful descriptive tools, not an all-purpose uncertainty model. Multiple exploratory comparisons should remain labeled exploratory; reserve one primary candidate/endpoint for confirmation.

My practical target remains **at least +0.01 absolute AP over native under a matched protocol**, supported by a paired uncertainty interval above zero and consistent seed behavior. On the current historical BF16 pooled test this would correspond to approximately 0.700319, but that number is a target, not a forecast. On an independent panel the target is relative to native measured on that panel, not to 0.690319. Require the predefined secondary guardrails too.

Keep precision and orientation policy identical. Reuse existing native/v1 predictions only while their checkpoint, sequence mapping, pair population and inference contract match exactly. A new holdout or materially changed precision protocol requires fresh corresponding baseline inference. Qualify common efficient FP32 inference where practical; otherwise retain a shared BF16 protocol and a numerical sensitivity check before outcomes are opened. Existing reproduction differences are much smaller in aggregate AP than our largest experimental gaps, but individual predictions can move more.

Strictly monotone calibration cannot improve AP. A temperature or threshold adjustment must not be marketed as solving our ranking objective. Conversely, if the operational objective later changes to calibrated probabilities or low-budget retrieval, declare it prospectively rather than replacing the endpoint after seeing results.

**Arrhenius feasibility is established for ordinary fine-tuning, but the budget must be staged.** The completed v2 runs are a stronger basis than the original paper's A100 timings:

| Completed v2 run | Four-GPU wall time | Allocated GPU-hours |
| --- | ---: | ---: |
| Masked/MLM full reference | 20 h 15 m 44 s | 81.05 |
| Capped reference | 11 h 10 m 50 s | 44.72 |
| Positive10 | 20 h 07 m 42 s | 80.51 |
| Clean BCE | 20 h 04 m 49 s | 80.32 |
| Total | Four separate allocations | 286.61 |

These include validation and checkpoint overhead; they are allocated time, not a utilization integral. Removing MLM did not yield a substantial runtime reduction in these actual runs. Capping saved about 44.8% relative to the reference. [Completed-run accounting](retrain-v2/provenance/final-accounting-3130215.txt).

The current read-only snapshot shows interactive node n519 with one GH200 exposed, and the gpu partition allowing three-day jobs with requeue preemption. The driver reports 97,871 MiB for the device named “GH200 120GB”; prior PyTorch qualification found 95.0 GiB available. Budget against measured available memory. Full nodes have four GH200 devices and Arm CPUs. Use the existing ARM64 SIF and its qualified efficient attention path. [Current snapshot](literature/plm-interact/improvement-review-v3/arrhenius-snapshot.json), [NAISS hardware description](https://www.naiss.se/resources/arrhenius-technical-description/).

Baseline longest-pair qualification succeeded at 16,322 training tokens and 39,391 validation tokens. The combined chain-aware/residue/separate-MLM variant took 64.85 seconds on the longest training pair versus 16.57 seconds for reference; that approximately 3.9× ratio combines three changes and must not be assigned to attention alone or extrapolated as an exact epoch multiplier. Its measured reserved peaks were below the device budget, but not a proof about all distributed production workloads. [Full-model qualification](retrain-v2/QUALIFICATION.md).

| Proposed work | Planning GPU-hours | Basis / limit |
| --- | ---: | --- |
| CPU evidence/score audits | No new GPU allocation needed | Curation/homology CPU costs are separate and data-dependent |
| Cached feature/head and numerical pilots | 10–20 | Allowance, not measured throughput; first profile actual export/head path |
| Stage 1: 6 development runs | 150–270 | 25–45 per 6,000-update run; scaling measured full-run cost gives roughly 38 |
| Stage 2: 6 development runs | 150–270 | Same planning envelope for modest readouts; profile overhead |
| Stage 3: 6 full confirmation runs | 480–600 | 80–100 each for baseline-sized full fine-tuning |
| Final inference/audits | 10–25 | Allowance; depends on external panel and precision |
| **Core staged campaign** | **800–1,185** | Spent only as earlier stages pass |
| Clean/capped optional run | 45–60 additional | Extrapolated from capped masked run; objective/runtime may differ |
| Native continuation pilots | 40–80 additional | Bounded pilot allowance, not a forecast of convergence |
| Structural proxy / new paired pretraining | Not yet estimated | Requires usable-data counts and representative throughput measurements |

For allocation planning, approximately **1,000–1,500 GPU-hours including contingency** is a reasonable envelope for the core campaign. It excludes optional branches and large data/pretraining expansion. The first adaptation decision costs roughly 150–270 GPU-hours plus small pilots; a failed screen should save most of the remaining budget. These are proposals, not resource reservations.

A single full run fits comfortably within the current 72-hour partition limit. Four simultaneous four-GPU jobs require four nodes and sixteen GPUs; available queue capacity and project allocation are separate from technical feasibility. Remaining project GPU allocation was not established. The local filesystem reported about 7.1 TiB available at capture; that is shared filesystem availability, not a verified personal quota reservation.

Keep roughly 0.6–1 TiB of working headroom for a campaign with retained optimizer checkpoints, frozen releases, score files and optional feature caches; external structures can add substantially more. Existing resumable payloads are approximately 7.82 GB each, with best/latest/fallback retention commonly around 23.5 GB per run. EMA or teacher state changes these numbers. Hard-link immutable benchmark selections when appropriate rather than copying large payloads unnecessarily.

Adapters, novel attention kernels and third-party paired models need ARM64/SIF compatibility tests before scheduling. Do not assume x86 wheels, quantization libraries or a public A100 speedup work unchanged on GH200. The current baseline already uses efficient attention and activation checkpointing, so published gains over eager implementations are not additive to our measured runtime.

**Resumability must include the new scientific state.** Reuse the tested v2 atomic checkpoint, checksum, fallback, pending-validation and SLURM requeue design. In addition to model/optimizer/RNG and sampler cursor, save adaptation stage and frozen/unfrozen parameter groups, anchor-checkpoint identity, adapter/EMA parameters, evidence/sampling weights, negative-pool version and cursor, fold assignments, and the model-selection ledger where relevant. A resumed head-first run must not restart with the wrong optimizer or unfreezing stage.

Requalify interrupted-versus-uninterrupted execution for any changed objective, sampler or staged optimizer, including an interruption during validation. Preserve the invariant that held-out labels cannot enter mining or teacher targets. Resume only under the same frozen data/code/container/world-size contract unless a separately validated migration is intended. The existing v2 recovery tests are valuable infrastructure, not blanket validation of new code. [Resume protocol](retrain-v2/PROTOCOL.md), [NAISS job guidance](https://hpc.pages.naiss.se/training/NAISS_Slurm/jobscripts/).

**The decision I would fund first.** Run the clean-objective adaptation comparison on the three existing development folds, preceded by cheap representation/readout diagnostics. In parallel, prepare an exposure-audited external panel and a small native-preserving head pilot. If lower LR and residue access survive their controls, confirm that combination; if they do not, move to evidence/partner supervision before buying a larger backbone or another long pretraining campaign.

I would keep the missing clean/capped run as the most economical way to close a specific ambiguity in v2. I would keep an independently selected native-plus-clean ensemble as a practical comparator, reporting its extra inference cost. I would not call either a scientific breakthrough without independent confirmation. The evidence supports several worthwhile next experiments; it does not support a promise that native can be surpassed by any particular one.

**Review artifacts and scope.** This turn created this proposal, [saved-score and training-log diagnostics](literature/plm-interact/improvement-review-v3/README.md), a [primary-source inventory](literature/plm-interact/improvement-review-v3/SOURCES.md), selected locally archived articles/code with [retrieval hashes](literature/plm-interact/improvement-review-v3/sources/retrieval-manifest.json), and the current read-only HPC snapshot. No production retraining, new PLM inference, checkpoint reselection, or fitted ensemble was performed. The new CPU analyses describe already observed results and do not create an independent evaluation.
