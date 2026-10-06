**Primary sources supporting proposal v3 — consulted 30 September 2026**

This was a broad, targeted research search, not a preregistered systematic review or an exhaustive survey. Search families covered PLM-interact follow-ups; Bernett/HIPPIE comparability; PPI shortcuts and split design; residue-level interaction representations; transfer learning and adapters; negative-label assumptions; human interaction resources; and Arrhenius execution. Search-engine dates were not treated as publication dates. Only primary papers, author repositories and official resource documentation support technical claims in the proposal.

“Full-text sections” below means the relevant sections were examined, not that every associated supplement was reread. For the original PLM-interact work, the present review also uses the earlier local reproduction and source-data audits. The Nature Machine Intelligence paper was accessible as a publisher abstract/figure overview; its conclusions are not represented as a full-text reanalysis.

| Source | Status by review date | Material examined / use |
| --- | --- | --- |
| [PLM-interact](https://www.nature.com/articles/s41467-025-64512-w) | Nature Communications, 2025 | Local article, relevant Methods/Bernett results; source-data audit and archived trainer from prior work |
| [PLM-interact supplementary information](../supplementary-information.pdf) | Published supplement | Local captions/tables; masking, symmetry, interface-confidence and mutation-epoch interpretation |
| [PLM-interact reporting summary](../reporting-summary.pdf) | Published associated document | Dataset/negative-label definitions and availability |
| [PLM-interact peer review](../peer-review.pdf) | Published associated document | Bernett five-epoch response and hyperparameter-selection exchange |
| [Bernett et al., Cracking the black box](https://academic.oup.com/bib/article/25/2/bbae076/7621029) | Briefings in Bioinformatics, 2024 | Previously archived source and dataset audits; degree-controlled negatives and benchmark provenance |
| [Reim et al., Deep learning models for unbiased sequence-based PPI prediction plateau at an accuracy of 0.65](https://doi.org/10.1093/bioinformatics/btaf192) | Bioinformatics, 2025 | Abstract, full-text discussion, author repository model/data scope; XML retained |
| [Bernett et al., Are You Learning Biological Signal or Shortcuts?](https://arxiv.org/abs/2609.10193) | Preprint v1, 9 September 2026 | Full-text results/methods on splitting and negative sampling; HTML retained |
| [PPLM](https://www.nature.com/articles/s41467-026-70457-5) | Nature Communications, first online March 2026 | Full-text architecture, training and PPI ensemble methods; XML retained |
| [MINT](https://www.nature.com/articles/s41467-025-67971-3) | Nature Communications, January 2026 | Full-text Bernett results/task table, training discussion from archived v2 review |
| [SMP](https://www.nature.com/articles/s41467-026-73885-5) | Nature Communications, May 2026 | Full-text results/methods and dataset counts; XML retained |
| [SMP-PPI author implementation](https://github.com/Split-and-Merge-Proxy/smp-ppi) | Author repository | README, tree, preparation script and AP implementation; sampled code pinned to commit a4828b9a01510dffec9fcc66fc9958605355a616 |
| [TUnA](https://escholarship.org/content/qt9q42r5vh/qt9q42r5vh.pdf) | Briefings in Bioinformatics, 2024 | Prior comparator review plus targeted uncertainty/calibration material |
| [PRING](https://arxiv.org/abs/2507.05101) | NeurIPS 2025 Datasets and Benchmarks | Abstract and [author task definitions](https://github.com/SophieSarceau/PRING); prior local review |
| [RaftPPI](https://openreview.net/pdf?id=Dp1RM3gPg8) | ICLR 2026 | Conference-paper abstract and author project description; retrieval/accuracy distinction |
| [Szymborski and Emad, A flaw in using pretrained protein language models in PPI inference models](https://www.nature.com/articles/s42256-025-01176-7) | Nature Machine Intelligence, February 2026 | Publisher abstract and figure overview; upstream exposure limitation |
| [Ahmadian et al., Reliable evaluation and learning in multi-input biological association prediction](https://doi.org/10.1093/bib/bbag376) | Briefings in Bioinformatics, 2026 | Abstract/figure descriptions on entity balancing and scope of applications; full text retained for follow-up |
| [DataSAIL](https://www.nature.com/articles/s41467-025-58606-8) | Nature Communications, April 2025 | Abstract and method/discussion summaries on similarity-aware splitting |
| [DataSAIL addendum](https://www.nature.com/articles/s41467-025-67495-w) | Nature Communications, February 2026 | Full-text comparison with existing PPI splits; distinction between aggregate leakage and maximum similarity |
| [SpanSeq](https://academic.oup.com/nargab/article/6/3/lqae106/7734174) | NAR Genomics and Bioinformatics, 2024 | Abstract and discussion of similarity-aware development; multi-input scope limitation |
| [Kumar et al., Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution](https://arxiv.org/abs/2202.10054) | ICLR 2022 | Abstract, theoretical/model scope and head-first transfer hypothesis |
| [Li et al., Explicit Inductive Bias for Transfer Learning](https://proceedings.mlr.press/v80/li18a.html) | ICML 2018 | Primary abstract and initial-weight regularization principle |
| [Wortsman et al., Robust Fine-Tuning of Zero-Shot Models](https://openaccess.thecvf.com/content/CVPR2022/html/Wortsman_Robust_Fine-Tuning_of_Zero-Shot_Models_CVPR_2022_paper.html) | CVPR 2022 | Abstract and interpolation setting; not a PPI result |
| [Çelik and Xie, Efficient inference, training, and fine-tuning of protein language models](https://pmc.ncbi.nlm.nih.gov/articles/PMC12481099/) | iScience, 2025 | Efficiency and task-dependent LoRA results; engineering implications |
| [Kiryo et al., Positive-Unlabeled Learning with Non-Negative Risk Estimator](https://papers.nips.cc/paper/2017/hash/7cce53cf90577442771720a370c3c723-Abstract.html) | NeurIPS 2017 | Primary abstract and risk-estimation motivation; conditional proposal only |
| [HuRI](https://www.nature.com/articles/s41586-020-2188-x) | Nature, 2020 | Study abstract, author resource and downloads; direct binary evidence source |
| [BioPlex](https://doi.org/10.1016/j.cell.2021.04.011) | Cell, 2021 | Study abstract and network context; complementary co-complex evidence |
| [IntAct metadata protocol](https://doi.org/10.1002/cpz1.70018) | Current Protocols, 2024 | Author abstract/resource description; metadata feasibility |
| [Negatome](https://mips.helmholtz-muenchen.de/proj/ppi/negatome/) | Author database resource | Resource definition and context-specific noninteraction evidence |
| [Arrhenius hardware](https://www.naiss.se/resources/arrhenius-technical-description/) | Official NAISS documentation, current access | GPU count, architecture and memory; cross-checked with live local tools |
| [NAISS SLURM guidance](https://hpc.pages.naiss.se/training/NAISS_Slurm/jobscripts/) | Official documentation, current access | Scheduling context; current local partition queried directly |

Novelty claims were intentionally limited. Several useful readout/attention/data ideas were already proposed in v2 but never production-tested. Newer papers motivate hypotheses; reported results on different splits, ensembles, candidate sets or task definitions are not treated as comparable benchmark wins.

Selected source payloads and author-code samples are in [sources/](sources/), with URLs, retrieval times and SHA-256 values in [retrieval-manifest.json](sources/retrieval-manifest.json). Full-text XML came from Europe PMC when browser access to the publisher/PMC was unreliable. The [selected discovery record](sources/discovery-record.json) is a partial search trace, not a record of every query or an endorsement of every search result.
