**Evidence supporting improvement proposal v3**

The proposal is [improvment-proposal-v3.md](../../../improvment-proposal-v3.md). New work in this folder consists of CPU analyses of previously evaluated outputs, literature review and a read-only Arrhenius snapshot. It performed no PLM inference, optimizer updates, ensemble fitting, checkpoint selection or SLURM submission.

| Artifact | Purpose |
| --- | --- |
| [saved-prediction-audit.json](saved-prediction-audit.json) | Seven-model AP/AUROC, equal-protein macro metrics, score correlations and fixed top-k overlap; includes source hashes |
| [protein-macro-metrics.csv](protein-macro-metrics.csv) | Individual eligible-protein AP/AUROC for both saved evaluation splits |
| [model-score-correlations.csv](model-score-correlations.csv) | Pairwise score rank correlations |
| [training-window-audit.json](training-window-audit.json) | Periodically logged training losses and pre-clipping gradient norms in two fixed update windows |
| [arrhenius-snapshot.json](arrhenius-snapshot.json) | Current host, queue, partition, GPU memory, filesystem availability and project-account information |
| [SOURCES.md](SOURCES.md) | Primary-source inventory, publication status and actual reading scope |
| [retrieval-manifest.json](sources/retrieval-manifest.json) | URLs and hashes for selected downloaded source material |

The score audit aligns exact row IDs and labels before comparing predictions. Macro AP gives equal weight to proteins having at least two labeled positives and two labeled negatives among their incident benchmark pairs. Pairs can contribute to both endpoints. Counts are 3,049 validation and 2,485 test proteins; their union covers 59,239 and 52,033 respective pair rows. These are incomplete, sampled candidate sets, not whole-proteome partner lists. No independent protein-level significance test is implied.

Top-k comparisons use fixed k values of 1,000 and 5,000, with row index breaking exact score ties. The larger union of two models' predictions is not evaluated as if it had the same prediction budget. Correlation and complementary positives motivate a future development-only ensemble test; no ensemble gain was measured.

The historical test was already observed before this research. New diagnostics are post-hoc and must not be presented as independent confirmation or used to select another old checkpoint. The frozen benchmark's primary endpoint remains full-coverage pooled AP.

To reproduce the score audit from the repository root:

    bash benchmark-v2/scripts/container.sh python literature/plm-interact/improvement-review-v3/audit_saved_predictions.py

To reproduce the training-window summary:

    python literature/plm-interact/improvement-review-v3/audit_training_windows.py

Both scripts write only their derived outputs in this folder. Training summaries use periodically recorded batches, so they are not exact whole-training-set losses. The two windows are updates 3,000–4,000 and 11,745–12,745; their sample counts are recorded.

The resource snapshot records observations, not reservations. Filesystem free space is shared and was not interpreted as a personal quota. Remaining GPU allocation is not established.
