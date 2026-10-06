**Published native PLM-interact training records — availability audit, 29 September 2026**

Detailed numerical learning histories are available for the separate mutation-effect fine-tuning experiment. I did not find comparable epoch-by-epoch training/validation histories for the main humanV11 PPI model or its Bernett-trained counterpart in the materials checked. This is a scoped search result, not a claim that the authors have no private logs.

| Experiment | Available evidence | Limitation for our current runs |
| --- | --- | --- |
| HumanV11 cross-species PPI | Paper describes ten training epochs and checkpoint selection by human-validation AUPR; final cross-species test scores and prediction files are released. | No per-epoch training/validation metric table or original training log located. |
| Bernett PPI | Peer-review response states five epochs and validation-based model selection. Figure 4 supplies final test performance, approximately AUPR 0.69 and AUROC 0.70. The checkpoint and dataset splits are released. | No per-epoch Bernett validation curve, training loss history, or numerical validation-selection history located. Final test metrics are not validation metrics. |
| Mutation-effect fine-tuning | Supplementary Figure 10 and the source-data workbook provide 40 numbered epochs of loss, precision-recall AUC and AUROC for train, validation and test. | Different task and data; these curves cannot establish the expected Bernett PPI learning trajectory. |

The mutation curves were extracted directly from the publisher's source-data workbook into:

- [Training curve](published-mutation-train-curve.csv)
- [Validation curve](published-mutation-val-curve.csv)
- [Test curve](published-mutation-test-curve.csv)

The original workbook sheet names are `Supplemtrary Figure10_train`, `Supplemtrary Figure10_val` and `Supplemtrary Figure10_test` (publisher's spelling). Every sheet contains 40 rows for epochs 1–40, with the original column names retained. The minimum mutation validation loss is at epoch 19, as stated in the supplement. This is loss-based selection; the largest validation precision-recall AUC in the table occurs at epoch 22. These are published records, not newly generated predictions or training runs.

**Checked sources**

The local paper, all supplementary figure captions, peer-review response, complete source-data archive inventory, all 34 workbook sheet names, and publication-archived training source were checked. Supplementary Figures 1–2 summarize final test performance of hyperparameter variants; they are not learning curves. Supplementary Figure 10 is on PDF page 9.

Current public GitHub file-tree metadata and the humanV11, humanV12 and Bernett Hugging Face model repositories were checked, together with their model cards and the Bernett/cross-species dataset file inventories. All six API inspections succeeded, and no tree was truncated. Exact URLs, revision hashes and file inventories are preserved in [availability-audit.json](availability-audit.json), alongside the returned metadata and pinned model cards. No original `train_loss_resume.csv`, TensorBoard/W&B event history, `trainer_state.json`, or corresponding PPI learning-curve artifact appeared in these inventories.

The archived `train_mlm.py` does contain code that writes `train_loss_resume.csv` with epoch, training-step/update counters and total/classification/MLM losses. The same logging mechanism exists in the currently published source. This establishes logging capability, not release of the authors' actual run files. Its saved training loss values are divided by gradient accumulation before logging, so even obtaining that CSV would require checking its scaling convention before comparison with our per-pair loss.

Primary sources: [paper](https://www.nature.com/articles/s41467-025-64512-w), [supplementary information](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-025-64512-w/MediaObjects/41467_2025_64512_MOESM1_ESM.pdf#page=9), [peer-review response](https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41467-025-64512-w/MediaObjects/41467_2025_64512_MOESM3_ESM.pdf), [public training source](https://github.com/liudan111/PLM-interact/blob/main/PLMinteract/train_mlm.py), and [released Bernett checkpoint](https://huggingface.co/danliu1226/PLM-interact-650M-Leakage-Free-Dataset/tree/main).

Our `reference-seed2` curve is a newly trained corrected reference, not an author-provided historical training log. Evaluating the released Bernett checkpoint on the same validation set could supply a comparable endpoint under a clearly stated inference protocol, but it would not reconstruct the missing original learning history. No such new inference was performed for this availability audit, and the two ongoing training jobs were not modified.
