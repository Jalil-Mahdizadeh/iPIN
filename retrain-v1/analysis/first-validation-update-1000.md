**First full validation: update 1,000**

Checked on 28 September 2026. All 59,258 validation pairs were scored exactly once in each run. Saved labels match the frozen validation data; all logits are finite. AP, AUROC and Brier scores were independently recomputed from saved predictions and agree with the training logs. No test inference was performed.

| Predictor | AP | AUROC | Brier (lower is better) |
| --- | ---: | ---: | ---: |
| Reference, dataset A–B order | 0.516822 | 0.502928 | 0.249966 |
| Reference, reversed B–A order (diagnostic) | 0.560892 | 0.548279 | 0.249203 |
| Reference, pooled logits | 0.549765 | 0.521258 | 0.249569 |
| Symmetric training, pooled logits | 0.548457 | 0.519055 | 0.249478 |

Validation prevalence is 0.499983. Random-ranking AP is approximately 0.50 and AUROC is 0.50; predicting the prevalence for every pair gives Brier 0.25. Current discrimination is weak and Brier improvement over this constant predictor is small.

The symmetric model exceeds the reference in its original input order, but the matched pooled-inference control is the more informative comparison: symmetric training is lower by 0.001308 AP and 0.002203 AUROC. Its Brier score is lower by 0.0000906. These are descriptive differences from one checkpoint and one seed, without a significance test; they do not establish an advantage for pooled-objective training.

The reversed-order reference scores materially differently from the original-order reference. Both are reported for transparency; the reversed-order diagnostic does not replace the prespecified predictor or model-selection rule. The symmetric model still has different internal orientation logits, while its final pooled prediction is invariant to exchanging the two terms.

At this validation each run had processed 64,000 training pairs, about 39.2% of the first epoch and 7.85% of its planned optimizer updates. The learning-rate warmup is 2,000 updates, so this checkpoint is halfway through warmup. These validation scores should not be compared directly with the paper's final test scores.

Both jobs committed their update-1,000 model/optimizer checkpoints and resumed training. No settings, validation selection rules or data were changed following this analysis. The next scheduled validation is update 2,000, followed by the first epoch boundary at update 2,549.

[Machine-readable audit](first-validation-update-1000.json) contains exact values, coverage checks and score ranges.
