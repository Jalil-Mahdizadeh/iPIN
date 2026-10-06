# V3 benchmark — prepared, deferred

Benchmarking is **not running and is not scheduled automatically**. The user requested retraining only for the current task; benchmarking will be conducted later in this separate folder.

Prepared artifacts include inference/analysis code and checksum-verified native, v1 and v2 baseline predictions. No production v3 checkpoint has been selected for testing and no new v3 test inference has run. The earlier end-to-end benchmark fixture was a software check using cached predictions, not a v3 model result.

The active training release is `../retrain-v3/releases/20261001-final-training/`. It ends after training. See the [current training protocol](../retrain-v3/FINAL_PROTOCOL.md).
