# v5 benchmark — complete

V5 does not establish a meaningful improvement over native PLM-interact (Bernett) on the original Bernett test. V5 ESM2 improves on the custom ILP-negative test, meeting the descriptive +0.010 AP / positive paired interval target.

See [REPORT.md](REPORT.md) for all twelve predictors on both tests, confidence intervals and exposure limitations. Machine-readable metrics and per-pair predictions are in [results/](results/).

The three X-PAIR releases (Bernett, default/X-fair, and humanV11) are separate entries. Original D-SCRIPT is included with documented length-safe execution and training-data exposure. SPRINT uses the v5 TRAIN-positive graph. iPIN checkpoint selection was frozen before test analysis; D-SCRIPT, the two original human STRING-trained releases, and X-PAIR humanV11 were added later at the user's request. No retraining occurred.

[Protocol](PROTOCOL.md) · [Re-running and resuming](RUNNING.md) · [Checkpoint selection](provenance/selection.json) · [X-PAIR V11 extension](XPAIR-V11-ADDENDUM.md)
