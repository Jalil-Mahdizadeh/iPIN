# Single positive-partition attempt: outcome

**Complete. Keep the existing dataset: no additional positives were retained.**

| Measure | Previous completed partition | This single attempt |
| --- | ---: | ---: |
| Retained positives | 433,253 | 433,253 |
| TRAIN positives | 350,382 | 350,382 |
| DEV positives | 82,871 | 82,871 |
| Discarded crossing positives | 287,829 | 287,829 |
| Relative solver gap | 49.4061% | 47.1791% |
| Solver time | 7,200 seconds | 1800.08 seconds |

The attempt used a compact equivalent formulation, the completed partition as an explicitly accepted feasible start, eight requested CPU threads and the original eligible positives, protein groups and balance constraints. It visited 73,792 branch-and-bound nodes. The gap target of 1% was not reached; the remaining gap is not evidence that a particular amount of additional retention is achievable.

The deadline callback stopped the solver at the 30-minute cap. Its `kInterrupt` / `Interrupted by user` status refers to this programmed budget stop, not an unexpected failure or a new user cancellation. The driver exited successfully and all attempt processes finished. Only the existing interactive allocation was used; no additional SLURM or GPU job was submitted.

The prespecified adoption criterion required at least 437,586 retained positives (a 1% increase), no reduction in training positives, and passing integrity checks. No retention improvement was found. Tightening the mathematical bound alone does not change the dataset.

Independent post-run review recounted all 721,082 eligible rows, confirmed that every protein has exactly the original split assignment, verified all 12 frozen code/input identities and all 23 completed export hashes, and confirmed the original completion manifest is byte-for-byte unchanged. TRAIN still has 700,764 balanced pairs and DEV 165,742. The original and ILP tests, sampled negatives and both backbone exports are unchanged, so no downstream regeneration is needed.

No further partition attempt, model retraining or benchmark inference was started. This bounded search does not prove the current partition globally optimal or establish any model-performance gain.

Evidence: [solver result](result.json), [independent review](review.json), [frozen contract](contract.json), [solver log](highs.log), [original dataset completion manifest](../completed.json).
