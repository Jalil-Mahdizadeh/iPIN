# Recovery of TRAIN/DEV negative sampling

The first validation MILP reached its two-hour limit while solving the root relaxation, without an integer incumbent. Its invalid zero-pair result was rejected, and the old driver cancelled training. The positive partition, completed ILP test, homology checks and frozen candidate pools remain unchanged. Their hashes are checked before this recovery starts.

This recovery changes solver execution and the algebraic formulation, preserving the original feasible selections and normalized degree/mean-GO objective. It uses HiGHS 1.15.1 directly because the installed CVXPY adapter only forwards a cached previous solver solution; setting a fresh CVXPY variable value would not supply our initial selection.

For each protein, the equivalent formulation is `negative_degree - above + below = positive_degree`, with `0 <= above <= 5*positive_degree` and `0 <= below <= positive_degree`. These imply the original `negative_degree <= 6*positive_degree` cap. At the optimum, the sum of these deviations gives the same absolute degree penalty as the original formulation. The GO equation uses the sum of candidate Jaccards, rather than their mean, with its objective coefficient divided by the negative count. A common factor of one million scales the whole objective, preserving its minimizers. All reported objectives and bounds are converted back to original units.

This reduces the validation constraint matrix from 18,072 rows to 6,025, and training from 39,333 to 13,112. It also avoids the very small GO matrix coefficients in the original formulation. The MIP relaxations use the interior-point algorithm, with eight requested threads and parallel search enabled. No candidate-pool, objective-weight, biological exclusion, annotation or partition change is made.

Before solving, a deterministic feasible selection is independently checked and saved. Its full binary selection and deviation variables are passed to HiGHS as an explicit MIP start. Every improving integer solution is checked against the hard constraints and saved as an immutable selection file, followed by an atomic manifest commit. The best valid selection cannot be overwritten by an empty or fractional return value. The original degree/GO objective is recomputed directly for every saved selection.

The new runtime policy supersedes the original two-hour attempt for the two unfinished samplers only:

- At most **30 minutes per split**, with both splits running concurrently.
- Stop after **five minutes without an improved verified incumbent**.
- Independent process watchdog with a bounded grace period.
- An unchanged diagnostic initialization is not published as a final dataset unless the solver establishes the requested optimality tolerance.
- A failure in one sampler preserves the other sampler's progress.

The complete policy and code/input hashes are in [the recovery contract](../provenance/recovery-v1/contract.json). Interrupted branch-and-bound trees are not restored; the checked incumbent is restored. The time budget is accounted across attempts. An attempt terminated without a final accounting record is charged conservatively against that budget, preventing an unlimited retry loop.

The bounded qualification compared every feasible selection on a small instance with an independent objective calculation and exhaustive optimum. HiGHS accepted the explicit start. A forced interruption preserved a feasible solution, reloading recovered it exactly, and an empty selection was rejected. See [qualification](../reports/recovery-v1-qualification.json).

Launch/resume from the project directory on an allocated compute node:

```bash
python data-preparation-v5/recovery-v1/launch.py
```

Progress is recorded in `../pipeline-state.json`, `../logs/recovery-v1-{train,val}.log`, and `../work/recovery-v1/{train,val}/heartbeat.json`. Each split's `best.json` identifies its checked selection and original-unit objective components. The final outputs retain the planned paths under `../prepared`; publication of `../completed.json` requires the independent final data checks to pass. No model training or test-model inference is part of this recovery.

Recovery completed successfully on 2026-10-02. HiGHS explicitly accepted both MIP starts and improved their original normalized objectives:

| Split | Solver seconds | Starting objective | Final objective | Actual relative gap |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 95.85 | 0.10196348 | 0.00183053 | 0.3362% |
| DEV | 17.45 | 0.10219640 | 0.00035199 | 0.0438% |

Both solves returned `kOptimal` within the configured 1% relative tolerance, not a proof of exact optimality. The final independent audit passed, both backbone exports were verified, and all 23 completion-manifest file hashes matched. The original candidate pools, positive partition and completed test data were preserved. The earlier positive-partition gap of 49.41% and test-negative gap of 2.98% remain unchanged and are reported separately. See [the final report](../REPORT.md) and [completion manifest](../completed.json).

Implementation references: [HiGHS Python model interface](https://ergo-code.github.io/HiGHS/stable/interfaces/python/example-py/), [solution callbacks](https://github.com/ERGO-Code/HiGHS/blob/master/docs/src/callbacks.md), and [solver options](https://ergo-code.github.io/HiGHS/dev/options/definitions/).
