# V3 readout screen: authorized continuation after the early LR decision

On 2026-09-30 the user selected the 2e-5 learning rate and authorized ending the LR screen and advancing. All six adaptation workers stopped at safe optimizer boundaries; none completed its originally planned 6,000 updates. [The decision](decisions/adaptation-early.json) records actual stop positions, every validation artifact, the protocol deviation, and verified best/latest checkpoint hashes. Best mean AP was 0.616488 for 2e-5 versus 0.600486 for 5e-6; all three folds favored 2e-5 under best-checkpoint selection.

## New scientific comparison

Train two clean-BCE residual readouts on each of the same three protein-disjoint development folds, at learning rate **2e-5** and seed 2:

| Readout | Runs | Information added to the baseline classifier |
| --- | ---: | --- |
| CLS-MLP | Three folds | Extra nonlinear capacity using the contextual CLS representation |
| Residue-MLP | Three folds | CLS plus the sum, absolute difference and elementwise product of the two residue means |

Both heads add **655,489 parameters**. Both residual outputs start at zero and preserve the initial baseline logits. Special tokens and padding are excluded from residue pooling. Each run initializes independently from the pinned ESM2 checkpoint; the stopped supervised controls do not initialize the new heads. Both heads receive full backbone fine-tuning, the same full-length pairs, both orientations, and clean BCE without masking or MLM. No additional data, native warm start or historical test inputs are introduced.

Reuse the three stopped **2e-5 CLS-linear** runs as controls. This tests capacity versus the original readout, then residue information beyond matched capacity. Data, sampling seed, global batch 64, optimizer, warmup, gradient clipping, precision and inference policy match those controls. The model, data and checkpoint code remain byte-identical. The new trainer changes one DDP option: gradients use independent storage instead of views into communication buckets. This fixes the resume mismatch detected during full-size residue-head qualification. The original controls retain their original code; amended versus original numerical trajectories are not claimed to be bitwise identical.

## Equal selection opportunities and a bounded compute budget

The common comparison window is **updates 1,000, 2,000, 3,000, 4,000 and 5,000** for every control and candidate. Select the maximum pooled AP over those five validations, retaining the earliest exact tie. The comparison window corresponds to 320,000 physical pair exposures and 640,000 encoded orientations.

**The optimizer still uses the original 6,000-update schedule and 1,000-update warmup.** Changing `total_updates` to 5,000 would change every learning-rate value after warmup and confound reuse of the existing controls. The new wrapper instead watches for a committed update-5,000 validation/checkpoint, requests a safe stop, then verifies `stopped.json` and writes `SCREEN_COMPLETE.json`. A race may permit a few extra optimizer updates; the accepted ceiling is 5,005, and selection remains restricted to the five frozen validation points. An unexpected overshoot or monitor error fails visibly for inspection.

`SCREEN_COMPLETE.json` means this screening window finished. It never fabricates the trainer's `completed.json` or claims the 6,000-update schedule completed. A manual stop or scheduler interruption before update 5,000 remains resumable and cannot enter the scientific comparison. The batch script retains array-task-specific requeue on time-limit signals and gives manual stops precedence.

This window was chosen after seeing LR development results, but **before readout production**. It is a recorded adaptive development decision, not a retrospectively completed original protocol. All candidate heads receive the same selection opportunities as the available controls. No cross-validation result here establishes superiority on an independent test population.

## Promotion rules

For each three-fold comparison: mean AP gain >= .005; positive gain on at least two folds; no fold AP drop > .01; mean AUROC drop <= .005; mean protein macro-AP drop <= .005. Macro AP retains the eligibility rule of at least two positive and two negative incident pairs per protein. These are development decision rules, not significance tests.

CLS-MLP must pass against CLS-linear. Residue-MLP must pass against both CLS-linear and matched CLS-MLP. Among passing candidates choose the largest mean AP gain against CLS-linear; exact ties favor CLS-MLP. If neither head passes, stop confirmation spending and reconsider evidence/partner supervision. Native adaptation and three-seed/external confirmation remain gated; they are not part of this submission.

The comparison includes descriptive paired endpoint-bootstrap AP intervals, AUROC, Brier, macro AP and length strata. It is conditional on these graphs and selected checkpoints, and excludes training-seed/model-selection uncertainty. It reads no historical test scores.

## Execution and qualification

The initial adaptation release and its `releases/CURRENT` pointer remain preserved. The new immutable readout release has its own **`releases/CURRENT_READOUT`** pointer and launcher:

```bash
python retrain-v3/scripts/launch_readout.py
python retrain-v3/scripts/status.py
```

The first command verifies the complete release and prints a dry-run plan. `--submit` performs the authorized launch; default concurrency is six, retaining the user's all-at-once preference. Each task requests one four-GPU GH200 node and uses the pinned SIF. Expected screening cost is **125–225 GPU-hours total**, derived from the original 150–270 estimate scaled to the common 5,000-update window; qualification and scheduler overhead are additional.

Resume uses the same release and four GPUs. After an intentional manual stop, inspect its state, remove `REQUEST_STOP` only when continuation is intended, and use `launch_readout.py --run <name> --submit`. A completed screening window cannot be resubmitted by the launcher. Concurrent duplicate readout/adaptation campaigns are refused.

Release requires both **full-size** readout heads to pass actual four-GPU short-input training and bitwise full-versus-resumed comparisons with the amended trainer. Job **3196205** passed CLS-MLP but failed residue-MLP: the step-2 checkpoint, optimizer and validation predictions matched exactly; the first post-restart gradient norm differed by one FP32 rounding unit, followed by divergent updates. The failure and its diagnostics remain preserved. Independent gradient storage is tested in replacement qualification **3196606**, with an AST check that this is the only executable trainer change. A separate GPU diagnostic demonstrates that the same gradient values at different storage offsets can produce slightly different norm reductions. This supports the numerical explanation without claiming a complete kernel-level trace. See [the engineering record](qualification/READOUT_ENGINE.md).

The existing full-length residue-head profile and scientific pooling/capacity tests remain relevant because the model code is unchanged. Independent gradient storage adds approximately 2.42 GiB per GPU; amended short-input DDP peak allocation is recorded in the qualification logs. The wrapper's actual shell, actual monitor and OS signal path are tested with local scheduler/worker doubles, covering normal budget stop, failure, manual stop, requeue, monitor errors and missed-window rejection. This is not a fresh real scheduler-requeue test. The previous signal/requeue evidence applies to unchanged checkpoint and shell mechanisms; new full-size tests specifically cover the changed DDP configuration.

After all six screening windows finish, use the frozen analysis entry point:

```bash
bash retrain-v3/scripts/container.sh python retrain-v3/releases/20260930-readout/code/compare_readout.py
```

It requires complete equal windows, verifies preserved control evidence and prediction hashes, writes `decisions/readout-window5000.json`, and submits no further jobs. The older `compare_development.py --stage readout` belongs to the original full-6,000-update protocol and must not be used for this early-stop amendment.
