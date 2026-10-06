**PLM-interact retrain-v1**

This folder contains two controlled, fully resumable PPI training runs: `symmetric-seed2`, implementing pooled symmetric classification, and `reference-seed2`, providing a matched orientation-wise control. Each trains the complete useful ESM-2-650M backbone from its original pretrained weights for five epochs on 163,085 audited human Bernett pairs. Both retain full sequences. [PROTOCOL.md](PROTOCOL.md) fixes the scientific comparison, settings, limitations and data provenance.

The production job IDs and submission times are recorded in [provenance/submitted-jobs.json](provenance/submitted-jobs.json). Live state is stored separately under each run, so this document does not imply that training has finished or that an accuracy improvement has been established. The existing SIF, original checkpoints and previous reproduction artifacts are preserved.

| Run | Initial Slurm job | Allocation |
| --- | ---: | --- |
| Symmetric objective | 3112401 | 4 GH200 GPUs on one node |
| Matched reference | 3112402 | 4 GH200 GPUs on one node |

At 20:00 UTC on 28 September 2026, both jobs were running and their update-100 checkpoints passed checksum, optimizer-state and batch-position verification. Each saved 538 AdamW parameter states covering 651,043,874 trainable parameters, four ranks' random states, and a cursor after 6,400 training pairs. See [launch-verification.json](provenance/launch-verification.json). These jobs run independently of the interactive allocation.

From `/nobackup/proj/disk/theo-storage/personal/jalil/iPIN`, inspect both runs with:

```bash
python retrain-v1/scripts/status.py
```

`runs/<name>/status.json` reports progress; `checkpoint-status.json` reports the last committed update. `events.jsonl` contains all attempts and validation results. Slurm output is appended to `logs/slurm-<jobid>.log`, including requeues. `best.json` points to the validation-selected checkpoint once full validation has completed. `completed.json` appears only when all planned updates and final validation have completed. Each run has 12,745 planned updates.

**Pause and resume**

Request a safe manual pause for one run:

```bash
touch retrain-v1/runs/symmetric-seed2/REQUEST_STOP
```

The trainer finishes its current optimizer update, or interrupts validation between microbatches, then saves all continuation state. Wait for `stopped.json` and for its Slurm job to exit. This pause does not automatically requeue. Replace `symmetric-seed2` with `reference-seed2` to control the reference.

Resume the same frozen run after the previous job has exited:

```bash
rm -f retrain-v1/runs/symmetric-seed2/REQUEST_STOP
sbatch retrain-v1/slurm/train.sbatch symmetric-seed2
```

The latest valid complete checkpoint loads automatically. Use the same four-GPU allocation and frozen configuration. If an application or node failure left no manual stop request, simply resubmit the same command after checking the failure log. Do not create a new output directory to resume, and do not edit frozen code/configuration. The file lock rejects concurrent writers. A completed run exits without training again.

Five minutes before the 48-hour Slurm limit, the launcher requests a checkpoint and requeues the job. At most eight warning-triggered automatic restarts are allowed per job; later manual resubmission remains possible. Application errors are left visible with their complete checkpoints retained, rather than retried indefinitely. Abrupt cancellation or hardware failure resumes from the most recent complete checkpoint and recomputes the intervening work. A checkpoint is scheduled every 100 updates or 15 minutes, checked at update boundaries, and after validation or a graceful stop.

Checkpoint state includes model and AdamW tensors, learning-rate position, epoch and next batch, all rank-specific RNG states, validation selection, and pending validation. SHA-256 checks detect damaged files; retained older checkpoints provide fallback. Partial writes are ignored. Code/configuration/data mismatches and changes in GPU count are rejected. Each production directory includes its own frozen source copy and image/configuration/source checksums. Shared inputs are also checked at startup.

**Evidence before production launch**

| Check | Evidence |
| --- | --- |
| Native/efficient attention comparison; full 16,322-token training backward and 39,391-token validation forward | [attention.json](qualification/attention.json) |
| Complete five-epoch row coverage, label/mask alignment, uneven gradient accumulation | [data-invariants.json](qualification/data-invariants.json) |
| Full 650M single-GPU stop before final validation, then recovery of final validation and identical complete state | [single-v3-resume-comparison.json](qualification/single-v3-resume-comparison.json) |
| Full 650M four-GPU interruption via batch USR1, real Slurm requeue onto another node, then bitwise-identical model, AdamW, counters and RNG states | [ddp-v3-resume-comparison.json](qualification/ddp-v3-resume-comparison.json) |
| Corrupted checkpoint/pointer fallback, ignored incomplete file, rejected incompatible or lost state | [checkpoint-faults.json](qualification/checkpoint-faults.json) |

After launch, [matched-initialization.json](qualification/matched-initialization.json) additionally verifies that both production runs started with identical model tensors, optimizer state, random states and counters. Their scientific fingerprints differ deliberately by classification objective and run name.

The qualification datasets are deliberately small and their AP values are not scientific performance estimates. The four-GPU test includes an uneven last batch with one empty rank. An earlier default-DDP attempt exposed topology/bucket-dependent rounding after resumption; the production implementation uses fixed rank-order summation, and the final qualification passed exactly. Earlier failed development logs remain available rather than being presented as successful tests.

**Compute and scientific scope**

Each production job uses one node, four GH200 GPUs, 72 CPU cores and 400 GiB requested host memory under account `naiss2025-3-10-gpu`. The devices expose about 95 GiB each despite their 120GB product name. The jobs have a 48-hour allocation per attempt and stop after their fixed five-epoch budget. A post-update checkpoint is about 7.82 GB; the last two and the best are retained, normally at most about 23.5 GB per run, with transient space needed during writes. Qualification artifacts consume additional space.

The wider proposal's database curation, new temporal holdout, alternative negatives, seed replication, no-MLM and two-encoder controls are deferred. This launch compares two objectives on an explicitly audited existing benchmark. Full fine-tuning, symmetry and sequence coverage are implemented; a stronger data package and a robust accuracy claim still require further experiments. The trainer does not score the test set. Original benchmark negatives remain unreported interactions rather than verified negatives.

The files live on `/nobackup`. Atomic checkpoints protect against interrupted writes and jobs, not loss of the underlying filesystem. Exact continuation is qualified for this software image, four ranks and GH200 hardware; arbitrary changes to hardware or software are outside that numerical guarantee.
