**PLM-interact benchmark v2**

This benchmark compares the best checkpoint from every completed initial v2 model with released native PLM-interact and the v1 reference and symmetric models. V2 reference, Positive10 and Clean BCE use update 4,000; the capped model uses update 8,000. All seven identities are fixed in [selection.json](provenance/selection.json).

Fresh inference for all four v2 models completed successfully as SLURM job **3172100** on one four-GPU node. The full seven-model comparison is in [REPORT.md](REPORT.md). Native and v1 predictions are reused after checkpoint, input, prediction and protocol checks. [Protocol](PROTOCOL.md), [qualification](provenance/qualification.json), and [reuse verification](provenance/reuse-verification.json) describe the scope.

From the workspace root:

```bash
python benchmark-v2/scripts/status.py
sacct -j 3172100 --format=JobID,State,ExitCode,Elapsed,NodeList
```

The following commands regenerate the verified comparison and report from the completed predictions:

```bash
bash benchmark-v2/scripts/container.sh python benchmark-v2/scripts/analyze.py
bash benchmark-v2/scripts/container.sh python benchmark-v2/scripts/write_report.py
```

The final report is `REPORT.md`; predictions, full metrics, paired intervals and plots are in `results/`. The primary endpoint is full-length, pooled-logit AP on all 52,048 historical Bernett test pairs. This is an exploratory comparison on a previously inspected test set.

To resume interrupted inference, resubmit `benchmark-v2/slurm/benchmark.sbatch` after confirming the earlier job has ended. The same frozen four-GPU command verifies and reuses committed chunks; exclusive per-rank locks prevent simultaneous writers. Keep the same configuration, checkpoint hashes, code and shard count. The existing checkpoint hard links preserve these selections independently of training retention. Do not rerun `prepare.py` to change this benchmark's model identities.
