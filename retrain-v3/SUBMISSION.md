# V3 production submission

Submitted all six Stage-1 runs as array **3177489** with a concurrency limit of **six** and **four GPUs per task** (24 GPUs maximum).

Status checked at 2026-09-30T11:16:17.394095+00:00: **3 running, 3 pending**. Pending tasks are waiting for SLURM resources; the array throttle permits six.

| Task | Configuration | State | Node | Last logged update |
| --- | --- | --- | --- | ---: |
| 3177489_0 | clean-cls-linear-lr2e-5-fold0-seed2 | RUNNING | n577 | 10 |
| 3177489_1 | clean-cls-linear-lr2e-5-fold1-seed2 | RUNNING | n71 | 20 |
| 3177489_2 | clean-cls-linear-lr2e-5-fold2-seed2 | RUNNING | n102 | 10 |
| 3177489_3 | clean-cls-linear-lr5e-6-fold0-seed2 | PENDING | waiting | 0 |
| 3177489_4 | clean-cls-linear-lr5e-6-fold1-seed2 | PENDING | waiting | 0 |
| 3177489_5 | clean-cls-linear-lr5e-6-fold2-seed2 | PENDING | waiting | 0 |

Running workers inspected so far have four distinct GPUs, the expected fold counts, finite logged losses and no detected startup errors. This is an early startup check, not a guarantee of the full run.

The user explicitly authorized all six simultaneous runs. The frozen launcher first verified the release, input/initialization manifests, SIF and qualification reports; its verified `--array=0-5%4` plan then received the scheduling-only override `--array=0-5%6`. The exact command and source-plan hash are in [submission provenance](provenance/submission-3177489.json).

Training uses the qualified immutable release and existing checkpoint/requeue recovery. All later readout/confirmation stages remain gated.

[Startup snapshot](provenance/startup-3177489.json). For current training progress: `python retrain-v3/scripts/status.py`. For current queue state: `squeue -r -j 3177489`.
