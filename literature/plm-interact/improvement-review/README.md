Evidence supporting [improvment-proposal.md](../../../improvment-proposal.md), collected on 28 September 2026.

The original article, supplementary information, reporting summary, peer-review responses and prior reproduction artifacts remain in their existing locations. This directory contains new feasibility evidence only.

- `arrhenius-resources.json`: read-only Slurm, GPU, software and project storage observations. `collect_resources.py` records the commands. The project-usage output retains only the total and accounting reliability warning; it does not establish a remaining allocation.
- `synthetic-profile.json` and `synthetic-long-profile.json`: full-model synthetic forward/backward timings and GPU memory. Both record zero optimizer steps, no biological sequence data, finite-gradient checks in the script and equality of every model state tensor to the initial checkpoint at completion.
- `profile_feasibility.py`: standalone profiling script, used with the existing ARM64 SIF. It does not instantiate an optimizer or write a model checkpoint. Profiling uses the released humanV11 model to instantiate the architecture; a future scientific retraining must start from original ESM-2 weights.
- The corresponding `.log` files retain the runs' raw console output.
- `budget-scenarios.json`: arithmetic underlying the proposal's explicitly assumed throughput/scaling scenarios. It is a planning calculation, not measured distributed throughput.

Commands executed from the project root:

```bash
python literature/plm-interact/improvement-review/collect_resources.py
bash plm-interact-reproducability/scripts/container.sh python literature/plm-interact/improvement-review/profile_feasibility.py
bash plm-interact-reproducability/scripts/container.sh python literature/plm-interact/improvement-review/profile_feasibility.py --extended-lengths
```

The profile uses artificial masked tokens, BF16 autocast, FP32 weights/gradients, native eager ESM attention, one warm-up and three measured passes per configuration. Tokenization, data loading, gradient accumulation across microbatches, clipping, optimizer-state allocation, optimizer updates, DDP/NCCL communication and model checkpoint I/O are absent from the timings. The in-memory weights remain unchanged. The small FP32 shared-pass qualification is a numerical check on one synthetic fixture, not a convergence-equivalence result.

All resource profiles ran within the already allocated single GPU on `n423`; no new Slurm job or training run was submitted. Inference parallelism established in the earlier reproduction does not establish DDP training scalability.
