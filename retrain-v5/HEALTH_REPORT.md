# V5 production startup health

Updated (UTC): 2026-10-02T21:14:12.273245+00:00

Both jobs were submitted at **2026-10-02 22:17:50 Stockholm time**, without dependencies. Each requests a separate node with four GH200 GPUs.

| Model | Job | Scheduler state | Latest observed update | Startup audit |
| --- | --- | --- | --- | --- |
| esm2-native-ilp-seed2 | 3301121 | RUNNING | 20 | passed |
| esmc-native-ilp-seed2 | 3301122 | RUNNING | 40 | passed |

Monitor status: **both startup audits passed**.

A passing audit requires four distinct GPUs, all 700,764 TRAIN and 165,742 DEV rows, at least 20 production updates with finite BCE/MLM losses and gradients, progress across observations, and a hash-verified checkpoint containing model, optimizer, four ranks of RNG and the correct dataset cursor.

The four-GPU resume qualification passed for both models before submission. Production execution health remains pending until this report records a pass. No production validation performance is claimed; the first full DEV evaluation is at update **2,738**.

This read-only monitor runs in the existing interactive allocation, polls once per minute, and ends after both startup audits pass, a failure is detected, or 2026-10-03 09:00 UTC. It launches no additional SLURM jobs and does not change training. Earlier termination of the interactive allocation also ends the monitor. Detailed evidence: [startup-health.json](provenance/startup-health.json).
