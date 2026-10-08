# Vx v2 status

Completed and independently checked. SLURM job **3516311** exited successfully (`0:0`), with all **8,000/8,000** feature records complete.

Scientific decision: **no_go_or_inconclusive**. Local true-MSA assessment AP is **0.655735**, versus **0.662517** for local shuffled MSA and **0.664361** for the original global true-MSA model. The primary pairing-gap improvement is **-0.007896**, 95% interval **[-0.023482, +0.007892]**. Neither a pairing benefit nor a spatial-coherence benefit was established.

The independent CPU audit verified the frozen inputs, record checksums, unchanged eligibility, TRAIN-only scaling, calibration choices, saved predictions, metrics, exact baseline fallback and primary bootstrap intervals. No heads were refitted and no TEST input was accessed.

Read the [completion review](review-20261008/REVIEW.md), [frozen-analysis report](results/REPORT.md), [verification record](review-20261008/verification.json) and [final resource accounting](results/resources.json).

Runtime was **2:01:01**, consuming **8.0678 allocated GPU-hours**. Cumulative charge, including qualification/audit reservations, is **21.9611 of 24 GPU-hours**. This ablation stops as planned; no additional experiment or production continuation has been launched.
