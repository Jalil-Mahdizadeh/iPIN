# V3 experiment ledger at preparation

Status on 2026-09-30: **production unsubmitted; zero production updates.** The machine-readable contract is [configs/campaign.json](configs/campaign.json). Qualification trials are excluded from model-performance comparisons.

| Initial configuration | Learning rate | Train / validation | Horizon | Status |
| --- | ---: | ---: | ---: | --- |
| clean-cls-linear-lr2e-5-fold0-seed2 | 2e-5 | 73,691 / 17,490 | 6,000 | Ready, unsubmitted |
| clean-cls-linear-lr2e-5-fold1-seed2 | 2e-5 | 71,698 / 18,285 | 6,000 | Ready, unsubmitted |
| clean-cls-linear-lr2e-5-fold2-seed2 | 2e-5 | 71,986 / 18,515 | 6,000 | Ready, unsubmitted |
| clean-cls-linear-lr5e-6-fold0-seed2 | 5e-6 | 73,691 / 17,490 | 6,000 | Ready, unsubmitted |
| clean-cls-linear-lr5e-6-fold1-seed2 | 5e-6 | 71,698 / 18,285 | 6,000 | Ready, unsubmitted |
| clean-cls-linear-lr5e-6-fold2-seed2 | 5e-6 | 71,986 / 18,515 | 6,000 | Ready, unsubmitted |

There are six scheduled checkpoint-selection opportunities per run (updates 1,000 through 6,000), 36 across the initial stage. Use the same opportunity count for both learning rates. These folds are used for development, so their selected-checkpoint results are not unbiased final estimates.

## Conditional work

1. Apply the prospective adaptation decision after all six runs finish. Low LR must pass all five gates; otherwise retain the control for the next screen and disclose any horizon limitation.
2. At the selected rate, run three CLS-MLP and three residue-MLP folds. Reuse the corresponding Stage-1 CLS-linear controls. The other rate's six templates remain unused. A new release is required.
3. If an adaptation/readout candidate passes, freeze it and its matched control before three-seed official-data confirmation. If no candidate passes, stop this confirmation allocation and develop the evidence/partner-supervision branch.
4. The one clean/capped full-data arm is a separate historical interaction test. It is disabled initially and is not chosen from historical test outcomes.
5. Native-preserving heads/continuation require an exposure-audited external development panel. They are not fold candidates and are not implemented as a production branch here.

Any changed objective, learning-rate schedule, seed, horizon, split or candidate creates a recorded new experiment. Resume is only for the same contract. Record unsuccessful arms and non-promotions alongside successes; never replace a failed trial name with a new recipe.

## Resource plan

| Stage | Estimated GPU-hours | Launch status |
| --- | ---: | --- |
| Six adaptation folds | 150–270 | Initial release only; not submitted |
| Six readout folds | 150–270 | Gated |
| Six full-data seed confirmations | 480–600 | Gated, external evaluation unresolved |
| Complete core including pilots/evaluation/contingency | 1,000–1,500 | Not an authorization to launch all stages |
| Optional clean/capped run | 45–60 extra | Disabled |
| Optional bounded native continuation | 40–80 extra | Requires separate qualification and exposure contract |

Each production array task requests one node, four GPUs, 72 CPU cores, 400 GB host RAM and a 48-hour limit. Default maximum concurrency is two tasks (eight GPUs). Estimated adaptation runtime is roughly 6.25–11.25 hours per run; three waves would be approximately 19–34 hours, excluding queue and failure delays. Estimates derive from completed v2 runs, not extrapolation from short qualification timings.

Allow about 0.6–1 TiB campaign storage headroom. A full checkpoint is roughly 7.8 GB; retained latest/fallback/best and a fallback's older-best dependency can require **four payloads**, plus transient writes. Qualification payloads are retained separately. Shared filesystem availability is not a personal quota reservation. Project GPU allocation balance was not verified.

## Engineering trials performed

- SLURM **3176832**, `plmi-v3-QUALIFY`, four GH200 GPUs on n496: completed in **3 minutes 12 seconds**, exit 0, approximately **0.2133 allocated GPU-hours**. Full 650M clean model, four updates uninterrupted versus stop at two and resume; production global batch 64, 67 short training pairs, eight validation pairs. Tiny-model head branches additionally exercised uneven batches and a zero-contribution rank.
- Interactive node: real signal recovery, synthetic corruption/fallback checks, actual longest-fold memory/gradient profiles and fixed CPU baseline fits. These consumed the existing interactive allocation and were not production SLURM submissions.
- The first synthetic decision fixture was a single star graph and correctly failed the >=95% usable-bootstrap-replicates guard when its hub was omitted. Its failed fixture/log were retained; a multi-component fixture then exercised stage orchestration. No scientific threshold or production training code was relaxed.

No qualification checkpoint is eligible for scientific selection or production initialization.
