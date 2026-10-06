# V3 experiment ledger

Preparation originally finished with production unsubmitted. **Subsequently, the user authorized all six runs concurrently; array `3177489` was submitted on 2026-09-30 at 11:13 UTC (13:13 Stockholm).** See [submission provenance](provenance/submission-3177489.json). The frozen release retains the pre-submission ledger. The original machine-readable contract is [configs/campaign.json](configs/campaign.json); the current readout amendment is [configs/readout-stage.json](configs/readout-stage.json). The LR array is now stopped and readout array **3196827** is submitted, as recorded below. Qualification trials are excluded from model-performance comparisons.

| Initial configuration | Learning rate | Train / validation | Horizon | Status |
| --- | ---: | ---: | ---: | --- |
| clean-cls-linear-lr2e-5-fold0-seed2 | 2e-5 | 73,691 / 17,490 | 6,000 originally | Stopped safely; see amendment below |
| clean-cls-linear-lr2e-5-fold1-seed2 | 2e-5 | 71,698 / 18,285 | 6,000 originally | Stopped safely; see amendment below |
| clean-cls-linear-lr2e-5-fold2-seed2 | 2e-5 | 71,986 / 18,515 | 6,000 originally | Stopped safely; see amendment below |
| clean-cls-linear-lr5e-6-fold0-seed2 | 5e-6 | 73,691 / 17,490 | 6,000 originally | Stopped safely; see amendment below |
| clean-cls-linear-lr5e-6-fold1-seed2 | 5e-6 | 71,698 / 18,285 | 6,000 originally | Stopped safely; see amendment below |
| clean-cls-linear-lr5e-6-fold2-seed2 | 5e-6 | 71,986 / 18,515 | 6,000 originally | Stopped safely; see amendment below |

The original protocol scheduled six checkpoint-selection opportunities per run (updates 1,000 through 6,000), 36 across the initial stage. The amendment below uses the five completed opportunities for both learning rates and new heads. These folds are used for development, so their selected-checkpoint results are not unbiased final estimates.

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
| Six adaptation folds | 150–270 | Stopped safely; actual 190.8 allocated GPU-hours |
| Six readout folds | 125–225 for amended window | Submitted as array 3196827 |
| Six full-data seed confirmations | 480–600 | Gated, external evaluation unresolved |
| Complete core including pilots/evaluation/contingency | 1,000–1,500 | Not an authorization to launch all stages |
| Optional clean/capped run | 45–60 extra | Disabled |
| Optional bounded native continuation | 40–80 extra | Requires separate qualification and exposure contract |

Each production array task requests one node, four GPUs, 72 CPU cores, 400 GB host RAM and a 48-hour limit. The prepared default is two tasks; the authorized array requests six concurrent tasks (24 GPUs) using a recorded scheduling override. Estimated adaptation runtime is roughly 6.25–11.25 hours per run; with all six running together, the stage is roughly one such wave, excluding queue and failure delays. Estimates derive from completed v2 runs, not extrapolation from short qualification timings.

Allow about 0.6–1 TiB campaign storage headroom. A full checkpoint is roughly 7.8 GB; retained latest/fallback/best and a fallback's older-best dependency can require **four payloads**, plus transient writes. Qualification payloads are retained separately. Shared filesystem availability is not a personal quota reservation. Project GPU allocation balance was not verified.

## Engineering trials performed

- SLURM **3176832**, `plmi-v3-QUALIFY`, four GH200 GPUs on n496: completed in **3 minutes 12 seconds**, exit 0, approximately **0.2133 allocated GPU-hours**. Full 650M clean model, four updates uninterrupted versus stop at two and resume; production global batch 64, 67 short training pairs, eight validation pairs. Tiny-model head branches additionally exercised uneven batches and a zero-contribution rank.
- Interactive node: real signal recovery, synthetic corruption/fallback checks, actual longest-fold memory/gradient profiles and fixed CPU baseline fits. These consumed the existing interactive allocation and were not production SLURM submissions.
- The first synthetic decision fixture was a single star graph and correctly failed the >=95% usable-bootstrap-replicates guard when its hub was omitted. Its failed fixture/log were retained; a multi-component fixture then exercised stage orchestration. No scientific threshold or production training code was relaxed.

No qualification checkpoint is eligible for scientific selection or production initialization.

## 2026-09-30 amendment: end LR screening and submit readout stage

The user authorized: "yes! end this screen and move on to the next step." All six LR workers honored `REQUEST_STOP`, saved committed state and exited successfully. SLURM `COMPLETED` indicates clean process exit, not completion of the original 6,000-update scientific schedule. The [early decision](decisions/adaptation-early.json) preserves this distinction, all validation artifacts and hash-verified best/latest checkpoint payloads. Mean best AP through validation 5,000 was 0.616488 at 2e-5 versus 0.600486 at 5e-6; the higher rate won all three folds.

| LR | Fold | Actual stop | Selected update |
| --- | ---: | ---: | ---: |
| 2e-5 | 0 | 5,367 | 3,000 |
| 2e-5 | 1 | 5,419 | 3,000 |
| 2e-5 | 2 | 5,477 | 4,000 |
| 5e-6 | 0 | 5,380 | 5,000 |
| 5e-6 | 1 | 5,386 | 4,000 |
| 5e-6 | 2 | 5,385 | 4,000 |

The new [readout protocol](READOUT_PROTOCOL.md) freezes five selection opportunities (1,000–5,000) before any head production. Every new run keeps the original 6,000-update LR schedule and stops after the committed validation at 5,000. A bounded race may permit up to five extra optimizer updates, excluded from selection. This adaptive window supersedes the original completed-horizon analysis for these two stages; it is not a new unbiased test.

Array **3196827** submits the six clean readout folds concurrently at **2e-5**, four GPUs each: tasks 0–2 are CLS-MLP folds 0–2; tasks 3–5 are residue-MLP folds 0–2. Both heads add 655,489 parameters and start from ESM2. The old high-LR linear runs are reused as controls. See [submission provenance](provenance/submission-3196827.json) and the immutable [readout release](releases/20260930-readout/release.json). Confirmation and external/native-adaptation branches remain gated.

Readout engineering trials:

- Job **3196205**, four GPUs, 315 seconds, failed exact residue-head resume while CLS-MLP passed. This is retained as a failed engineering trial, not a model-performance result.
- Job **3196606**, four GPUs, 317 seconds, passed bitwise model/optimizer/RNG/state/prediction resume for both actual 650M heads after disabling DDP gradient bucket views. The single executable trainer change is AST-verified; model/data/checkpoint code remains byte-identical. Details and limitations are in [the engineering record](qualification/READOUT_ENGINE.md).
- Local wrapper qualification includes manual stop and requeue before the first validation, normal automatic budget stop, failed workers, monitor errors and missed-window rejection. Scheduler calls in this wrapper fixture are mocked. Historical real signal/requeue evidence remains separately labeled.
- [SLURM accounting](provenance/adaptation-stop-and-readout-qualification-accounting.txt) records 190.818 allocated GPU-hours for the ended LR array and 0.702 GPU-hours for the two full-size head qualification jobs. Interactive diagnostics are additional.

The readout stage adds an estimated 125–225 GPU-hours of production; scheduler overhead and failures are additional. No checkpoint in a qualification directory may initialize production or enter scientific selection.

## 2026-10-01 completed readout review

All six tasks in array **3196827** completed successfully with exit `0:0`, no restarts and safe stops at update **5,001**, after all five planned validations through 5,000. Total production allocation was **178.164 GPU-hours**. The completion audit verified all **18 retained checkpoint payloads** (140.639 GB), including selected, fallback and resume states. See [the full results](READOUT_RESULTS.md) and [audit](provenance/readout-completion-3196827.json).

| Model | Mean best validation AP | Change vs CLS-linear | Best updates across folds |
| --- | ---: | ---: | --- |
| Preserved CLS-linear | 0.616488 | — | 3,000 / 3,000 / 4,000 |
| CLS-MLP | 0.614789 | -0.001699 | 3,000 / 3,000 / 3,000 |
| Residue-MLP | 0.620374 | +0.003886 | 3,000 / 3,000 / 4,000 |

Residue-MLP improved AP, AUROC and protein macro-AP on all three folds against CLS-linear. It passed the comparison rule against CLS-MLP (+0.005585 mean AP) but fell below the required +0.005 mean AP against CLS-linear. CLS-MLP failed the mean-AP and positive-fold rules. Consequently [the frozen decision](decisions/readout-window5000.json) selects **no candidate for confirmation**. No thresholds, configurations or selection windows were changed after seeing these outcomes; no further jobs were submitted. These development results do not establish superiority over native PLM-interact on a matched independent test population.


## 2026-10-01 final single-run training and scope correction

The user authorized one final residue-MLP at LR 2e-5 and then explicitly clarified that this folder is for **retraining only**, with benchmarking later in a separate `benchmark-v3` folder. The earlier +0.005 development gate remains failed; this final run is a user-authorized exploratory override. No additional LR/readout/seed search is scheduled.

Configuration `clean-residue-mean-official-seed2` uses the full official 163,085/59,258 train/validation partitions, seed 2, ESM2 initialization, clean BCE, no length cap, 12,745 updates, 2,000 warmup updates and four GPUs. Its budget and checkpoint-selection opportunities match the existing v2 clean-BCE control. The qualified v3 training/model/data/checkpoint core is unchanged.

Initial job **3206387** included an automatic post-training benchmark, which the user's clarification removed. It stopped safely at update **26**, checkpoint SHA `acffc13df23265c11ec46ed4d7587f6c648865c3787e7bbf383d549ce51189eb`, and exited 0:0 after 2m36s (0.1733 allocated GPU-hours). **No test inference ran.** Replacement job **3206610** uses the training-only release `20261001-final-training` and resumes that exact checkpoint/contract, fingerprint `e88b62a2ebf16f901ebd59fa6d486220f7d7173ebb8fbe77eccf62fed7b4cc21`. This is continuation of one model, not another training experiment.

The wrapper's normal completion, completed-run skip, manual stop, failed/incomplete training, safe requeue and requeue-limit branches passed mocked scheduler checks. The full-size four-GPU exact-resume evidence is inherited for the identical core. The replacement job has no automatic benchmark phase or benchmark-file dependency. Prepared benchmark code and cached baseline predictions are retained separately but inactive; their forward/CPU fixture checks are engineering evidence, not v3 test results.

The current protocol is [FINAL_PROTOCOL.md](FINAL_PROTOCOL.md). The former combined release remains historical and is superseded; `CURRENT_FINAL` points to the training-only release. Training completion produces the final selected checkpoint. Test comparison awaits a later benchmarking task.
