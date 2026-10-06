# V4 preparation completed

Prepared on 2026-10-01. **At the preparation handoff, three production runs were ready and none had been submitted.** They were subsequently submitted at the user's request at 10:38 UTC as jobs 3210885, 3210886 and 3210887; see the [submission record](provenance/submission-20261001T103847826554.json) and [current overview](README.md). The setup is frozen in [20261001-production](releases/20261001-production/release.json), and the verified pre-submission commands are retained in the [dry-run launch plan](provenance/production-launch-plan.json).

| Condition | New production run | Full-length execution | Four-GPU exact resumption |
| --- | --- | --- | --- |
| S1 | ESM2 650M, chain-aware attention | Passed | Passed |
| C0 | ESM C 600M, standard attention | Passed | Passed |
| C1 | ESM C 600M, chain-aware attention | Passed | Passed |

ESM2 standard attention (S0) reuses the final full-data v3 residue-MLP run. Its eventual best checkpoint follows the existing validation rule; no fourth v4 job is prepared. The existing v3 job was left untouched.

All three new runs use the same **163,085 training / 59,258 validation pairs**, residue-mean residual MLP, seed 2, LR 2e-5, 2,000 warmup updates and 12,745 total updates. They start from their respective unsupervised pretrained weights. Each requests one four-GPU node, 72 CPUs, 400 GB RAM and an initial 48-hour allocation. Thirteen complete official validations select the best checkpoint. There is no length cap, additional LR screen or test-set evaluation. See the fixed [protocol](PROTOCOL.md).

The actual largest training and validation inputs, **16,322 and 39,391 tokens**, passed with both orientations and all three full-size models. Training checks included backward and one disposable optimizer step. Attention forward/gradient checks, original-SDK embedding agreement, token identities and unchanged data were also verified. Full-length execution establishes computational feasibility; performance at long contexts remains to be learned from the production results.

Full-size four-GPU checks compared uninterrupted four-update trajectories against stopping after update two with validation pending, then resuming. Weights, optimizer, RNG, sampler/selection state and saved validation predictions were **bitwise identical for all three models**. Separate fault fixtures verified corrupt-checkpoint fallback and preservation of the selected best checkpoint. Ten shell fixtures exercised stop, requeue, error and duplicate-writer behavior; actual scheduler requeue was not induced. Detailed evidence is linked from [README.md](README.md).

The first qualification job, `3209856`, passed S1 but stopped before ESM C training because the inherited `torchrun` executable selected the base Python. The corrected launcher uses `python -m torch.distributed.run`, preserving the ESM C virtual environment. Worker imports passed, and qualification job `3210358` completed successfully with the remaining C0/C1 checks. Model/training core files were unchanged by this correction. Both jobs, frozen qualification code and logs are retained; none is a production run.

The existing native SIF serves S1; the dedicated pinned ESM C SIF serves C0/C1. No further image rebuild was necessary. Data, pretrained weights, images, configurations and qualification reports are hash-checked before launch. Production checkpoints include optimizer, RNG, sampler, selection and pending-validation state; safe time-limit handling supports continuation under the same four-GPU contract.

At the requested preparation stopping point, the dry-run launcher had verified and printed exactly three commands and `runs/` was empty. Subsequent production submission is recorded above. Benchmarking remains a separate future task. The launch instructions are in [README.md](README.md#launch-and-resume).
