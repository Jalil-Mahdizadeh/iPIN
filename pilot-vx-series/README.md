# Vx pilot series

The completed Vx pilots live here. Their code, inputs, results and frozen hashes are unchanged by the move on 10 October 2026. [migration.json](migration.json) records the verified file hashes.

| Study | Question | Assessment AP |
|---|---|---:|
| [Vx](pilot-vx/results/REPORT.md) | Original depth-128 matched MSA | True 0.6644; shuffled 0.6632 |
| [Vx-v2](pilot-vx-v2/results/REPORT.md) | Local pooling | Local true 0.6557 |
| [Vx-v3](pilot-vx-v3/results/REPORT.md) | Column-shuffled homologs | 0.6501 |
| [Vx-v4](pilot-vx-v4/results/REPORT.md) | Independent homolog sampling | 0.6493 |
| [Vx-v5](pilot-vx-v5/results/REPORT.md) | Nested depth extension to 256 | True 0.6591 |

Native AP on this shared 958-pair DEV assessment is 0.6409. The repeated reference predictions are not independent replications. No pilot evaluates TEST. [Vy](../pilot-vy/results/REPORT.md) and [Vz](../pilot-vz/results/REPORT.md) remain separate studies.

Small compatibility symlinks at the old repository paths preserve recorded absolute paths. Symlinks in this directory preserve the project-relative dependencies of immutable historical scripts. These are references to existing files, not additional copies or additional studies. Do not launch completed pilots again.

The authorized full TRAIN/VAL continuation is [retrain-vx](../retrain-vx/README.md); it excludes shuffled encoding and TEST evaluation.
