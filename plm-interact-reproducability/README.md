This directory checks PLM-interact checkpoint reproducibility using the supplied SIF on Arrhenius GH200 GPUs. **No retraining is performed.** Start with [REPORT.md](REPORT.md) for the results and [PROTOCOL.md](PROTOCOL.md) for the evaluation choices, precision conventions, and limitations.

Fresh inference covers Figure 2's five cross-species tests (242,000 rows), Figure 4's Bernett test (52,048 rows), and Figure 5's mutation test (841 rows, including the exact 598-case published subset). Mutation evaluation uses both the released mutation checkpoint and zero-shot humanV11. Separate runs test sequence truncation and a fixed pair-order subset. Baseline comparisons and training ablations use deposited scores; they are identified as source-data audits.

The files needed to inspect the result are:

- `results/fresh-metrics.csv`: AP, trapezoidal PR-AUC, AUROC, fixed-threshold classification metrics, and precision sensitivity.
- `results/published-score-metrics.csv`: independent metric recomputation from the paper's source workbook.
- `results/score-concordance.csv`: score differences and classification agreement, including sequence truncation and pair reversal.
- `results/coverage.json`: expected row counts, exact row/label coverage, verified file hashes, and GPU identities. `complete` must be true for a final result.
- `results/predictions/`: raw inference shards with row IDs, labels, logits, probabilities, length handling, and per-worker manifests. Merged CSVs are directly under `results/`.
- `results/plots/`: standalone PDF and PNG research figures.
- `results/*audit*.json` and supplementary CSVs: sample matching, split overlap, metadata/date repairs, duplicate rows, masking, identity bins, and reporting checks.
- `provenance/`: pinned releases, original publication code, checkpoint/download hashes, and run settings.

The SIF is `../images/plm-interact/plm-interact-native-arm64-v1.sif`. Its humanV11 checkpoint and ESM configuration/tokenizer are used locally; the Bernett and mutation checkpoints are in `checkpoints/`. The container wrapper disables online model fetching, AMP, and TF32. Host-side download/extraction scripts use Python and `requests`; scientific analysis runs inside the SIF. Model construction and all loaded tensors are checked against the released state dictionaries.

To verify or resume this exact experiment on a GPU allocation, run from this directory:

```bash
python scripts/fetch_resources.py
python scripts/fetch_source_members.py
python scripts/extract_source.py
bash scripts/container.sh python tests/test_metrics.py
bash scripts/container.sh python scripts/qualify.py
bash scripts/container.sh python scripts/audit_sources.py
bash scripts/container.sh python scripts/audit_supplement.py
bash scripts/container.sh python scripts/audit_reporting.py
sbatch slurm/evaluate.sbatch
```

The submitted job runs the cross-species, reverse-order subset, and Bernett tasks on eight GPUs. The six mutation tasks run on the interactive GPU with:

```bash
bash scripts/container.sh python -u scripts/infer.py \
  --job-id "${SLURM_JOB_ID:-interactive}" \
  --tasks mutation_zero_full mutation_zero_1603 mutation_zero_2196 \
          mutation_ft_full mutation_ft_1603 mutation_ft_2196
```

Completed shards are reused only when input hashes, code hashes, and evaluation settings agree. A changed configuration produces an error instead of overwriting evidence. For a new independent run, archive the existing `results/` directory first or use a separate copy of this experiment directory. If copying the directory, update the absolute experiment paths in the two SLURM scripts. The report and accounting helper record the original job IDs; update those for a new allocation. Preserve this run's results for comparison.

After all workers finish, produce the verified metrics and final artifacts:

```bash
python scripts/fetch_audit_splits.py
python scripts/audit_mutation_splits.py
bash scripts/container.sh python scripts/audit_truncation.py
bash scripts/container.sh python scripts/aggregate.py
bash scripts/container.sh python scripts/diagnose_bernett.py
bash scripts/container.sh python scripts/plot_results.py
bash scripts/container.sh python scripts/write_report.py
python scripts/finalize_provenance.py
```

`aggregate.py` requires every expected shard and exact test-row coverage. `--allow-partial` is available for progress inspection, but `write_report.py` refuses incomplete coverage. `diagnose_bernett.py` is a separately labeled post-hoc test of the missing length-cap hypothesis, preserving the primary results and all probe outputs. The extra mutation split files are used only for read-only overlap checks. No evaluation script creates an optimizer or updates model weights.

The original run used SLURM job 3110591 plus interactive allocation 3085063. Sources are the [published paper](https://www.nature.com/articles/s41467-025-64512-w), [publication code archive](https://doi.org/10.5281/zenodo.16643324), [authors' repository](https://github.com/liudan111/PLM-interact), and pinned Hugging Face metadata. The paper, supplements, reporting summary, peer review, and workbook are preserved under `../literature/plm-interact/`.
