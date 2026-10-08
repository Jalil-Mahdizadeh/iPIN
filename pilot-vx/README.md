# Bernett Vx pilot

The authorized pilot tests whether frozen paired-MSA trunk features add useful information to the native Bernett-trained PLM-interact baseline. The revised [proposal](../improvment-proposal-vx.md) fixes the scientific question and resource limits. Current execution is recorded in [STATUS.md](STATUS.md); machine-readable evidence is under `provenance/`, `qualification/` and `results/`.

The pilot contains 4,000 original TRAIN and 4,000 original DEV pairs, sampled before inspecting alignment availability. The complete original source CSVs were checked against the raw pair arrays. The later 107 TRAIN and two DEV homology exclusions were not applied. Test labels and test predictions are not inputs.

DEV homology groups use MMseqs2 identity ≥40%, coverage ≥80% of both sequences and E-value ≤0.001. Whole components are assigned without labels: 1,068 sampled edges are calibration, 958 assessment, and 1,974 cross the internal boundary. Crossing edges remain in the fixed 4,000-row DEV report. This does not establish universal family independence or make the historically reused dataset newly blind.

The native baseline uses full sequences, mean AB/BA logits, FP32 weights and BF16 autocast. All 4,000 sampled DEV scores were reused from verified `benchmark-v1` predictions with the identical contract. TRAIN baseline logits do not enter head fitting.

The primary encoder uses frozen MSA Pairformer representations after zero-based layer 15. It does not call the released structural contact heads. Each orientation symmetrizes the directed inter-chain representation blocks, projects 256 channels onto 32 fixed seeded Gaussian directions, then records mean, standard deviation, maximum and 95th percentile: 128 scalars. Average the two orientations. Projection and pooling are fixed before outcomes are examined.

Controls use hierarchical family/order/class shuffled pairing, and 68 symmetric quality/profile features. The same TRAIN-only standardization, 32-component PCA and regularized logistic classifier are used for all three heads. Fit only the evidence-eligible TRAIN population. Fusion alpha is chosen only on calibration from the declared grid; exact baseline fallback applies on every unavailable example. Report weakly changed shuffled controls as a limitation, not evidence of pairing specificity. For a positive pairing-sensitivity decision, at least 80% of covered assessment pairs must have at least half their non-query rows changed by the taxonomic null. A material AUROC decline is predefined as greater than 0.005.

The archive is read as a stream after checksum verification. No full archive extraction is needed. Store at most 4,096 accession-prioritized homolog rows per target monomer, retain full query coordinates, join genomic accessions, and then select at most 127 homologs plus the query. Paired alignments need eight homologs and 50% high-quality coverage per chain; the initial combined-query cap is 1,536 residues. No windows or local-search rescue are enabled.

The archive's taxonomy gives genus through phylum, not species IDs. Taxonomic balancing uses the reported taxonomic family; the null falls back to order and then class for remaining singleton groups; they do not prove species independence. Exact identical homolog sequences in both slots are rejected, but biological instance identity and paralog collapse cannot be completely resolved from this archive. Query family dependence remains a limitation to review alongside bootstrap and subgroup results.

Limits: 24 allocated GPU-hours (four reserved for qualification, at most 20 in the batch stage), 2,000 CPU-core-hours and 150 GB additional working storage. There is no automatic requeue or production continuation. Incomplete computational coverage leads to an explicit inconclusive report; execution errors cannot become baseline fallback.

The isolated runtime is [images/msa-pairformer](../images/msa-pairformer/README.md). Source, model and data identities are pinned in `config.json` and provenance. `provenance/freeze.json` binds the executable pilot to its inputs. Per-pair feature artifacts have checksums and an embedded fingerprint; corruption or changed inputs fail closed.

Useful commands, from the repository root:

```bash
python -B pilot-vx/scripts/status.py
apptainer exec images/msa-pairformer/msa-pairformer-arm64-v1.sif python -B -m unittest discover -s pilot-vx/tests -v
```

Do not rerun preparation or modify frozen inputs while a worker is active. A resumed run must account for already consumed GPU-hours. The launch script refuses a second submission; an interrupted pilot needs an explicit remaining-budget calculation before another job.

Alignment-format qualification found homolog residue `J`, which is absent from the encoder vocabulary. Normalize homolog J to unknown X without deleting a column, record replacement counts, and continue to reject other unsupported symbols and invalid coordinate lengths. Original human-query matching remains exact.

The fixed 50% homolog coverage requirement applies separately to each chain's high-quality positions after masking, in addition to the monomer full-query coverage check. Rows supported only by discarded low-quality columns cannot create paired evidence.

Real-input qualification showed that family balancing can select one row per family, making a family-only null a no-op. The frozen null therefore deranges non-query rows within families first, then groups remaining singletons by order, then class. It leaves final singletons unchanged, never crosses classes globally, and records changed counts at each level. This retains only coarse phylogeny for some rows; pairing sensitivity is necessary but not sufficient evidence of direct coevolution.
