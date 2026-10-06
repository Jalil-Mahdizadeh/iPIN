**Data frozen for v2, 29 September 2026**

The first experiments use the same Bernett human PPI rows as the prepared v1 training and validation sets. Positives and negatives are unchanged. The negative labels describe sampled unreported interactions, with degree control in expectation; they do not establish experimentally confirmed noninteraction. No new evidence-curated corpus is claimed here.

| Partition | Training pairs | Validation pairs | Crossing pairs excluded |
| --- | ---: | ---: | ---: |
| Official | 163,085 | 59,258 | — |
| Development fold 0 | 73,691 | 17,490 | 71,904 |
| Development fold 1 | 71,698 | 18,285 | 73,102 |
| Development fold 2 | 71,986 | 18,515 | 72,584 |

Official training has 81,550 positives and 4,285 distinct sequences. Official validation has 29,628 positives and 3,710 distinct sequences. Exact sequence sets are disjoint. The inherited v1 preparation additionally purged detected sequence homologues against the historical held-out sets; its audit is preserved in [the parent manifest](provenance/v1-prepared-manifest.json). This historical sequence-only audit is distinct from reading test labels in v2.

The coverage control contains 130,461 training pairs with at most 2,193 combined residues. All other arms retain all 163,085 pairs. The longest training and validation inputs have 16,322 and 39,391 tokens, respectively, including CLS and two EOS tokens. Validation coverage is never capped. The token budget controls microbatch packing; an oversized pair runs alone and is not truncated or dropped.

The v1 shared token cache originally included unused historical test sequences. V2's cache was rebuilt to retain only the 7,995 sequences actually referenced by official training or validation. Existing protein IDs are preserved, unused IDs have zero-length entries, and the cache now contains 5,416,846 residue tokens. [Token-subset provenance](provenance/token-subset.json). No test CSV, test pair array, test score file, or unused test sequence is part of v2's prepared inputs.

**Development folds**

The folds partition only the official training proteins. A CPU MMseqs2 all-versus-all search used sensitivity 7.5, minimum identity 0.40, coverage 0.80 of both sequences, and E-value at most 0.001. Connected components of detected edges are assigned whole to folds. There are 4,154 components, with a largest component of three proteins. These thresholds limit detected close homology; they do not prove independence at every family or domain level.

Assignment uses seed 20260929 and balances protein counts without consulting labels, scores, official validation, or test outcomes. For each fold, validation requires **both** endpoints in its held-out protein group; training requires both endpoints outside. Crossing pairs are excluded from that fold. Each training/validation/excluded union reconstructs the parent training array exactly. Training and validation endpoints share neither exact sequences nor detected component edges.

These are internal development partitions. The released native Bernett checkpoint has already seen their parent training labels and is not a clean fold control. Each fold control and candidate must start afresh from the same pinned ESM-2 model. Model selection on these folds does not turn them into external tests.

The [prepared manifest](data/prepared/manifest.json) records array hashes and counts. [Fold assignments](data/prepared/fold-groups.json), [homology hits](data/development-homology.tsv), [search script](scripts/audit_homology.sh), and [scientific qualification](qualification/science.json) make the construction reviewable.

**Requirements before data expansion or a generalization claim**

Evidence-aware training data and a new external holdout are later work packages. Their pair counts are currently unknown. Before enabling those stages:

1. Freeze source versions and sequence mappings. Retain source IDs, publication dates, assay, evidence strength, taxon, and whether a positive denotes direct binding or co-complex membership.
2. Deduplicate undirected pairs across sources; resolve sequence and label conflicts explicitly. Keep sampled unknowns distinct from experimentally supported negative evidence.
3. Reserve the external protein/evidence groups before further candidate optimization. Audit exact pairs, proteins, homologues, evidence reuse, dates, and all known checkpoint pretraining/fine-tuning sources. Separate one-protein-unseen and both-proteins-unseen endpoints.
4. Restrict nuisance features and negative-sampling graph statistics to training information. Include fixed length/composition and training-only similarity controls; do not infer negatives solely from location or arbitrary in-batch pairing.
5. Match pair exposure when comparing old and new supervision, and freeze external labels away from the trainer and development analysis. Require independent review of the exposure manifest before opening external results.

Experimental interface supervision additionally requires biological-assembly and residue mapping checks, exclusion of unresolved residues from the auxiliary loss, and homology-separated complexes. It is not enabled by the initial release. These conditional corpus requirements are not represented as completed data preparation.
