"""Read-only source audits for the STRING v6 proposal; creates research summaries only."""
from collections import Counter
import csv
import gzip
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[1]
TUNA = PROJECT.parent / "iPIN-OpenPPI/benchmark/tuna/upstream/TUnA"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def fasta(path):
    op = gzip.open if path.suffix == ".gz" else open
    seqs = {}
    with op(path, "rt") as f:
        key = None
        parts = []
        for line in f:
            if line.startswith(">"):
                if key is not None:
                    seqs[key] = "".join(parts)
                key, parts = line[1:].split()[0], []
            else:
                parts.append(line.strip())
        if key is not None:
            seqs[key] = "".join(parts)
    return seqs


def describe(rows, lengths):
    labels = Counter()
    pairs = {0: Counter(), 1: Counter()}
    proteins = {0: set(), 1: set()}
    self_rows = Counter()
    for a, b, y in rows:
        labels[y] += 1
        pairs[y][tuple(sorted((a, b)))] += 1
        proteins[y].update((a, b))
        self_rows[y] += int(a == b)
    all_proteins = proteins[0] | proteins[1]
    return {
        "rows": len(rows), "label_counts": dict(labels),
        "unique_unordered_pairs_by_label": {y: len(pairs[y]) for y in (0, 1)},
        "duplicate_rows_by_label": {y: labels[y] - len(pairs[y]) for y in (0, 1)},
        "opposite_label_pairs": len(pairs[0].keys() & pairs[1].keys()),
        "self_pair_rows_by_label": dict(self_rows),
        "unique_sequences": len(all_proteins),
        "positive_sequences": len(proteins[1]), "negative_sequences": len(proteins[0]),
        "negative_sequences_not_in_positives": len(proteins[0] - proteins[1]),
        "negative_rows_with_endpoint_absent_from_same_split_positives": sum(
            y == 0 and (a not in proteins[1] or b not in proteins[1]) for a, b, y in rows),
        "sequence_length_range": [min(lengths[a] for a in all_proteins), max(lengths[a] for a in all_proteins)],
        "max_pair_length": max(lengths[a] + lengths[b] for a, b, _ in rows),
    }


def audit_legacy():
    splits, lengths, sources = {}, {}, {}
    for split in ("train", "test"):
        p = TUNA / f"data/processed/xspecies/human_{split}_dictionary.tsv"
        lookup = {}
        with p.open() as f:
            for identifier, seq in csv.reader(f, delimiter="\t"):
                digest = hashlib.sha256(seq.encode()).hexdigest()
                lookup[identifier] = digest
                lengths[digest] = len(seq)
        pairfile = TUNA / f"data/processed/xspecies/human_{split}_interaction.tsv"
        rawfile = TUNA / f"data/raw/xspecies/pairs/human_{split}.tsv"
        dscript = HERE / f"dscript-human-{split}.tsv"
        idrows = [(*sorted((a, b)), int(y)) for a, b, y in csv.reader(pairfile.open(), delimiter="\t")]
        rawrows = [(*sorted((a, b)), int(y)) for a, b, y in map(str.split, rawfile.open())]
        dsrows = [(*sorted((a, b)), int(y)) for a, b, y in map(str.split, dscript.open())]
        assert Counter(idrows) == Counter(rawrows) == Counter(dsrows)
        rows = [(*sorted((lookup[a], lookup[b])), y) for a, b, y in idrows]
        native = PROJECT / f"benchmark-v5-nonhuman/data/exposure/human.ppi.qrels.seq.{split}.csv"
        memo = {}
        def digest(s):
            if s not in memo:
                memo[s] = hashlib.sha256(s.encode()).hexdigest()
            return memo[s]
        with native.open() as f:
            native_counter = Counter((*sorted((digest(r["query"]), digest(r["text"]))), int(r["label"]))
                                     for r in csv.DictReader(f))
        assert native_counter == Counter(rows)
        splits[split] = rows
        sources[split] = {
            "native_tuna_dscript_unordered_sequence_pair_label_multisets_identical": True,
            "files": [{"path": str(x), "sha256": sha(x), "bytes": x.stat().st_size}
                      for x in [p, pairfile, rawfile, dscript, native]],
        }
    train, dev = splits["train"], splits["test"]
    train_p = set(a for r in train for a in r[:2])
    dev_p = set(a for r in dev for a in r[:2])
    train_pair = {(a, b) for a, b, _ in train}
    train_labeled = set(train)
    result = {
        "identity_unit": "exact amino-acid sequence SHA256, unordered pairs; row multiplicity preserved",
        "sources": sources,
        "train": describe(train, lengths), "validation_named_test": describe(dev, lengths),
        "combined": describe(train + dev, lengths),
        "cross_split": {
            "shared_sequences": len(train_p & dev_p), "dev_only_sequences": len(dev_p - train_p),
            "dev_rows_with_both_endpoints_seen_in_train": sum(a in train_p and b in train_p for a, b, _ in dev),
            "shared_unordered_pairs": len(train_pair & {(a, b) for a, b, _ in dev}),
            "dev_rows_with_same_labeled_pair_in_train": sum(r in train_labeled for r in dev),
            "dev_rows_with_opposite_labeled_pair_in_train": sum((a, b, 1-y) in train_labeled for a, b, y in dev),
        },
    }
    (HERE / "legacy-source-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print("LEGACY", json.dumps({k: v for k, v in result.items() if k != "sources"}, indent=2), flush=True)


def audit_string():
    results, eligible_sets = {}, {}
    for version in ("12.0", "12.5"):
        seqs = fasta(HERE / f"9606.protein.sequences.v{version}.fa.gz")
        path = HERE / f"9606.protein.physical.links.full.v{version}.txt.gz"
        all_edges, direct, eligible, filtered_humanv12 = {}, {}, {}, set()
        directed_rows = missing_sequence_rows = self_rows = 0
        with gzip.open(path, "rt") as f:
            header = f.readline().split()
            for line in f:
                parts = line.split()
                a, b = parts[:2]
                assert a.startswith("9606.") and b.startswith("9606.")
                directed_rows += 1
                self_rows += int(a == b)
                pair = tuple(sorted((a, b)))
                evidence = tuple(map(int, parts[2:]))
                assert pair not in all_edges or all_edges[pair] == evidence
                all_edges[pair] = evidence
                fields = dict(zip(header[2:], evidence))
                if fields["experiments"] > 0:
                    direct[pair] = evidence
                    if a not in seqs or b not in seqs:
                        missing_sequence_rows += 1
                        continue
                    if 50 <= len(seqs[a]) <= 800 and 50 <= len(seqs[b]) <= 800:
                        eligible[pair] = evidence
                    if len(seqs[a]) + len(seqs[b]) <= 2101 and fields["homology"] == 0 and fields["combined_score"] >= 400:
                        filtered_humanv12.add(pair)
        idx = {c: i for i, c in enumerate(header[2:])}
        seqpair = lambda p: tuple(sorted(hashlib.sha256(seqs[a].encode()).hexdigest() for a in p))
        unique_seqpairs = {seqpair(p) for p in eligible}
        results[version] = {
            "header": header, "sequences": len(seqs), "directed_rows": directed_rows,
            "unique_physical_id_pairs": len(all_edges), "self_rows": self_rows,
            "unique_pairs_direct_experiments_positive": len(direct),
            "missing_sequence_direct_rows": missing_sequence_rows,
            "eligible_50_800_direct_id_pairs_before_clustering": len(eligible),
            "eligible_unique_sequence_pairs_before_clustering": len(unique_seqpairs),
            "eligible_identical_sequence_pairs": sum(a == b for a, b in unique_seqpairs),
            "eligible_unique_ids": len({x for p in eligible for x in p}),
            "eligible_unique_sequences": len({s for p in unique_seqpairs for s in p}),
            "eligible_combined_score_ge_400": sum(e[idx["combined_score"]] >= 400 for e in eligible.values()),
            "eligible_homology_positive": sum(e[idx["homology"]] > 0 for e in eligible.values()),
            "eligible_with_transferred_experimental_support_also": sum(e[idx["experiments_transferred"]] > 0 for e in eligible.values()),
            "humanV12_paper_filters_before_clustering_id_pairs": len(filtered_humanv12),
            "note": "Counts precede 40% pair redundancy reduction, splits, test protection, negative sampling and full sequence validation; not proposed final TRAIN size.",
        }
        eligible_sets[version] = eligible
    old, new = eligible_sets["12.0"], eligible_sets["12.5"]
    results["v12.5_vs_v12.0_eligible_id_pairs"] = {
        "shared": len(old.keys() & new.keys()), "added": len(new.keys() - old.keys()),
        "removed": len(old.keys() - new.keys()),
        "shared_with_changed_direct_experiments_score": sum(old[p][1] != new[p][1] for p in old.keys() & new.keys()),
        "sequence_gzip_identical": sha(HERE / "9606.protein.sequences.v12.0.fa.gz") == sha(HERE / "9606.protein.sequences.v12.5.fa.gz"),
    }
    (HERE / "string-source-audit.json").write_text(json.dumps(results, indent=2) + "\n")
    print("STRING", json.dumps(results, indent=2), flush=True)


if __name__ == "__main__":
    audit_legacy()
    audit_string()
