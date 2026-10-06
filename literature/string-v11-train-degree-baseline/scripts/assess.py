"""Reproduce the TRAIN-degree baseline from both exact V11 source formats, on CPU."""
from collections import Counter
import csv
import datetime
import gzip
import hashlib
import json
import math
from pathlib import Path
import platform
import sys

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results"


def sha(path, uncompress=False):
    h = hashlib.sha256()
    with (gzip.open if uncompress else open)(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.write_text(json.dumps(obj, indent=2) + "\n")


def load_native(split):
    cache = {}
    def key(seq):
        if seq not in cache:
            cache[seq] = hashlib.sha256(seq.encode()).hexdigest()
        return cache[seq]
    with gzip.open(ROOT / f"inputs/native/{split}.csv.gz", "rt", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == ["query", "text", "label"]
        rows = [(key(r["query"]), key(r["text"]), int(r["label"])) for r in reader]
    return rows, {h: len(s) for s, h in cache.items()}


def load_tuna(split):
    lookup, lengths = {}, {}
    with gzip.open(ROOT / f"inputs/tuna/{split}.dictionary.tsv.gz", "rt", newline="") as f:
        for identifier, seq in csv.reader(f, delimiter="\t"):
            h = hashlib.sha256(seq.encode()).hexdigest()
            assert identifier not in lookup
            lookup[identifier] = h
            lengths[h] = len(seq)
    with gzip.open(ROOT / f"inputs/tuna/{split}.interactions.tsv.gz", "rt", newline="") as f:
        rows = [(lookup[a], lookup[b], int(y)) for a, b, y in csv.reader(f, delimiter="\t")]
    return rows, lengths


def canonical(rows):
    return Counter((*sorted((a, b)), y) for a, b, y in rows)


def counter_sha(rows):
    h = hashlib.sha256()
    for (a, b, y), count in sorted(rows.items()):
        h.update(f"{a}\t{b}\t{y}\t{count}\n".encode())
    return h.hexdigest()


def training_degrees(train):
    degrees = {0: Counter(), 1: Counter()}
    for a, b, label in train:
        assert label in (0, 1)
        degrees[label].update({a, b})  # A self pair contributes once to its one endpoint.
    return degrees


def calculate_scores(pairs, degrees):
    # Only identities, never validation labels, enter this calculation.
    return np.asarray([math.log((degrees[1][a] + 1) / (degrees[0][a] + 1))
                       + math.log((degrees[1][b] + 1) / (degrees[0][b] + 1))
                       for a, b in pairs], dtype=np.float64)


def independent_metrics(labels, scores):
    """Grouped-threshold AP and Mann-Whitney AUROC, independent of sklearn."""
    order = np.argsort(scores, kind="stable")
    scores, labels = scores[order], labels[order]
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(scores))]
    sizes = np.diff(np.r_[starts, len(labels)])
    positives = np.add.reduceat(labels, starts)
    total_p, total_n = int(labels.sum()), int(len(labels) - labels.sum())
    ranks = starts + (sizes + 1) / 2.0
    auroc = (np.dot(ranks, positives) - total_p * (total_p + 1) / 2.0) / (total_p * total_n)
    tp = np.cumsum(positives[::-1])
    seen = np.cumsum(sizes[::-1])
    ap = np.sum((positives[::-1] / total_p) * (tp / seen))
    return {"ap": float(ap), "auroc": float(auroc)}


def describe(rows):
    pairs = canonical(rows)
    proteins = {a for a, b, y in rows} | {b for a, b, y in rows}
    positive = {a for a, b, y in rows if y} | {b for a, b, y in rows if y}
    positives = sum(r[2] for r in rows)
    return {"rows": len(rows), "positives": positives, "negatives": len(rows) - positives,
            "sequences": len(proteins), "positive_sequences": len(positive),
            "duplicate_labeled_unordered_sequence_rows": len(rows) - len(pairs),
            "opposite_label_pairs": sum(y == 1 and (a, b, 0) in pairs for a, b, y in pairs),
            "negative_rows_with_endpoint_absent_from_split_positives":
                sum(y == 0 and (a not in positive or b not in positive) for a, b, y in rows),
            "canonical_labeled_sequence_pair_multiset_sha256": counter_sha(pairs)}


def main():
    OUT.mkdir(exist_ok=True)
    complete = OUT / "COMPLETE.json"
    if complete.exists():
        complete.unlink()  # Never leave a stale completion marker during explicit reproduction.
    inputs = json.loads((ROOT / "provenance/inputs.json").read_text())
    for rec in inputs["inputs"]:
        path = ROOT / rec["archive"]
        assert sha(path) == rec["archive_sha256"], str(path)
        assert sha(path, uncompress=True) == rec["uncompressed_sha256"], str(path)
    print("All six archived source files match the pinned release hashes.", flush=True)
    results, objects, metric_rows = {}, {}, []
    for source, loader in [("native", load_native), ("tuna", load_tuna)]:
        train, train_lengths = loader("train")
        validation, validation_lengths = loader("validation")
        assert len(train) == 421792 and sum(y for _, _, y in train) == 38344
        assert len(validation) == 52725 and sum(y for _, _, y in validation) == 4794
        assert all(y in (0, 1) for _, _, y in validation)
        assert all(train_lengths.get(k, length) == length for k, length in validation_lengths.items())
        degrees = training_degrees(train)
        proteins = set(degrees[0]) | set(degrees[1])
        labeled_train = canonical(train)
        train_pairs = {(a, b) for a, b, _ in labeled_train}
        val_pairs = [tuple(sorted((a, b))) for a, b, _ in validation]
        scores = calculate_scores([(a, b) for a, b, _ in validation], degrees)
        labels = np.asarray([r[2] for r in validation], dtype=np.int8)
        overlap = np.asarray([pair in train_pairs for pair in val_pairs])
        simple = np.asarray([int(a in degrees[1]) + int(b in degrees[1]) for a, b, _ in validation])
        subsets = {"all_validation": np.ones(len(validation), dtype=bool),
                   "excluding_train_pair_overlap": ~overlap}
        source_result = {"train": describe(train), "validation": describe(validation),
                         "validation_sequences_seen_in_train": len(set(validation_lengths) & proteins),
                         "validation_sequences_unseen_in_train": len(set(validation_lengths) - proteins),
                         "validation_rows_with_both_endpoints_seen_in_train":
                             sum(a in proteins and b in proteins for a, b, _ in validation),
                         "overlap_validation_rows": int(overlap.sum()),
                         "same_label_overlap_rows": sum((*pair, int(y)) in labeled_train for pair, y in zip(val_pairs, labels)),
                         "opposite_label_overlap_rows": sum((*pair, 1-int(y)) in labeled_train for pair, y in zip(val_pairs, labels)),
                         "metrics": {}}
        for subset, mask in subsets.items():
            entry = {"rows": int(mask.sum()), "positives": int(labels[mask].sum()),
                     "negatives": int(mask.sum() - labels[mask].sum())}
            for baseline, score in [("degree_ratio", scores), ("positive_endpoint_count", simple)]:
                metrics = {"ap": float(average_precision_score(labels[mask], score[mask])),
                           "auroc": float(roc_auc_score(labels[mask], score[mask]))}
                check = independent_metrics(labels[mask], score[mask])
                assert all(abs(metrics[k] - check[k]) < 1e-12 for k in metrics)
                entry[baseline] = metrics
                metric_rows.append({"source": source, "subset": subset, "baseline": baseline,
                                    **{k: entry[k] for k in ["rows", "positives", "negatives"]}, **metrics})
            source_result["metrics"][subset] = entry
        # Scores are frozen before output labels are inspected by metric routines.
        np.savez_compressed(OUT / f"validation-scores-{source}.npz", labels=labels, score=scores,
                            positive_endpoint_count=simple, pair_seen_in_train=overlap,
                            protein_a=np.asarray([r[0] for r in validation]),
                            protein_b=np.asarray([r[1] for r in validation]))
        with gzip.open(OUT / f"validation-scores-{source}.csv.gz", "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(["source_row_1based", "protein_a_sha256", "protein_b_sha256", "label", "degree_ratio_log_score",
                        "train_positive_count_a", "train_negative_count_a", "train_positive_count_b", "train_negative_count_b", "pair_seen_in_train"])
            for i, ((a, b, y), score, seen) in enumerate(zip(validation, scores, overlap), 1):
                w.writerow([i, a, b, y, repr(float(score)), degrees[1][a], degrees[0][a], degrees[1][b], degrees[0][b], int(seen)])
        objects[source] = (canonical(train), canonical(validation), degrees,
                           Counter((*pair, int(y), float(s)) for pair, y, s in zip(val_pairs, labels, scores)))
        results[source] = source_result
        print(source, source_result["metrics"], flush=True)
    # Independent source loaders must reproduce the full data, TRAIN statistics, and every score.
    assert all(a == b for a, b in zip(objects["native"], objects["tuna"]))
    assert results["native"] == results["tuna"]
    degrees = objects["native"][2]
    with gzip.open(OUT / "train-sequence-degrees.csv.gz", "wt", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sequence_sha256", "train_positive_rows", "train_negative_rows", "log_smoothed_degree_ratio"])
        for h in sorted(set(degrees[0]) | set(degrees[1])):
            w.writerow([h, degrees[1][h], degrees[0][h], repr(math.log((degrees[1][h]+1)/(degrees[0][h]+1)))])
    previous = json.loads((ROOT / "provenance/legacy-degree-baseline.json").read_text())["results"]
    for current, old in [("all_validation", "all_released_validation"),
                         ("excluding_train_pair_overlap", "excluding_exact_train_pair_overlap")]:
        for k, value in results["native"]["metrics"][current]["degree_ratio"].items():
            assert abs(value - previous[old]["smoothed_degree_ratio"][k]) < 1e-12
    summary = {"created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
               "study": "Exact released V11 TRAIN-degree baseline, human validation",
               "formula": "log((d_train_positive(A)+1)/(d_train_negative(A)+1)) + log((d_train_positive(B)+1)/(d_train_negative(B)+1))",
               "identity": "exact amino-acid sequence SHA256; no sequence normalization or truncation",
               "counting": "TRAIN row occurrence counts, not distinct-neighbor degrees; identical-sequence pairs contribute once to their endpoint",
               "smoothing": 1, "source_row_multiplicity_preserved": True,
               "validation_used_for_degree_counts_or_tuning": False,
               "validation_role": "Released human_test file is human validation for the two releases, not the five-species tests",
               "checks": {"pinned_input_hashes_verified": True, "source_pair_label_multisets_identical": True,
                          "independent_source_degrees_and_all_scores_identical": True,
                          "independent_AP_and_AUROC_agree_to_1e_minus_12": True,
                          "reproduces_previous_V11_result_to_1e_minus_12": True},
               "environment": {"python": sys.version, "numpy": np.__version__, "sklearn": sklearn.__version__,
                               "architecture": platform.machine(), "hostname": platform.node(), "device": "CPU"},
               "sources": results}
    write_json(OUT / "summary.json", summary)
    with (OUT / "metrics.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(metric_rows[0])); w.writeheader(); w.writerows(metric_rows)
    report(summary)
    artifacts = []
    for path in sorted(ROOT.rglob("*")):
        if path.is_file() and path != complete and "__pycache__" not in path.parts:
            artifacts.append({"path": str(path.relative_to(ROOT)), "bytes": path.stat().st_size, "sha256": sha(path)})
    write_json(complete, {"complete": True, "created_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                          "checks": summary["checks"], "artifacts": artifacts})
    print("Complete: both exact V11 source formats independently reproduce the same baseline.", flush=True)


def report(summary):
    data = summary["sources"]["native"]
    full = data["metrics"]["all_validation"]
    filtered = data["metrics"]["excluding_train_pair_overlap"]
    lines = ["# TRAIN-degree baseline on the exact released STRING v11 human data", "",
             "Completed: " + summary["created_at_utc"], "",
             "**The earlier AP 0.8360 / AUROC 0.9635 result was already measured on these V11 files. This standalone assessment reproduces it independently from both PLM-interact humanV11's sequence CSVs and TUnA human seed 47's identifier/sequence TSVs.**", "",
             "Both sources have 421,792 TRAIN rows (38,344 positive / 383,448 negative) and 52,725 human validation rows (4,794 positive / 47,931 negative). Their unordered exact-sequence-pair/label multisets, including duplicate multiplicities, match exactly. These are two source-format checks of the same dataset, not independent biological replications.", "",
             "The files called `human_test` supply human validation for these releases. No STRING v12/v12.5 data, v5 HIPPIE/ILP training data, or nonhuman labels enter this calculation.", "",
             "| Exact V11 source | Human validation rows | AP | AUROC |", "|---|---:|---:|---:|"]
    for source, label in [("native", "PLM-interact humanV11 released CSVs"), ("tuna", "TUnA human seed-47 released TSVs")]:
        m = summary["sources"][source]["metrics"]["all_validation"]["degree_ratio"]
        lines.append(f"| {label} | 52,725 | {m['ap']:.6f} | {m['auroc']:.6f} |")
    lines += ["", "These are the counting baseline's scores on each source dataset. They are not predictions from the two neural models.", "",
              "**Method.** For every exact amino-acid sequence, count its occurrences in positive and negative TRAIN rows. Self pairs count once toward their endpoint. The score for a validation pair A–B is:", "",
              "```text", summary["formula"], "```", "",
              "This is equivalent in ranking to multiplying the two smoothed positive/negative count ratios. The +1 smoothing was fixed before this assessment. All released TRAIN rows, aliases resolved by exact sequence, label conflicts, and repeated rows are retained in the primary calculation. There is no sequence embedding, learned parameter, hyperparameter search, score-direction choice, or validation-label input to the score. Values are ranking scores, not calibrated interaction probabilities.", "",
              "**Sensitivity to repeated TRAIN/validation pairs.** Remove every validation row whose unordered exact sequence pair appears anywhere in TRAIN, irrespective of label. TRAIN counts and the score rule remain fixed.", "",
              "| Validation cohort, shared by both sources | Rows | Positives | Negatives | AP | AUROC |",
              "|---|---:|---:|---:|---:|---:|"]
    for label, m in [("All released rows", full), ("Excluding TRAIN-pair overlaps", filtered)]:
        lines.append(f"| {label} | {m['rows']:,} | {m['positives']:,} | {m['negatives']:,} | {m['degree_ratio']['ap']:.6f} | {m['degree_ratio']['auroc']:.6f} |")
    lines += ["", "The exclusion removes 89 rows: 82 same-label matches and seven opposite-label matches. The nearly unchanged result shows that direct pair repetition is not the main explanation for this baseline's high validation score. Protein identities still overlap.", "",
              "**Interpretation.** Every one of the 15,351 distinct validation sequences already appears in TRAIN. TRAIN contains 15,631 distinct sequences, of which only 7,492 appear in its positive rows. Approximately 77.44% of negative TRAIN rows contain an endpoint absent from TRAIN positives. The baseline uses this difference in protein-level label frequencies, without learning whether particular partners physically bind.", "",
              f"As a simpler diagnostic, counting how many endpoints appeared in TRAIN positives gives AP {full['positive_endpoint_count']['ap']:.6f} / AUROC {full['positive_endpoint_count']['auroc']:.6f}. The full-cohort positive prevalence, the reference for a noninformative ranking, is {full['positives']/full['rows']:.6f}.", "",
              "This is evidence of a substantial sampling signal in the human validation task. It does not establish that a degree lookup generalizes to unseen proteins or explain the native model's five-species results by itself. This study evaluates the human validation split only. No neural model or production training job was run. Full released row multiplicity is retained; metrics are deterministic point estimates without confidence intervals.", "",
              "**Verification.** Six lossless archived inputs were checked against the recorded release SHA-256 hashes. Native CSVs and TUnA TSVs were separately parsed and scored; the complete labeled score multisets agree. AP was independently recomputed using grouped precision/recall thresholds, and AUROC using the Mann–Whitney rank formula; both agree with sklearn within 1e-12. Results also reproduce the earlier V11 assessment within 1e-12.", "",
              "**Files.** [Source provenance](provenance/inputs.json); [full results and checks](results/summary.json); [metrics CSV](results/metrics.csv); [per-sequence TRAIN degrees](results/train-sequence-degrees.csv.gz); [native validation scores](results/validation-scores-native.csv.gz); [TUnA validation scores](results/validation-scores-tuna.csv.gz). Each score table preserves its original source row order. Compressed NPZ score arrays are also provided. [Completion manifest](results/COMPLETE.json) records artifact hashes.", "",
              "**Reproduction.** From the iPIN project root, run:", "", "```bash",
              "bash literature/string-v11-train-degree-baseline/scripts/run.sh", "```", "",
              "This uses the existing native PLM-interact ARM64 SIF as a CPU Python environment; GPU passthrough is disabled. All six required data files are archived under this study's `inputs/` directory, so reproduction does not require access to the sibling TUnA checkout. The provenance identifies the pinned public releases; it does not reconstruct any unavailable private checkpoint-training history.", ""]
    (ROOT / "REPORT.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
