"""Read-only data/prediction audit for the second improvement proposal.

Run from the workspace in the existing SIF. No fitting, inference, checkpoint
selection, or changes to the frozen benchmark or active training runs.
"""
from pathlib import Path
import datetime
import hashlib
import json

import numpy as np
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
DATA = ROOT / "retrain-v1/data/prepared"
BENCH = ROOT / "benchmark-v1"
INPUTS = []


def record(path):
    INPUTS.append({"path": str(path.relative_to(ROOT)),
                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return path


def quantiles(a):
    return dict(zip(["p0", "p25", "p50", "p75", "p95", "p99", "p100"],
                    np.quantile(a, [0, .25, .5, .75, .95, .99, 1]).tolist()))


def metrics(y, z):
    return {"rows": len(y), "prevalence": float(y.mean()),
            "ap": float(average_precision_score(y, z)),
            "auroc": float(roc_auc_score(y, z)),
            "brier": float(np.mean((expit(z) - y) ** 2))}


def predictions(path, rows):
    with np.load(record(path), allow_pickle=False) as saved:
        a = saved["predictions"]
    a = a[np.argsort(a[:, 0])]
    assert a.shape == (len(rows), 4)
    assert np.array_equal(a[:, 0], np.arange(len(rows)))
    assert np.array_equal(a[:, 1], rows[:, 2])
    assert np.isfinite(a).all()
    return a[:, 2:4].mean(1)


def main():
    result = {"created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "purpose": "Exploratory description of already observed data and predictions; no new holdout claim",
              "splits": {}, "existing_prediction_strata": {}, "training_snapshot": {}}
    lengths = np.diff(np.load(record(DATA / "offsets.npy")))
    for split in ["train", "val", "test"]:
        rows = np.load(record(DATA / f"{split}.npy"))
        y = rows[:, 2]
        combined = lengths[rows[:, 0]] + lengths[rows[:, 1]]
        proteins = np.unique(rows[:, :2])
        positive_degrees = np.bincount(rows[y == 1, :2].ravel(), minlength=len(lengths))
        negative_degrees = np.bincount(rows[y == 0, :2].ravel(), minlength=len(lengths))
        pos, neg = positive_degrees[proteins], negative_degrees[proteins]
        degree = pos + neg
        ranked = np.argsort(degree)[::-1]
        top_count = max(1, int(np.ceil(len(proteins) * .01)))
        result["splits"][split] = {
            "rows": len(rows), "positives": int(y.sum()), "unique_sequences": len(proteins),
            "self_pairs": int((rows[:, 0] == rows[:, 1]).sum()),
            "combined_residues": quantiles(combined),
            "by_label": {str(label): {
                "rows": int((y == label).sum()),
                "combined_residues": quantiles(combined[y == label]),
                "over_2193_fraction": float((combined[y == label] > 2193).mean())
            } for label in [0, 1]},
            "over_2193_rows": int((combined > 2193).sum()),
            "over_2193_fraction": float((combined > 2193).mean()),
            "degree_positive_negative_spearman": float(spearmanr(pos, neg).statistic),
            "positive_only_proteins": int(((pos > 0) & (neg == 0)).sum()),
            "negative_only_proteins": int(((neg > 0) & (pos == 0)).sum()),
            "top_1pct_proteins_endpoint_fraction": float(degree[ranked[:top_count]].sum() / degree.sum()),
            "degree_note": "Each pair contributes two endpoints, including self-pairs. Within-split label descriptions are not available deployment features."
        }
        if split == "train":
            continue
        native = BENCH / "results" / f"native-bernett-{split}.npz"
        paths = {"native": native}
        for name in ["reference", "symmetric"]:
            paths[name] = (BENCH / "predictions" / f"{name}-seed2-selected-validation.npz"
                           if split == "val" else BENCH / "results" / f"{name}-seed2-test.npz")
        scores = {name: predictions(path, rows) for name, path in paths.items()}
        groups = {"all": np.ones(len(rows), dtype=bool),
                  "le_2193": combined <= 2193, "gt_2193": combined > 2193}
        result["existing_prediction_strata"][split] = {
            group: {name: metrics(y[mask], score[mask]) for name, score in scores.items()}
            for group, mask in groups.items()
        }
    for name in ["reference-seed2", "symmetric-seed2"]:
        directory = ROOT / "retrain-v1/runs" / name
        # Copy bytes to this audit directory so the time-dependent observations
        # remain reproducible after the live log has grown.
        raw = (directory / "events.jsonl").read_bytes()
        lines = raw.splitlines(keepends=True)
        if lines and not lines[-1].endswith(b"\n"):
            lines = lines[:-1]
        snapshot = OUT / f"{name}-events-snapshot.jsonl"
        snapshot.write_bytes(b"".join(lines))
        record(snapshot)
        events = [json.loads(line) for line in lines]
        initial = next(e for e in events if e["event"] == "initialized")
        updates = [e for e in events if e["event"] == "update"]
        validations = [e for e in events if e["event"] == "validation"]
        last = updates[-1]
        hours = (datetime.datetime.fromisoformat(last["time_utc"]) -
                 datetime.datetime.fromisoformat(initial["time_utc"])).total_seconds() / 3600
        best_update = max((e for e in validations if e.get("best")),
                          key=lambda e: e["update"])["update"]
        result["training_snapshot"][name] = {
            "latest_logged_update": last["update"], "latest_log_time_utc": last["time_utc"],
            "elapsed_hours": hours, "world_size": initial["world_size"],
            "best_selected_update": best_update, "latest_validation": validations[-1],
            "latest_training_bce": last["mean_bce"],
            "max_logged_rank0_peak_allocated_gib": max(e.get("gpu_peak_allocated_gib", 0) for e in updates),
            "projected_five_epoch_wall_hours_linear": hours * 12745 / last["update"],
            "projection_caveat": "Estimate only: linear scaling includes validations and checkpoints so far, but length mix and remaining overhead may differ. Rank-zero allocated memory is not all-rank reserved memory."
        }
    result["inputs"] = INPUTS
    (OUT / "existing-data-audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ["created_utc", "splits", "existing_prediction_strata", "training_snapshot"]}, indent=2))


if __name__ == "__main__":
    main()
