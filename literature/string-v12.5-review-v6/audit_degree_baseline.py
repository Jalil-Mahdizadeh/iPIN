"""TRAIN-only identity/degree diagnostic on released human validation, no PLM or fitting."""
from collections import Counter
import csv
import hashlib
import json
import math

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from audit_sources import HERE, TUNA


def rows(split):
    path = TUNA / f"data/processed/xspecies/human_{split}_dictionary.tsv"
    seq = {k: hashlib.sha256(s.encode()).hexdigest()
           for k, s in csv.reader(path.open(), delimiter="\t")}
    path = TUNA / f"data/processed/xspecies/human_{split}_interaction.tsv"
    return [(*sorted((seq[a], seq[b])), int(y))
            for a, b, y in csv.reader(path.open(), delimiter="\t")]


train, dev = rows("train"), rows("test")
degrees = {0: Counter(), 1: Counter()}
for a, b, y in train:
    degrees[y].update({a, b})
score = [math.log((degrees[1][a] + 1) / (degrees[0][a] + 1))
         + math.log((degrees[1][b] + 1) / (degrees[0][b] + 1)) for a, b, _ in dev]
simple = [int(a in degrees[1]) + int(b in degrees[1]) for a, b, _ in dev]
y = np.asarray([r[2] for r in dev])
seen = {(a, b) for a, b, _ in train}
subsets = {"all_released_validation": np.ones(len(dev), dtype=bool),
           "excluding_exact_train_pair_overlap": np.asarray([(a, b) not in seen for a, b, _ in dev])}
out = {"definition": "sum over endpoints log((TRAIN-positive row-degree+1)/(TRAIN-negative row-degree+1)); self pairs counted once; sequence SHA identities",
       "selection": "No fitting, tuning, PLM or test-species information; degree diagnostic fixed in this script.",
       "limitation": "This is human warm-protein validation; no identity lookup transfer claim to nonhuman proteins.",
       "results": {}}
for name, mask in subsets.items():
    out["results"][name] = {"rows": int(mask.sum()), "positives": int(y[mask].sum())}
    for label, pred in [("smoothed_degree_ratio", score), ("number_of_endpoints_seen_in_train_positives", simple)]:
        pred = np.asarray(pred)
        out["results"][name][label] = {"ap": average_precision_score(y[mask], pred[mask]),
                                       "auroc": roc_auc_score(y[mask], pred[mask])}
print(json.dumps(out, indent=2))
