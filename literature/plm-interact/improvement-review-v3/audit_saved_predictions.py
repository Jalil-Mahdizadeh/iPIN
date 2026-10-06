"""CPU-only descriptive audit of frozen, already evaluated predictions.

No new inference, checkpoint selection, fitting, blending, or test optimization.
Run from the repository root with benchmark-v2/scripts/container.sh python PATH.
"""
import csv
import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
NAMES = ['native-bernett', 'v1-reference', 'v1-symmetric', 'v2-reference',
         'v2-capped', 'v2-positive10', 'v2-clean-bce']


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2 ** 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    out = dict(created_utc=datetime.now(timezone.utc).isoformat(),
               purpose='Post-hoc descriptive audit; historical test is not independent confirmation',
               fitting=False, inference=False, checkpoint_selection=False,
               sources={}, splits={})
    macro_rows, comparison_rows = [], []
    for split in ['val', 'test']:
        p = ROOT / f'benchmark-v2/data/{split}.npy'
        data = np.load(p, allow_pickle=False)
        out['sources'][str(p.relative_to(ROOT))] = sha(p)
        y = data[:, 2].astype(int)
        scores = {}
        for name in NAMES:
            p = ROOT / f'benchmark-v2/results/{name}-{split}.npz'
            a = np.load(p, allow_pickle=False)['predictions']
            a = a[np.argsort(a[:, 0])]
            assert np.array_equal(a[:, 0], np.arange(len(y)))
            assert np.array_equal(a[:, 1], y) and np.isfinite(a).all()
            scores[name] = a[:, 2:4].mean(axis=1)
            out['sources'][str(p.relative_to(ROOT))] = sha(p)
        rows_by_protein = {}
        for i, (a, b) in enumerate(data[:, :2]):
            for protein in set([int(a), int(b)]):
                rows_by_protein.setdefault(protein, []).append(i)
        eligible = {p: np.array(ix) for p, ix in rows_by_protein.items()
                    if y[ix].sum() >= 2 and (1-y[ix]).sum() >= 2}
        rows_covered = np.unique(np.concatenate(list(eligible.values())))
        result = dict(rows=len(y), proteins=len(rows_by_protein),
                      macro_rule='Equal protein weight; >=2 labeled positives AND >=2 labeled negatives; incident pairs only',
                      eligible_proteins=len(eligible), eligible_unique_rows=len(rows_covered),
                      note='Incident pairs are incomplete sampled candidate sets; proteins share edges. No independent protein p-values.',
                      models={}, complementarity={})
        for name in NAMES:
            ap, auc = [], []
            for p, ix in eligible.items():
                v = average_precision_score(y[ix], scores[name][ix])
                u = roc_auc_score(y[ix], scores[name][ix])
                ap.append(v)
                auc.append(u)
                macro_rows.append(dict(split=split, model=name, protein_id=p,
                    rows=len(ix), positives=int(y[ix].sum()), ap=float(v), auroc=float(u)))
            result['models'][name] = dict(micro_ap=float(average_precision_score(y, scores[name])),
                micro_auroc=float(roc_auc_score(y, scores[name])),
                macro_ap=float(np.mean(ap)), macro_auroc=float(np.mean(auc)))
        for a, b in itertools.combinations(NAMES, 2):
            record = dict(spearman_logit=float(spearmanr(scores[a], scores[b]).statistic))
            for k in [1000, 5000]:
                # Deterministic row-index tie break; this is a descriptive fixed-budget comparison.
                ia = np.lexsort((np.arange(len(y)), -scores[a]))[:k]
                ib = np.lexsort((np.arange(len(y)), -scores[b]))[:k]
                sa, sb = set(ia.tolist()), set(ib.tolist())
                record[f'top{k}'] = dict(overlap=len(sa & sb), precision_a=float(y[ia].mean()),
                    precision_b=float(y[ib].mean()),
                    positives_only_a=int(y[list(sa-sb)].sum()),
                    positives_only_b=int(y[list(sb-sa)].sum()))
            result['complementarity'][f'{a} | {b}'] = record
            comparison_rows.append(dict(split=split, a=a, b=b, **{k:v for k,v in record.items() if not isinstance(v,dict)}))
        out['splits'][split] = result
    p = ROOT / 'retrain-v2/analysis/completed-20260930T082601Z/validation-history.csv'
    out['sources'][str(p.relative_to(ROOT))] = sha(p)
    out['script_sha256'] = sha(Path(__file__).resolve())
    (HERE / 'saved-prediction-audit.json').write_text(json.dumps(out, indent=2, allow_nan=False)+'\n')
    for name, rows in [('protein-macro-metrics.csv', macro_rows), ('model-score-correlations.csv', comparison_rows)]:
        with (HERE / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps({s:dict(eligible_proteins=v['eligible_proteins'], models=v['models'],
        native_clean=v['complementarity']['native-bernett | v2-clean-bce'])
        for s,v in out['splits'].items()}, indent=2))


if __name__ == '__main__':
    main()
