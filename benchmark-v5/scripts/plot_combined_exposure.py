"""Compare all eleven frozen predictors after excluding either source's proteins."""
import csv

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from bench_utils import ROOT, atomic, load_npz, now, read, record, save_npz
from collect import NAMES, verify
from plot_exposure_subsets import plot


def main():
    collection = read(ROOT / 'results/collection.json')
    assert collection['complete'] and set(collection['models']) == set(NAMES)
    predictions = {
        name: load_npz(verify(collection['models'][name]['file']))['scores']
        for name in NAMES
    }
    exposure_report = read(ROOT / 'provenance/exposure.json')
    flags = load_npz(verify(exposure_report['flags']))
    dscript = flags['dscript__exact__endpoints'] > 0
    xpair = flags['xpair-default__ankh-normalized__endpoints'] > 0
    assert np.array_equal(xpair, flags['xpair-default__exact__endpoints'] > 0)
    exposure = dscript | xpair
    mapping = load_npz(ROOT / 'data/pair-mapping.npz')
    union = np.load(ROOT / 'data/union.npy')
    tables, csv_rows, retained, positives = {}, [], {}, {}
    for test in ['original', 'ilp']:
        rows = np.load(ROOT / 'data' / f'{test}.npy')
        ids = mapping[test]
        assert np.array_equal(rows[:, 2], union[ids, 2])
        assert np.array_equal(np.sort(rows[:, :2], axis=1), np.sort(union[ids, :2], axis=1))
        keep = ~exposure[rows[:, :2]].any(axis=1)
        assert np.array_equal(keep, (~dscript[rows[:, :2]].any(axis=1)) & (~xpair[rows[:, :2]].any(axis=1)))
        kept = rows[keep]
        assert not dscript[kept[:, :2]].any() and not xpair[kept[:, :2]].any()
        y = kept[:, 2]
        assert len(np.unique(y)) == 2
        counts = {
            'pairs': len(kept), 'positives': int(y.sum()), 'negatives': int((y == 0).sum()),
            'prevalence': float(y.mean()), 'unique_proteins': int(len(np.unique(kept[:, :2]))),
            'fraction_of_full_test': float(keep.mean()),
        }
        assert counts['pairs'] == {'original': 1073, 'ilp': 1057}[test]
        assert counts['positives'] == 566
        retained[test + '_row_ids'] = np.flatnonzero(keep)
        retained[test + '_union_ids'] = ids[keep]
        positives[test] = {tuple(sorted(map(int, row[:2]))) for row in kept if row[2] == 1}
        values = {}
        for name in NAMES:
            scores = predictions[name][ids[keep]]
            assert scores.shape == y.shape and np.isfinite(scores).all()
            values[name] = {
                'ap': float(average_precision_score(y, scores)),
                'auroc': float(roc_auc_score(y, scores)),
            }
            csv_rows.append({'test': test, 'model': name, **counts, **values[name]})
        tables[test] = {'counts': counts, 'models': values}
    assert positives['original'] == positives['ilp']
    assert len(csv_rows) == 2*len(NAMES)
    csv_path = ROOT / 'results/combined-exposure-removed-metrics.csv'
    with csv_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    retained_path = ROOT / 'results/combined-exposure-removed-rows.npz'
    save_npz(retained_path, **retained)
    spec = {
        'stem': 'combined-exposed-sequences-removed',
        'title': 'Performance after excluding D-SCRIPT and X-PAIR exposure',
        'source': 'Exclude proteins in D-SCRIPT human TRAIN or X-PAIR default TRAIN + DEV (interaction / interface)',
        'scope': 'Both exposure lists are applied together; about 2% of each full test remains.',
    }
    figure_files = plot(spec, tables)
    sources = ['results/collection.json', 'provenance/exposure.json', 'provenance/exposure-flags.npz',
               'data/pair-mapping.npz', 'data/original.npy', 'data/ilp.npy', 'data/union.npy']
    atomic(ROOT / 'provenance/combined-exposure-subset-figure.json', {
        'at_utc': now(), 'script': record(__file__),
        'plot_helper': record(ROOT / 'scripts/plot_exposure_subsets.py'),
        'sources': [record(ROOT / source) for source in sources],
        'exclusion_rule': 'Remove a pair if either endpoint is exposed to D-SCRIPT OR X-PAIR default.',
        'exposed_sequences': {'dscript': int(dscript.sum()), 'xpair_default': int(xpair.sum()),
                              'union': int(exposure.sum()), 'remaining': int((~exposure).sum())},
        'checks': {'all_models': True, 'same_rows_within_each_test': True,
                   'no_retained_endpoints_exposed_to_either_source': True,
                   'previously_reported_counts_verified': True, 'positive_pairs_identical_between_tests': True},
        'data': tables, 'metrics_csv': record(csv_path), 'retained_rows': record(retained_path),
        'figure': {'spec': spec, 'files': figure_files},
        'inference_repeated': False, 'confidence_intervals': False,
    })
    print({'figure': spec['stem'], 'test_counts': {t: x['counts'] for t, x in tables.items()},
           'model_test_evaluations': len(csv_rows), 'inference_repeated': False}, flush=True)


if __name__ == '__main__':
    main()
