"""Extend every main figure using the two completed historical v2 predictors."""
import csv
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
import analyze
from bench_utils import ROOT, atomic, load_npz, now, read, record
from collect import BASE_NAMES, NAMES, LABELS, TESTS, verify

EXT = ROOT / 'v1-v4-comparison'
ADDED = ['v2-capped', 'v2-clean-bce']
ADDED_LABELS = {'v2-capped': 'V2 length-capped (u8000)',
                'v2-clean-bce': 'V2 clean BCE (u4000)'}
FIGURE_NAMES = NAMES + ADDED
ARCHIVE = ROOT / 'archive/before-xpair-v11'


def csvrows(path):
    with Path(path).open() as stream:
        return list(csv.DictReader(stream))


def main():
    threadpool_limits(1)
    original = read(ARCHIVE / 'results/COMPLETE.json')
    assert original['complete'] and original['models'] == BASE_NAMES
    # Verify the sealed historical inputs at their preserved archive locations.
    for item in original['artifacts']:
        relative = Path(item['path']).resolve().relative_to(ROOT)
        verify({**item, 'path': str(ARCHIVE / relative)})
    archived = read(ARCHIVE / 'archive.json')
    for relative, item in archived['files'].items():
        verify({**item, 'path': str(ARCHIVE / relative)})
    historical = read(EXT / 'results/COMPLETE.json')
    assert historical['complete'] and historical['rows_per_model'] == 242000
    for item in list(historical['artifacts'].values()) + [historical['report'], historical['summary'], historical['analysis_code']]:
        verify(item)
    summary = read(ROOT / 'results/summary.json')
    history = read(EXT / 'results/summary.json')
    collection = read(ROOT / 'results/collection.json')
    assert summary['complete'] and collection['complete']
    assert history['bootstrap']['replicates'] == analyze.REPLICATES == 1000
    assert history['bootstrap']['seed'] == analyze.SEED == 20261005
    prediction_files = {n: verify(collection['models'][n]['file']) for n in NAMES}
    for name in ADDED:
        prediction_files[name] = verify(history['prediction_provenance'][name]['file'])
    scores = {name: load_npz(path)['scores'] for name, path in prediction_files.items()}
    union = np.load(ROOT / 'data/union.npy')
    assert all(z.shape == (len(union),) and np.isfinite(z).all() for z in scores.values())
    mapping = load_npz(ROOT / 'data/pair-mapping.npz')
    meta = read(ROOT / 'data/sequences.json')
    flags = load_npz(ROOT / 'provenance/exposure-flags.npz')
    known = np.zeros(len(meta['sequence']), dtype=bool)
    human = known.copy()
    for key in read(ROOT / 'provenance/exposure.json')['audits']:
        known |= flags[key + '__endpoints'] > 0
        if not key.startswith('xpair-default'):
            human |= flags[key + '__endpoints'] > 0
    v11 = read(ROOT / 'provenance/xpair-v11-exposure.json')
    assert v11['existing_human_and_all_source_masks_cover_all_identified_exposure']
    v11_flags = load_npz(verify(v11['flags']))
    for mode in ['exact', 'ankh-normalized']:
        seen = v11_flags[mode + '__endpoints'] > 0
        assert not (seen & ~human).any() and not (seen & ~known).any()
    historical_flags = load_npz(EXT / 'provenance/exposure-flags.npz')
    for name in ADDED:
        seen = historical_flags[name + '__endpoints'] > 0
        assert not (seen & ~human).any() and not (seen & ~known).any()
    old_subsets = csvrows(ROOT / 'results/subsets.csv')
    historical_subsets = csvrows(EXT / 'results/subsets.csv')
    historical_ci = csvrows(EXT / 'results/confidence-intervals.csv')
    records, intervals, testdata = [], [], {}
    sources = list(prediction_files.values())
    for test in TESTS:
        rows = np.load(ROOT / 'data' / (test + '.npy'))
        ids = mapping[test]
        y = rows[:, 2]
        assert np.array_equal(np.sort(rows[:, :2], axis=1), union[ids, :2])
        assert np.array_equal(rows[:, 3], np.arange(len(rows)))
        values = {n: {'scores': scores[n][ids]} for n in FIGURE_NAMES}
        testdata[test] = {'rows': rows, 'data': values}
        masks = {'full': np.ones(len(rows), dtype=bool),
                 'common_known_endpoint_unexposed': ~known[rows[:, :2]].any(axis=1),
                 'common_human_sources_endpoint_unexposed': ~human[rows[:, :2]].any(axis=1)}
        for subset, mask in masks.items():
            assert len(np.unique(y[mask])) == 2
            for name in FIGURE_NAMES:
                z = values[name]['scores'][mask]
                item = {'test': test, 'subset': subset, 'model': name,
                        'rows': int(mask.sum()), 'positives': int(y[mask].sum()),
                        'negatives': int((1-y[mask]).sum()), 'prevalence': float(y[mask].mean()),
                        'ap': float(average_precision_score(y[mask], z)),
                        'auroc': float(roc_auc_score(y[mask], z))}
                if subset == 'full':
                    prior = (summary['tests'][test]['models'][name] if name in NAMES else
                             next(r for r in history['metrics'] if r['test'] == test and r['model'] == name))
                else:
                    candidates = old_subsets if name in NAMES else historical_subsets
                    matches = [r for r in candidates if r['test'] == test and r['model'] == name and r['subset'] == subset]
                    prior = matches[0] if matches else None
                if prior is not None:
                    assert all(abs(item[k] - float(prior[k])) < 1e-12 for k in ['ap', 'auroc'])
                    if 'rows' in prior:
                        assert item['rows'] == int(prior['rows']) and item['positives'] == int(prior['positives'])
                records.append(item)
        base_boot = load_npz(ROOT / 'results' / (test + '-bootstrap.npz'))
        hist_boot = load_npz(EXT / 'results' / (test + '-bootstrap.npz'))
        # Shared models independently demonstrate identical bootstrap draws and arithmetic.
        for name in set(base_boot['names']) & set(hist_boot['names']):
            left = base_boot['samples'][:, base_boot['names'].tolist().index(name)]
            right = hist_boot['samples'][:, hist_boot['names'].tolist().index(name)]
            assert np.array_equal(left, right), (test, name)
        for name in FIGURE_NAMES:
            boot = base_boot if name in NAMES else hist_boot
            samples = boot['samples'][:, boot['names'].tolist().index(name)]
            point = next(r for r in records if r['test'] == test and r['model'] == name and r['subset'] == 'full')
            for k, metric in enumerate(['ap', 'auroc']):
                prior = next(r for r in (summary['confidence_intervals'] if name in NAMES else historical_ci)
                             if r['test'] == test and r['model'] == name and r['metric'] == metric)
                low, high = np.quantile(samples[:, k], [.025, .975])
                assert abs(float(prior['estimate']) - point[metric]) < 1e-12
                assert abs(float(prior['low']) - low) < 1e-12 and abs(float(prior['high']) - high) < 1e-12
                intervals.append({'test': test, 'model': name, 'metric': metric,
                                  'estimate': point[metric], 'low': float(low), 'high': float(high)})
        sources.extend([ROOT / 'data' / (test + '.npy'), ROOT / 'results' / (test + '-bootstrap.npz'),
                        EXT / 'results' / (test + '-bootstrap.npz')])
    # Reuse the established plotting function; its statistical implementation stays frozen.
    analyze.NAMES = FIGURE_NAMES
    analyze.LABELS = {**LABELS, **ADDED_LABELS}
    analyze.COLORS = analyze.COLORS + ['#b12879', '#159b9c']
    analyze.plots({}, intervals, testdata, records)
    analyze.csvfile(ROOT / 'results/combined-figure-metrics.csv', records)
    analyze.csvfile(ROOT / 'results/combined-figure-confidence-intervals.csv', intervals)
    sources += [ROOT / 'results/summary.json', ROOT / 'results/collection.json',
                ROOT / 'results/subsets.csv', ROOT / 'data/union.npy', ROOT / 'data/pair-mapping.npz',
                ROOT / 'data/sequences.json', ROOT / 'provenance/exposure.json',
                ROOT / 'provenance/xpair-v11-exposure.json',
                ROOT / 'provenance/xpair-v11-exposure-flags.npz',
                ROOT / 'provenance/exposure-flags.npz', EXT / 'provenance/exposure.json',
                EXT / 'provenance/exposure-flags.npz', EXT / 'provenance/selection.json',
                EXT / 'results/summary.json', EXT / 'results/COMPLETE.json',
                EXT / 'results/subsets.csv', EXT / 'results/confidence-intervals.csv']
    figures = sorted(p for p in (ROOT / 'results').iterdir() if p.suffix in ['.png', '.pdf', '.svg'])
    assert len(figures) == 11
    atomic(ROOT / 'provenance/historical-figure-extension.json', {
        'at_utc': now(), 'complete': True, 'models': FIGURE_NAMES,
        'labels': {**LABELS, **ADDED_LABELS}, 'added_models': ADDED,
        'historical_scope': 'The two previously requested and benchmarked v2 representatives from the v1-v4 comparison.',
        'rows_per_model': 242000, 'figure_groups': 4, 'figure_files': [record(p) for p in figures],
        'inference_repeated': False, 'bootstrap_recomputed': False,
        'checks': {'previous_numeric_results_unchanged': True, 'all_metrics_recomputed_from_saved_scores': True,
                   'existing_full_metrics_and_intervals_match': True, 'shared_bootstrap_samples_identical': True,
                   'common_exposure_mask_covers_both_historical_models': True, 'same_rows_for_all_models': True},
        'metrics': record(ROOT / 'results/combined-figure-metrics.csv'),
        'intervals': record(ROOT / 'results/combined-figure-confidence-intervals.csv'),
        'script': record(__file__), 'plot_code': record(ROOT / 'scripts/analyze.py'),
        'sources': [record(p) for p in dict.fromkeys(sources)],
        'previous_figures': record(ARCHIVE / 'archive.json')})
    print({'figure_models': len(FIGURE_NAMES), 'files_updated': len(figures),
           'metric_rows_checked': len(records), 'inference_repeated': False}, flush=True)


if __name__ == '__main__':
    main()
