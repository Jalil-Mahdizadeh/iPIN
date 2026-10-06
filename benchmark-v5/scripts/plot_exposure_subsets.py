"""Plot the two existing protein-exposure exclusions without new inference.

Run from the project root:
  bash benchmark-v5/scripts/container.sh analysis python scripts/plot_exposure_subsets.py
"""
import csv
import os

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from bench_utils import ROOT, atomic, load_npz, now, read, record, sha
from collect import LABELS, NAMES, verify

os.environ.setdefault('MPLCONFIGDIR', str(ROOT / 'cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


SPECS = [
    {
        'stem': 'xpair-exposed-sequences-removed',
        'subset': 'xpair_default_endpoint_unexposed',
        'flag': 'xpair-default__ankh-normalized__endpoints',
        'title': 'Performance after removing X-PAIR-exposed proteins',
        'source': 'Exposure source: X-PAIR default TRAIN + DEV, interaction and interface tasks',
        'scope': 'This filter removes X-PAIR default exposure; D-SCRIPT exposure can remain.',
    },
    {
        'stem': 'dscript-exposed-sequences-removed',
        'subset': 'dscript_endpoint_unexposed',
        'flag': 'dscript__exact__endpoints',
        'title': 'Performance after removing D-SCRIPT-exposed proteins',
        'source': 'Exposure source: D-SCRIPT original public human training data',
        'scope': 'This also removes both added human releases’ exact TRAIN/validation exposure; X-PAIR exposure can remain.',
    },
]
TESTS = ['original', 'ilp']
TEST_LABELS = {'original': 'Original Bernett', 'ilp': 'Custom ILP negatives'}
COLORS = {'original': '#0072B2', 'ilp': '#D55E00'}
MARKERS = {'original': 'o', 'ilp': 'D'}
OFFSETS = {'original': -.14, 'ilp': .14}


def verified_tables():
    """Reconstruct each mask and confirm all plotted metrics from saved scores."""
    with (ROOT / 'results/subsets.csv').open() as stream:
        csv_rows = list(csv.DictReader(stream))
    collection = read(ROOT / 'results/collection.json')
    assert collection['complete'] and set(collection['models']) == set(NAMES)
    predictions = {
        name: load_npz(verify(collection['models'][name]['file']))['scores']
        for name in NAMES
    }
    flags = load_npz(verify(read(ROOT / 'provenance/exposure.json')['flags']))
    mapping = load_npz(ROOT / 'data/pair-mapping.npz')
    union = np.load(ROOT / 'data/union.npy')
    tables = {}
    for spec in SPECS:
        tables[spec['subset']] = {}
        exposure = flags[spec['flag']] > 0
        for test in TESTS:
            rows = np.load(ROOT / 'data' / f'{test}.npy')
            ids = mapping[test]
            assert np.array_equal(rows[:, 2], union[ids, 2])
            assert np.array_equal(np.sort(rows[:, :2], axis=1), np.sort(union[ids, :2], axis=1))
            keep = ~exposure[rows[:, :2]].any(axis=1)
            assert not exposure[rows[keep, :2]].any()
            y = rows[keep, 2]
            assert len(np.unique(y)) == 2
            selected = [r for r in csv_rows if r['subset'] == spec['subset'] and r['test'] == test]
            assert len(selected) == len(NAMES) and {r['model'] for r in selected} == set(NAMES)
            counts = {
                'pairs': int(keep.sum()), 'positives': int(y.sum()),
                'negatives': int(len(y) - y.sum()), 'prevalence': float(y.mean()),
                'unique_proteins': int(len(np.unique(rows[keep, :2]))),
            }
            values = {}
            for row in selected:
                name = row['model']
                assert int(row['rows']) == counts['pairs'] and int(row['positives']) == counts['positives']
                assert abs(float(row['prevalence']) - counts['prevalence']) < 1e-12
                scores = predictions[name][ids[keep]]
                recomputed = {
                    'ap': float(average_precision_score(y, scores)),
                    'auroc': float(roc_auc_score(y, scores)),
                }
                assert all(abs(recomputed[k] - float(row[k])) < 1e-12 for k in recomputed)
                values[name] = {k: float(row[k]) for k in ['ap', 'auroc']}
            tables[spec['subset']][test] = {'counts': counts, 'models': values}
    return tables


def plot(spec, data):
    plt.rcParams.update({
        'font.family': 'DejaVu Sans', 'font.size': 10,
        'axes.titlesize': 13, 'axes.labelsize': 11,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'axes.edgecolor': '#B8BEC7', 'text.color': '#202938',
        'axes.labelcolor': '#202938', 'xtick.color': '#526071',
        'ytick.color': '#202938', 'savefig.facecolor': 'white',
    })
    fig, axes = plt.subplots(1, 2, figsize=(14.3, 9.0), sharey=True)
    fig.subplots_adjust(left=.205, right=.975, bottom=.205, top=.765, wspace=.15)
    fig.suptitle(spec['title'], x=.025, y=.977, ha='left', fontsize=19, weight='bold')
    fig.text(.025, .928, spec['source'], fontsize=11, color='#526071')
    handles = []
    for test in TESTS:
        c = data[test]['counts']
        label = (f"{TEST_LABELS[test]}  |  {c['pairs']:,} pairs  |  "
                 f"{c['positives']:,} positive ({c['prevalence']:.1%})")
        handles.append(Line2D([], [], linestyle='none', marker=MARKERS[test],
                              color=COLORS[test], markersize=7, label=label))
    fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(.018, .906),
               frameon=False, fontsize=10.5, handletextpad=.6, labelspacing=.75)
    y = np.arange(len(NAMES))
    for ax, metric, title in zip(axes, ['ap', 'auroc'], ['Average precision (AP)', 'AUROC']):
        for j in y[::2]:
            ax.axhspan(j-.48, j+.48, color='#F4F6F8', zorder=0)
        for j, name in enumerate(NAMES):
            endpoints = [data[t]['models'][name][metric] for t in TESTS]
            ax.plot(endpoints, [j+OFFSETS[t] for t in TESTS], color='#C2C8D0', linewidth=1.1, zorder=2)
            for test in TESTS:
                value = data[test]['models'][name][metric]
                assert 0 <= value <= 1
                ax.scatter(value, j+OFFSETS[test], color=COLORS[test], marker=MARKERS[test],
                           s=42, edgecolors='white', linewidths=.5, zorder=3)
                ax.text(value+.009, j+OFFSETS[test], f'{value:.4f}', va='center',
                        fontsize=9, color=COLORS[test], zorder=4)
        if metric == 'ap':
            for test in TESTS:
                ax.axvline(data[test]['counts']['prevalence'], color=COLORS[test],
                           linestyle=(0, (3, 3)), linewidth=1, alpha=.65, zorder=1)
        else:
            ax.axvline(.5, color='#687586', linestyle=(0, (3, 3)), linewidth=1, zorder=1)
        ax.set(title=title, xlabel='Score (higher is better)', xlim=(0., 1.), ylim=(len(NAMES)-.55, -.6))
        ax.set_xticks(np.arange(0., 1.01, .20))
        ax.set_axisbelow(True)
        ax.grid(axis='x', color='#E4E8ED', linewidth=.7)
        ax.spines[['top', 'right', 'left']].set_visible(False)
        ax.tick_params(axis='y', length=0, pad=10)
    axes[0].set_yticks(y, [LABELS[n].rstrip('*') for n in NAMES])
    for label in axes[0].get_yticklabels():
        if label.get_text().startswith('iPIN v5'):
            label.set_weight('bold')
    fig.text(.025, .127, 'Retained pairs have neither protein in the named exposure list; all eleven models use the same rows within each test.', fontsize=10)
    fig.text(.025, .093, 'Dashed lines: AP = subset positive fraction; AUROC = 0.5. Point estimates only; no confidence intervals.', fontsize=10, color='#526071')
    fig.text(.025, .059, spec['scope'] + ' Homology and pretraining exposure are not filtered.', fontsize=9.5, color='#526071')
    files = []
    for extension in ['png', 'pdf', 'svg']:
        path = ROOT / 'results' / f"{spec['stem']}.{extension}"
        fig.savefig(path, dpi=240)
        files.append(record(path))
    plt.close(fig)
    return files


def main():
    tables = verified_tables()
    figures = []
    for spec in SPECS:
        files = plot(spec, tables[spec['subset']])
        figures.append({**spec, 'data': tables[spec['subset']], 'files': files})
    sources = ['results/subsets.csv', 'results/collection.json', 'provenance/exposure.json',
               'provenance/exposure-flags.npz', 'data/pair-mapping.npz',
               'data/original.npy', 'data/ilp.npy', 'data/union.npy']
    atomic(ROOT / 'provenance/exposure-subset-figures.json', {
        'at_utc': now(), 'script': record(__file__), 'sources': [record(ROOT / s) for s in sources],
        'checks': {'all_models': True, 'same_rows_within_each_test': True,
                   'no_retained_exposed_endpoints': True, 'all_metrics_recomputed_and_match_csv': True},
        'inference_repeated': False, 'confidence_intervals': False, 'figures': figures,
    })
    print({'created_figures': len(figures), 'formats': ['png', 'pdf', 'svg'],
           'metrics_verified': 8*len(NAMES), 'inference_repeated': False}, flush=True)


if __name__ == '__main__':
    main()
