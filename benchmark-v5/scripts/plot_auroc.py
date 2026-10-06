"""Plot saved AUROC estimates and protein-bootstrap intervals in the AP figure style."""
import csv
import os

import numpy as np

from bench_utils import ROOT, atomic, load_npz, now, read, record
from collect import LABELS, NAMES


def main():
    summary = read(ROOT / 'results/summary.json')
    assert summary['complete'] and summary['names'] == NAMES
    with (ROOT / 'results/confidence-intervals.csv').open() as stream:
        rows = [row for row in csv.DictReader(stream) if row['metric'] == 'auroc']
    assert len(rows) == 2 * len(NAMES)
    data = []
    bootstrap_files = []
    for test in ['original', 'ilp']:
        path = ROOT / 'results' / f'{test}-bootstrap.npz'
        boot = load_npz(path)
        assert boot['names'].tolist() == NAMES
        metric_index = boot['metrics'].tolist().index('auroc')
        bootstrap_files.append(record(path))
        for j, name in enumerate(NAMES):
            selected = [row for row in rows if row['test'] == test and row['model'] == name]
            assert len(selected) == 1, (test, name)
            row = selected[0]
            item = {key: row[key] for key in ['test', 'model', 'metric']}
            item.update({key: float(row[key]) for key in ['estimate', 'low', 'high']})
            assert item in summary['confidence_intervals']
            assert abs(item['estimate'] - summary['tests'][test]['models'][name]['auroc']) < 1e-12
            assert np.array_equal(
                [item['low'], item['high']],
                np.quantile(boot['samples'][:, j, metric_index], [.025, .975]),
            )
            assert 0 <= item['low'] <= item['estimate'] <= item['high'] <= 1
            item['label'] = f"{item['estimate']:.4f}"
            data.append(item)

    os.environ['MPLCONFIGDIR'] = str(ROOT / 'cache/matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    colors = ['#202020', '#c44e52', '#4c72b0', '#55a868', '#8172b3', '#ccb974',
              '#64b5cd', '#8c6143', '#da8bc3', '#e17c05', '#146b6b', '#b33f88']
    labels = [LABELS[name] for name in NAMES]
    assert len(colors) == len(NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5), sharey=True)
    for ax, test in zip(axes, ['original', 'ilp']):
        for j, name in enumerate(NAMES):
            item = next(row for row in data if row['test'] == test and row['model'] == name)
            value = item['estimate']
            ax.errorbar(value, j, xerr=[[value - item['low']], [item['high'] - value]],
                        fmt='o', color=colors[j], capsize=3)
            ax.annotate(item['label'], xy=(item['high'], j), xytext=(7, 0),
                        textcoords='offset points', ha='left', va='center',
                        fontsize=9, color='#202020')
        ax.set_title('Original Bernett test' if test == 'original' else 'Custom ILP-negative test')
        ax.set_xlabel('AUROC (95% protein-bootstrap interval)')
        ax.grid(axis='x', alpha=.2)
        ax.axvline(.5, color='gray', linestyle=':', linewidth=1)
        lower, upper = ax.get_xlim()
        ax.set_xlim(lower, upper + .075 * (upper - lower))
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].invert_yaxis()
    fig.suptitle('Frozen checkpoints | same positives, different negative distributions')
    fig.text(.02, .015,
             '* Documented supervised-data overlap; common-subset results are reported separately.',
             fontsize=9, color='#526071')
    fig.tight_layout(rect=(0, .035, 1, 1))
    files = []
    for ext in ['png', 'pdf']:
        path = ROOT / 'results' / f'auroc.{ext}'
        fig.savefig(path, dpi=180, bbox_inches='tight')
        files.append(record(path))
    plt.close(fig)

    atomic(ROOT / 'provenance/auroc-figure.json', {
        'at_utc': now(), 'script': record(__file__), 'models': NAMES, 'labels': labels,
        'metric': 'auroc', 'point_estimate_decimal_places': 4,
        'sources': [record(ROOT / 'results/summary.json'),
                    record(ROOT / 'results/confidence-intervals.csv'), *bootstrap_files],
        'data': data, 'figures': files,
        'checks': {'point_estimates_match_saved_summary': True,
                   'intervals_match_saved_csv_and_bootstrap_quantiles': True},
        'inference_repeated': False, 'bootstrap_recomputed': False,
    })
    print({'created': [item['path'] for item in files], 'point_labels': len(data)}, flush=True)


if __name__ == '__main__':
    main()
