"""Read-only interim validation audit; writes dated reports, never evaluates test data."""
import csv
import datetime
import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
started = datetime.datetime.now(datetime.timezone.utc)
OUT = ROOT / 'analysis' / started.strftime('progress-%Y%m%dT%H%M%SZ')
OUT.mkdir()

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''): h.update(block)
    return h.hexdigest()

def read_json(path): return json.loads(path.read_text())
def save_json(path, value): path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

manifest = read_json(ROOT / 'data/prepared/manifest.json')
val_path = ROOT / 'data/prepared/official/val.npy'
assert sha(val_path) == manifest['files']['official/val.npy']
val = np.load(val_path)
campaign = read_json(ROOT / 'configs/campaign.json')['groups']['initial-screen']
baseline_path = ROOT.parent / 'benchmark-v1/results/validation-metrics.csv'
with baseline_path.open() as handle:
    baseline = {r['model']: {key: float(r[key]) for key in ['ap', 'auroc', 'brier']}
                for r in csv.DictReader(handle) if r['view'] == 'pooled'}
histories, progress, predictions, audited = {}, {}, {}, []
all_rows = []
for name in campaign:
    run = ROOT / 'runs' / name
    raw_events = (run / 'events.jsonl').read_bytes()
    # Keep complete JSONL records if the active writer is between writes.
    raw_events = raw_events[:raw_events.rfind(b'\n') + 1]
    (OUT / f'{name}-events.jsonl').write_bytes(raw_events)
    events = [json.loads(line) for line in raw_events.splitlines()]
    updates = [e for e in events if e['event'] == 'update']
    validations = [e for e in events if e['event'] == 'validation']
    initialized = next(e for e in events if e['event'] == 'initialized')
    contract, latest, best = [read_json(run / f'{part}.json') for part in ['contract', 'latest', 'best']]
    assert not contract['qualification'] and len(val) == initialized['val_rows'] == 59258
    assert all(math.isfinite(e[k]) for e in updates for k in ['mean_loss', 'mean_bce', 'mean_mlm', 'grad_norm'])
    histories[name] = []
    predictions[name] = {}
    for event in validations:
        update = event['update']
        path = run / 'validation' / f'update-{update:09d}.npz'
        metadata = read_json(path.with_suffix('.json'))
        assert metadata['fingerprint'] == contract['fingerprint']
        assert sha(path) == metadata['sha256']
        with np.load(path) as values: array = values['predictions']
        assert array.shape == (len(val), 4) and np.isfinite(array).all()
        assert np.array_equal(array[:, 0], np.arange(len(val)))
        assert np.array_equal(array[:, 1], val[:, 2])
        score = array[:, 2:4].mean(1); prob = expit(score)
        measured = {'pooled_ap': float(average_precision_score(val[:, 2], score)),
                    'auroc': float(roc_auc_score(val[:, 2], score)),
                    'brier': float(np.mean((prob - val[:, 2])**2))}
        for key, value in measured.items():
            assert abs(value - metadata['metrics'][key]) < 1e-12
            assert abs(value - event['metrics'][key]) < 1e-12
        record = {'model': name, 'update': update, 'validation_utc': event['time_utc'], **measured,
                  'predicted_probability_mean': float(prob.mean()),
                  'mean_probability_true_positive': float(prob[val[:, 2] == 1].mean()),
                  'mean_probability_true_negative': float(prob[val[:, 2] == 0].mean()),
                  'fraction_predicted_positive_at_half': float((prob >= .5).mean())}
        histories[name].append(record); all_rows.append(record)
        predictions[name][update] = array
        audited.append({'path': str(path), 'sha256': metadata['sha256'], 'rows': len(val), 'metrics_recomputed': True})
    selected = max(histories[name], key=lambda item: item['pooled_ap'])
    assert best['update'] == selected['update'] and abs(best['best_ap'] - selected['pooled_ap']) < 1e-12
    assert latest['update'] >= best['update']
    assert (run / 'checkpoints' / latest['file']).stat().st_size == latest['bytes']
    assert (run / 'checkpoints' / best['file']).stat().st_size == best['bytes']
    most_recent = updates[-1]
    config = contract['configuration']
    progress[name] = {'current_logged_update': most_recent['update'],
        'planned_updates': config['total_updates'], 'progress_fraction': most_recent['update']/config['total_updates'],
        'last_logged_utc': most_recent['time_utc'], 'latest_committed_update': latest['update'],
        'latest_validation': histories[name][-1], 'selected_best': selected,
        'training_pairs': initialized['train_rows'], 'validation_pairs': len(val),
        'mean_step_seconds_last_100_logged': statistics.mean(e['seconds'] for e in updates[-100:]),
        'rank0_peak_allocated_gib': max(e['gpu_peak_allocated_gib'] for e in updates),
        'all_logged_losses_and_gradients_finite': True,
        'resume_events': sum(e['event'] == 'resumed' for e in events),
        'best_checkpoint': best, 'latest_checkpoint': latest}

common = max(set.intersection(*(set(predictions[name]) for name in campaign)))
matched = {name: next(r for r in histories[name] if r['update'] == common) for name in campaign}
reference = matched['reference-official-seed2']
for name, row in matched.items():
    row = dict(row)
    row['delta_ap_vs_reference'] = row['pooled_ap'] - reference['pooled_ap']
    row['delta_auroc_vs_reference'] = row['auroc'] - reference['auroc']
    row['delta_brier_vs_reference'] = row['brier'] - reference['brier']
    matched[name] = row

for filename, records in [('validation-history.csv', all_rows), ('matched-update.csv', list(matched.values()))]:
    with (OUT / filename).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
names = {'reference-official-seed2': 'Reference', 'capped-official-seed2': 'Length-capped',
         'positive10-official-seed2': 'Positive weight 10', 'clean-bce-official-seed2': 'Clean BCE'}
colors = ['#2563eb', '#d97706', '#059669', '#9333ea']
fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
for ax, metric, title in zip(axes, ['pooled_ap', 'auroc', 'brier'], ['Validation AP (higher is better)', 'Validation AUROC (higher is better)', 'Brier (lower is better)']):
    for name, color in zip(campaign, colors):
        records = histories[name]
        ax.plot([r['update'] for r in records], [r[metric] for r in records], marker='o', lw=1.8, ms=4,
                label=names[name], color=color)
    key = 'ap' if metric == 'pooled_ap' else metric
    ax.axhline(baseline['native-bernett'][key], color='#303030', ls='--', lw=1.1, label='Released native (historical validation)')
    ax.set_title(title, fontsize=10); ax.set_xlabel('Optimizer updates'); ax.grid(alpha=.18)
    ax.ticklabel_format(style='plain', axis='x')
    ax.axvline(common, color='#999999', lw=.8, ls=':', alpha=.7)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .015), ncol=3, fontsize=9, frameon=False)
fig.suptitle('V2 interim results: same 59,258 validation pairs, pooled A–B/B–A logits', fontsize=12)
fig.tight_layout(rect=(0, .16, 1, .93))
fig.savefig(OUT / 'validation-progress.png', dpi=180)
fig.savefig(OUT / 'validation-progress.svg')
plt.close(fig)

local = started.astimezone(ZoneInfo('Europe/Stockholm'))
summary = {'snapshot_utc': started.isoformat(), 'snapshot_stockholm': local.isoformat(),
           'validation_pairs': len(val), 'positive_fraction': float(val[:, 2].mean()),
           'progress': progress, 'common_validated_update': common, 'matched_comparison': matched,
           'historical_validation_baselines': baseline, 'baseline_csv_sha256': sha(baseline_path),
           'audited_prediction_files': audited, 'no_test_predictions_read': True,
           'training_or_selection_modified': False, 'confidence_intervals_computed': False}
save_json(OUT / 'snapshot.json', summary)
lines = ['**V2 interim retraining results**', '',
         f'Snapshot: {local.strftime("%d %B %Y, %H:%M:%S %Z")} ({started.strftime("%H:%M:%S UTC")}).', '',
         'All figures below are validation results on the same 59,258 pairs, using clean sequences, BF16 inference and mean A–B/B–A logits. These are interim point estimates; no test inference or model-selection changes were performed.', '',
         '| Model | Logged training update / 12,745 | Latest validation update | Validation AP | AUROC | Brier ↓ |',
         '| --- | ---: | ---: | ---: | ---: | ---: |']
for name in campaign:
    p = progress[name]; v = p['latest_validation']
    lines.append(f"| {names[name]} | {p['current_logged_update']:,} | {v['update']:,} | {v['pooled_ap']:.6f} | {v['auroc']:.6f} | {v['brier']:.6f} |")
latest_are_best = all(p['latest_validation']['update'] == p['selected_best']['update'] for p in progress.values())
selection_note = ('For all four runs, the latest completed validation is also their best AP so far.' if latest_are_best else
                  'Latest-validation rows and best-so-far selections are recorded separately in snapshot.json; they are not necessarily the same checkpoint.')
lines += ['', selection_note + ' Compare at common optimizer updates to separate training progress from model differences.', '',
          f'**Matched comparison at update {common:,}**', '',
          '| Model | AP | AP difference from reference | AUROC | Brier ↓ |',
          '| --- | ---: | ---: | ---: | ---: |']
for name in campaign:
    v = matched[name]
    lines.append(f"| {names[name]} | {v['pooled_ap']:.6f} | {v['delta_ap_vs_reference']:+.6f} | {v['auroc']:.6f} | {v['brier']:.6f} |")
cap = progress['capped-official-seed2']['latest_validation']
clean = matched['clean-bce-official-seed2']; weighted = matched['positive10-official-seed2']
leader = max(matched, key=lambda name: matched[name]['pooled_ap'])
lines += ['', '**Interpretation**', '',
          f"- {names[leader]} has the highest AP at the common {common:,}-update checkpoint. Clean BCE's differences from reference are AP {clean['delta_ap_vs_reference']:+.6f} and AUROC {clean['delta_auroc_vs_reference']:+.6f}. The clean-BCE comparison tests the combined removal of masking and MLM and does not separate their effects. No significance or seed-robustness claim is made.",
          f"- The capped model's latest AP is {cap['pooled_ap']:.6f}, but it has reached update {cap['update']:,}. At update {common:,}, its AP difference from reference was {matched['capped-official-seed2']['delta_ap_vs_reference']:+.6f}. Its current lead in wall-clock progress is not evidence of superior performance at matched exposure.",
          f"- Positive weight 10 has AP difference {weighted['delta_ap_vs_reference']:+.6f} from reference at the common checkpoint and Brier {weighted['brier']:.6f}. Its mean predicted probability is {weighted['predicted_probability_mean']:.4f} on a split with prevalence {val[:,2].mean():.4f}, and {weighted['fraction_predicted_positive_at_half']:.2%} of pairs are predicted positive at threshold 0.5. Interpret these probabilities in light of the positively weighted objective. Total training losses across different objectives should not be compared directly.",
          f"- The historical native pooled validation AP/AUROC are {baseline['native-bernett']['ap']:.6f}/{baseline['native-bernett']['auroc']:.6f}. The capped snapshot's native-validation AP difference is {cap['pooled_ap']-baseline['native-bernett']['ap']:+.6f}; prior v1 reference validation AP was {baseline['reference-seed2']['ap']:.6f}. Earlier v1 validation/test rankings reversed; none of these validation observations establishes a native test-performance improvement.",
          '- Retain the prespecified training horizon and validation selection rule. The next useful comparisons are common optimizer updates, followed by the planned development-fold checks. Do not select new checkpoints using historical test outcomes.',
          '', '**Verification**', '',
          f"Recomputed AP, AUROC and Brier for all {len(audited)} completed validation files after SHA-256 checks, exact row/label alignment, and finite-score checks. Logged metrics matched to 1e-12. Best-checkpoint pointers agree with the maximum observed pooled AP. Every logged training loss and gradient norm is finite; latest/best checkpoint payload sizes match their manifests. No GPU inference or retraining was launched for this inspection.",
          '', '![Validation progress](validation-progress.png)', '',
          '[Machine-readable snapshot](snapshot.json) · [Validation history](validation-history.csv) · [Matched-update comparison](matched-update.csv) · [SVG plot](validation-progress.svg)']
(OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n')
print(json.dumps({'report': str(OUT / 'REPORT.md'), 'snapshot_stockholm': local.isoformat(),
                  'validated_prediction_files': len(audited), 'matched': matched,
                  'progress': {n: p['current_logged_update'] for n,p in progress.items()}}, indent=2))
