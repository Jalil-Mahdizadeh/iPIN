"""Audit completed initial v2 runs using stored validation predictions only.

Run inside the pinned SIF. No model loading, inference, training, or test access.
Writes a new dated directory; production code and checkpoints remain read-only.
"""
import csv
import datetime as dt
import hashlib
import json
import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
START = dt.datetime.now(dt.timezone.utc)
OUT = ROOT / 'analysis' / START.strftime('completed-%Y%m%dT%H%M%SZ')
OUT.mkdir()
NAMES = {'reference-official-seed2': 'Reference', 'capped-official-seed2': 'Length capped',
         'positive10-official-seed2': 'Positive weight 10', 'clean-bce-official-seed2': 'Clean BCE'}

def read(path):
    return json.loads(path.read_text())

def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            digest.update(block)
    return digest.hexdigest()

def save(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')

def metrics(y, score):
    return {'ap': float(average_precision_score(y, score)),
            'auroc': float(roc_auc_score(y, score)),
            'brier': float(np.mean((expit(score) - y)**2))}

def write_csv(name, records):
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

manifest = read(ROOT / 'data/prepared/manifest.json')
assert not manifest['test_imported']
for relative in ['official/train.npy', 'official/val.npy', 'offsets.npy']:
    assert sha(ROOT / 'data/prepared' / relative) == manifest['files'][relative]
val = np.load(ROOT / 'data/prepared/official/val.npy')
y = val[:, 2]
assert len(val) == 59258 and y.sum() == 29628
lengths = np.diff(np.load(ROOT / 'data/prepared/offsets.npy'))
pair_lengths = lengths[val[:, 0]] + lengths[val[:, 1]]
release = ROOT / 'releases' / (ROOT / 'releases/CURRENT').read_text().strip()
campaign = read(ROOT / 'configs/campaign.json')['groups']['initial-screen']
assert campaign == list(NAMES)
baseline_path = ROOT.parent / 'benchmark-v1/results/validation-metrics.csv'
with baseline_path.open() as handle:
    baselines = {row['model']: {k: float(row[k]) for k in ['ap', 'auroc', 'brier']}
                 for row in csv.DictReader(handle) if row['view'] == 'pooled'}
accounting_path = ROOT / 'provenance/final-accounting-3130215.txt'
with accounting_path.open() as handle:
    accounting = {row['JobID']: row for row in csv.DictReader(handle, delimiter='|')}
assert len(accounting) == 4

runs, history, best_scores, file_audit, hash_tasks, summaries = {}, [], {}, [], [], []
batch_digests = {}
for index, name in enumerate(campaign):
    run = ROOT / 'runs' / name
    contract, best, latest, completed, status = [read(run / f'{x}.json')
        for x in ['contract', 'best', 'latest', 'completed', 'status']]
    config = contract['configuration']
    state = completed['state']
    assert config == read(release / 'configs' / f'{name}.json')
    for code, digest in contract['code'].items():
        assert sha(release / 'code' / code) == digest
    assert contract['data_manifest_sha256'] == sha(ROOT / 'data/prepared/manifest.json')
    assert contract['world_size'] == config['world_size'] == 4
    assert not contract['qualification'] and not completed['qualification']
    assert completed['fingerprint'] == contract['fingerprint']
    assert completed['test_evaluated'] is False
    assert state['update'] == state['last_validation_update'] == config['total_updates'] == 12745
    assert state['examples_seen'] == state['update'] * config['global_pairs_per_update'] == 815680
    assert not state['pending_validation']
    assert status['phase'] == 'complete' and status['state'] == state
    assert len(completed['model_hashes_by_rank']) == 4 and len(set(completed['model_hashes_by_rank'])) == 1
    job = accounting[f'3130215_{index}']
    assert job['State'] == 'COMPLETED' and job['ExitCode'] == '0:0'
    assert 'gres/gpu=4' in job['AllocTRES']
    raw = (run / 'events.jsonl').read_bytes()
    assert raw.endswith(b'\n')
    (OUT / f'{name}-events.jsonl').write_bytes(raw)
    events = [json.loads(line) for line in raw.splitlines()]
    initialized = next(e for e in events if e['event'] == 'initialized')
    updates = [e for e in events if e['event'] == 'update']
    validations = [e for e in events if e['event'] == 'validation']
    assert initialized['val_rows'] == len(val)
    expected_train = 130461 if config['train_cap_residues'] == 2193 else 163085
    assert initialized['train_rows'] == expected_train
    assert all(math.isfinite(e[k]) for e in updates for k in ['mean_loss', 'mean_bce', 'mean_mlm', 'grad_norm'])
    assert all(e['global_pairs'] == 64 for e in updates)
    assert [e['update'] for e in validations] == list(range(1000, 12001, 1000)) + [12745]
    batch_digests[name] = {e['update']: e['batch_digest'] for e in updates}
    records = []
    for event in validations:
        update = event['update']
        path = run / 'validation' / f'update-{update:09d}.npz'
        meta = read(path.with_suffix('.json'))
        assert meta['fingerprint'] == contract['fingerprint'] and sha(path) == meta['sha256']
        with np.load(path) as payload:
            a = payload['predictions']
        assert a.shape == (len(val), 4) and np.isfinite(a).all()
        assert np.array_equal(a[:, 0], np.arange(len(val))) and np.array_equal(a[:, 1], y)
        score = a[:, 2:4].mean(axis=1)
        measured = metrics(y, score)
        for key, value in measured.items():
            logged_key = 'pooled_ap' if key == 'ap' else key
            assert abs(value - meta['metrics'][logged_key]) < 1e-12
            assert abs(value - event['metrics'][logged_key]) < 1e-12
        record = {'model': name, 'update': update, 'validation_utc': event['time_utc'], **measured}
        records.append(record)
        history.append(record)
        file_audit.append({'path': str(path), 'sha256': meta['sha256'], 'rows': len(val)})
        if update == best['update']:
            best_scores[name] = score
    selected = max(records, key=lambda r: r['ap'])
    final = records[-1]
    assert best['update'] == state['best_update'] == selected['update']
    assert best['best_ap'] == state['best_ap'] == selected['ap']
    assert latest['update'] == state['update']
    for label, pointer in [('best', best), ('final', latest)]:
        assert pointer['fingerprint'] == contract['fingerprint']
        path = run / 'checkpoints' / pointer['file']
        assert path.stat().st_size == pointer['bytes']
        hash_tasks.append((name, label, path, pointer))
    log_path = ROOT / 'logs' / f'production-3130215_{index}.log'
    log_text = log_path.read_text()
    fatal = [line for line in log_text.splitlines() if re.search(
        r'Traceback \(most recent call last\)|CUDA out of memory|NCCL.*(?:ERROR|Error)|Segmentation fault|slurmstepd: error:', line)]
    assert not fatal, fatal[:3]
    probability = expit(best_scores[name])
    strata = {}
    for label, mask in [('up_to_2193_residues', pair_lengths <= 2193), ('over_2193_residues', pair_lengths > 2193)]:
        strata[label] = {'pairs': int(mask.sum()), 'prevalence': float(y[mask].mean()),
                         **metrics(y[mask], best_scores[name][mask])}
    summary = {'model': name, 'job_id': job['JobID'], 'state': job['State'],
        'train_pairs': expected_train, 'validation_pairs': len(val), 'final_update': state['update'],
        'examples_seen': state['examples_seen'], 'effective_epochs': state['examples_seen'] / expected_train,
        'best_update': selected['update'], 'best_ap': selected['ap'], 'best_auroc': selected['auroc'],
        'best_brier': selected['brier'], 'final_ap': final['ap'], 'final_auroc': final['auroc'],
        'final_brier': final['brier'], 'ap_change_best_to_final': final['ap'] - selected['ap'],
        'elapsed': job['Elapsed'], 'elapsed_seconds': int(job['ElapsedRaw']),
        'gpu_hours': int(job['ElapsedRaw']) * 4 / 3600}
    summaries.append(summary)
    runs[name] = {'summary': summary, 'contract': contract, 'completion': completed, 'job_accounting': job,
        'best_checkpoint': best, 'final_checkpoint': latest, 'best_validation': selected, 'final_validation': final,
        'selected_length_strata': strata, 'selected_probability_mean': float(probability.mean()),
        'selected_positive_fraction_at_half': float((probability >= .5).mean()),
        'all_logged_losses_gradients_finite': True, 'fatal_log_matches': fatal,
        'peak_logged_rank0_allocated_gib': max(e['gpu_peak_allocated_gib'] for e in updates),
        'event_counts': {kind: sum(e['event'] == kind for e in events) for kind in sorted({e['event'] for e in events})}}
    print(json.dumps({'audited_validations': name, 'count': len(records), 'best': selected}), flush=True)

for name in ['positive10-official-seed2', 'clean-bce-official-seed2']:
    assert batch_digests[name] == batch_digests['reference-official-seed2']

def hash_checkpoint(task):
    name, label, path, pointer = task
    measured = sha(path)
    assert measured == pointer['sha256'], str(path)
    result = {'model': name, 'selection': label, 'path': str(path), 'bytes': pointer['bytes'], 'sha256': measured}
    print(json.dumps({'verified_checkpoint': str(path)}), flush=True)
    return result

with ThreadPoolExecutor(max_workers=2) as pool:
    checkpoint_audit = list(pool.map(hash_checkpoint, hash_tasks))

# Paired Poisson protein bootstrap. A self-pair contains one distinct protein,
# so its weight is that protein's multiplicity, not its multiplicity squared.
# Sort fixed predictions once; retain all equal-score observations as one group.
proteins, endpoints = np.unique(val[:, :2], return_inverse=True)
endpoints = endpoints.reshape(-1, 2)
self_pair = endpoints[:, 0] == endpoints[:, 1]
orders, ends = {}, {}
for name, score in best_scores.items():
    order = np.argsort(-score, kind='stable')
    orders[name] = order
    ends[name] = np.r_[np.flatnonzero(np.diff(score[order]) != 0), len(score) - 1]

def weighted_ap(name, weights):
    order, end = orders[name], ends[name]
    positive = np.cumsum(weights[order] * y[order])[end]
    count = np.cumsum(weights[order])[end]
    precision = np.divide(positive, count, out=np.zeros_like(positive, dtype=float), where=count > 0)
    return float(np.dot(np.diff(np.r_[0, positive]), precision) / positive[-1])

rng = np.random.default_rng(20260930)
boot = []
for replicate in range(1000):
    multiplicity = rng.poisson(1., len(proteins))
    weights = multiplicity[endpoints[:, 0]] * multiplicity[endpoints[:, 1]]
    weights[self_pair] = multiplicity[endpoints[self_pair, 0]]
    assert np.unique(y[weights > 0]).size == 2
    row = [weighted_ap(name, weights) for name in campaign]
    if replicate < 20:
        for name, value in zip(campaign, row):
            assert abs(value - average_precision_score(y, best_scores[name], sample_weight=weights)) < 1e-12
    boot.append(row)
boot = np.array(boot)
np.savez_compressed(OUT / 'protein-bootstrap.npz', ap=boot, models=np.array(campaign))
reference = summaries[0]
comparisons = []
for index, summary in enumerate(summaries):
    name = summary['model']
    differences = boot[:, index] - boot[:, 0]
    comparison = {'model': name, 'best_update': summary['best_update'],
        'delta_ap_vs_reference': summary['best_ap'] - reference['best_ap'],
        'delta_auroc_vs_reference': summary['best_auroc'] - reference['best_auroc'],
        'delta_brier_vs_reference': summary['best_brier'] - reference['best_brier'],
        'protein_bootstrap_ap_delta_ci_low': float(np.quantile(differences, .025)),
        'protein_bootstrap_ap_delta_ci_high': float(np.quantile(differences, .975)),
        'delta_ap_vs_historical_native_validation': summary['best_ap'] - baselines['native-bernett']['ap']}
    comparisons.append(comparison)

write_csv('selected-and-final.csv', summaries)
write_csv('validation-history.csv', history)
write_csv('selected-comparisons.csv', comparisons)
strata_rows = [{'model': name, 'stratum': label, **m} for name in campaign
               for label, m in runs[name]['selected_length_strata'].items()]
write_csv('selected-length-strata.csv', strata_rows)
matched_rows = []
for update in list(range(1000, 12001, 1000)) + [12745]:
    records = {r['model']: r for r in history if r['update'] == update}
    ref = records['reference-official-seed2']
    for name in campaign:
        r = records[name]
        matched_rows.append({**r, **{f'delta_{k}_vs_reference': r[k]-ref[k] for k in ['ap', 'auroc', 'brier']}})
write_csv('matched-update-comparisons.csv', matched_rows)

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.7))
colors = ['#2563eb', '#d97706', '#059669', '#9333ea']
for ax, metric, title in zip(axes, ['ap', 'auroc', 'brier'],
        ['Validation AP (higher is better)', 'Validation AUROC (higher is better)', 'Brier (lower is better)']):
    for name, color in zip(campaign, colors):
        rows = [r for r in history if r['model'] == name]
        ax.plot([r['update'] for r in rows], [r[metric] for r in rows], marker='o', lw=1.7, ms=3.5,
                color=color, label=NAMES[name])
        selected = runs[name]['best_validation']
        ax.scatter([selected['update']], [selected[metric]], marker='*', s=120, color=color,
                   edgecolors='black', linewidth=.45, zorder=4)
    ax.axhline(baselines['native-bernett'][metric], color='#333333', ls='--', lw=1.1,
               label='Released native (historical validation)')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('Optimizer updates')
    ax.grid(alpha=.18)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .015), ncol=3, fontsize=9, frameon=False)
fig.suptitle('Completed v2 runs: 59,258 validation pairs; stars mark AP-selected checkpoints', fontsize=12)
fig.tight_layout(rect=(0, .15, 1, .93))
for suffix in ['png', 'svg']:
    fig.savefig(OUT / f'validation-curves.{suffix}', dpi=180)
plt.close(fig)

snapshot = {'audit_started_utc': START.isoformat(), 'audit_finished_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
    'audit_script_sha256': sha(Path(__file__)), 'runs': runs, 'comparisons': comparisons,
    'validation_predictions_verified': file_audit, 'checkpoint_payloads_verified': checkpoint_audit,
    'historical_validation_baselines': baselines, 'baseline_csv_sha256': sha(baseline_path),
    'scheduler_accounting_sha256': sha(accounting_path), 'scheduler_times_timezone': 'Europe/Stockholm',
    'bootstrap': {'replicates': 1000, 'seed': 20260930, 'proteins': int(len(proteins)),
                  'self_pairs': int(self_pair.sum()), 'self_pair_weight': 'one multiplicity per distinct protein',
                  'algorithm': 'paired Poisson(1) protein resampling with endpoint-product pair weights',
                  'interval': '2.5 and 97.5 percentiles of candidate-minus-reference AP',
                  'limitations': 'Conditions on this validation graph and fixed selected predictions; excludes training-seed and checkpoint/model-selection uncertainty and does not establish independent generalization.',
                  'sklearn_cross_checks': 80},
    'total_allocated_gpu_hours': sum(r['gpu_hours'] for r in summaries),
    'inference': 'stored BF16 clean-sequence predictions, mean AB/BA logits',
    'same_logged_batches_reference_positive10_clean': True,
    'test_predictions_read': False, 'gpu_inference_launched': False,
    'training_or_selection_modified': False}
save(OUT / 'snapshot.json', snapshot)
print(json.dumps({'output': str(OUT), 'selected_and_final': summaries, 'comparisons': comparisons,
                  'validated_prediction_files': len(file_audit), 'checkpoint_hashes_verified': len(checkpoint_audit),
                  'total_gpu_hours': snapshot['total_allocated_gpu_hours']}, indent=2), flush=True)
