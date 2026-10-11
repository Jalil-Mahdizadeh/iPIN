"""Two single TRAIN fits; fixed DEV calibration and full VAL reporting."""
from collections import Counter
import time
import numpy as np
from shared import ROOT, BASE, config, read, atomic, save_npz, now, sha, verify_freeze, ARMS
from collect import collect
from heads import fit_head
from statistics_rx import summarize, intervals, decision


def masks_for(rows):
    role = np.asarray([r['role'] for r in rows]); pilot = np.asarray([r['pilot_row'] for r in rows])
    return {'assessment_new_rows': (role == 'assessment') & ~pilot, 'assessment_all': role == 'assessment',
        'calibration': role == 'calibration', 'crossing': role == 'crossing',
        'full_val': role != 'train', 'pilot_val': (role != 'train') & pilot}


def main():
    start = time.monotonic(); cfg = config(); fp = verify_freeze()
    rows = read(ROOT / 'data/pairs.json'); shards = read(ROOT / 'data/shards.json')
    features, records, workers = collect(ROOT, rows, shards, fp, cfg['chunk_rows'])
    if len(rows) != cfg['train_pairs'] + cfg['dev_pairs']:
        raise ValueError('Wrong full cohort size')
    y = np.asarray([r['label'] for r in rows]); train = np.asarray([r['split'] == 'train' for r in rows]); dev = ~train
    gate = features['gate']; available = features['available']; fit = train & available; masks = masks_for(rows)
    with np.load(ROOT / 'data/baseline.npz', allow_pickle=False) as f:
        baseline = f['baseline']
    if baseline.shape != (int(dev.sum()),) or not np.isfinite(baseline).all():
        raise ValueError('Incomplete native VAL baseline')
    base = np.zeros(len(rows)); base[dev] = baseline
    if (ROOT / 'results/fit-start.json').exists():
        raise RuntimeError('One fit already attempted; automatic refit is forbidden')
    atomic(ROOT / 'results/fit-start.json', {'at_utc': now(), 'fingerprint': fp, 'all_rows_verified': len(rows),
        'train_fit_rows': int(fit.sum()), 'train_fit_classes': [int(np.sum(fit & (y == c))) for c in [0, 1]],
        'arms': list(ARMS), 'test_accessed': False, 'native_train_logits_used': False})
    scores = {'baseline': base}; heads = {}; evidence = {}
    for arm in ARMS:
        heads[arm], evidence[arm], scores[arm] = fit_head(features[arm], y, fit, base, gate, masks['calibration'], cfg)
        np.testing.assert_array_equal(scores[arm][dev & ~available], base[dev & ~available])
    atomic(ROOT / 'results/heads.json', heads)
    save_npz(ROOT / 'results/val_predictions.npz', uid=np.asarray([r['uid'] for r in rows])[dev], y=y[dev], gate=gate[dev],
        **{k: v[dev] for k, v in scores.items()})
    save_npz(ROOT / 'results/val_evidence.npz', **{k: v[dev] for k, v in evidence.items()})
    results = {}
    for name, mask in masks.items():
        ss = {k: v[mask] for k, v in scores.items()}; result = summarize(y[mask], ss, available[mask])
        if name in ['assessment_new_rows', 'assessment_all', 'full_val']:
            subset = [r for r, keep in zip(rows, mask) if keep]
            result['intervals'] = intervals(subset, y[mask], ss, cfg['fusion']['bootstrap_replicates'], cfg['seed'])
        results[name] = result
    strata = {}; primary = masks['assessment_new_rows']; lengths = np.asarray([r['length'] for r in rows])
    depths = np.asarray([r['depth'] for r in records])
    for lo, hi in [(0, 512), (512, 1024), (1024, 1536), (1536, 2048), (2048, 10**9)]:
        mask = primary & (lengths > lo) & (lengths <= hi)
        strata[f'length_{lo+1}_{hi}'] = summarize(y[mask], {k: v[mask] for k, v in scores.items()}, available[mask])
    for lo, hi in [(0, 0), (1, 32), (33, 64), (65, 127), (128, 128)]:
        mask = primary & (depths >= lo) & (depths <= hi)
        strata[f'depth_{lo}_{hi}'] = summarize(y[mask], {k: v[mask] for k, v in scores.items()}, available[mask])
    for name, scope in [('eligible', available), ('fallback', ~available)]:
        mask = primary & scope
        strata[name] = summarize(y[mask], {k: v[mask] for k, v in scores.items()}, available[mask])
    coverage_min = np.asarray([min(r['msa']['quality_a'], r['msa']['quality_b']) if r.get('msa') else 0. for r in records])
    for lo, hi in [(.5, .75), (.75, .9), (.9, 1.01)]:
        mask = primary & available & (coverage_min >= lo) & (coverage_min < hi)
        strata[f'quality_mask_fraction_{lo}_{hi}'] = summarize(y[mask], {k: v[mask] for k, v in scores.items()}, available[mask])
    pilot_proteins = {r[k] for r in rows if r['pilot_row'] and r['split'] == 'val' for k in ['a', 'b']}
    overlap = sum(r['a'] in pilot_proteins or r['b'] in pilot_proteins for r, keep in zip(rows, primary) if keep)
    coverage = {split: {'rows': int(mask.sum()), 'eligible': int((mask & available).sum()),
        'reasons': dict(Counter(r['reason'] for r, keep in zip(records, mask) if keep))}
        for split, mask in [('train', train), ('val', dev)]}
    result = {'at_utc': now(), 'fingerprint': fp, 'primary': cfg['primary'], 'results': results, 'strata': strata,
        'coverage': coverage, 'fit_rows': int(fit.sum()), 'alphas': {k: h['alpha'] for k, h in heads.items()},
        'new_assessment_rows_sharing_pilot_val_protein': overlap, 'bootstrap': 'Matched protein Poisson multipliers; fixed fitted predictions; endpoint-product weights',
        'assessment_exploratory': True, 'test_accessed': False, 'shuffled_arm': False,
        'feature_source_counts': dict(Counter(r['kind'] for r in records)), 'seconds': time.monotonic() - start}
    atomic(ROOT / 'results/metrics.json', result)
    answer = {**decision(results[cfg['primary']]), 'at_utc': now(), 'fingerprint': fp,
        'metrics_sha256': sha(ROOT / 'results/metrics.json'), 'automatic_test_continuation': False}
    atomic(ROOT / 'results/decision.json', answer)
    lines = ['# Vx-128 full TRAIN/VAL results', '', f"Decision: **{answer['status']}**.", '',
        'Original depth-128 matched MSA, frozen encoders, two TRAIN-only heads. No shuffled encoding or TEST evaluation.', '',
        '| VAL population | Rows | Eligible | Native AP | True128 AP | Quality AP |', '|---|---:|---:|---:|---:|---:|']
    for name, r in results.items():
        m = r['metrics']; lines.append(f"| {name} | {r['rows']} | {r['eligible']} | {m['baseline']['ap']:.6f} | {m['true']['ap']:.6f} | {m['quality']['ap']:.6f} |")
    p = results[cfg['primary']]; lines += ['', 'Primary: previously unevaluated rows within the fixed assessment components. These can share proteins/families with the pilot and are not a newly blind dataset.', '',
        '| Primary AP contrast | Estimate | 95% protein interval |', '|---|---:|---|']
    for k, c in p['intervals'].items():
        lines.append(f"| {k} | {p['contrasts'][k]:+.6f} | [{c['low']:+.6f}, {c['high']:+.6f}] |")
    lines += ['', f"Eligible TRAIN fit rows: {int(fit.sum())}. Fusion alphas: {result['alphas']}.", '',
        'Full VAL and pilot-row results are descriptive. Calibration selects alpha; crossing pairs do not. All unavailable pairs retain exact native scores. Complete numerical and source verification is required before completion.', '',
        'The predictive criterion is >=0.010 primary AP gain over native with a positive interval, AUROC loss <=0.005 and positive separation from quality/profile. No pairing-specific biological conclusion follows. TEST is not authorized by this run.']
    (ROOT / 'results/REPORT.md').write_text('\n'.join(lines) + '\n')
    print(answer, flush=True)


if __name__ == '__main__':
    main()
