"""Four tiny TRAIN-only fits, frozen calibration, and paired assessment contrasts."""
import json
import os
import time
import warnings
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.exceptions import ConvergenceWarning
from study import ROOT, config, read, atomic, now, sha, verify_freeze, load_record, fuse

CONTRASTS = {
    'pairing_gap_improvement': {'local_true': 1, 'local_shuffled': -1, 'global_true': -1, 'global_shuffled': 1},
    'local_minus_baseline': {'local_true': 1, 'baseline': -1},
    'local_minus_global_true': {'local_true': 1, 'global_true': -1},
    'local_minus_shuffled': {'local_true': 1, 'local_shuffled': -1},
    'local_minus_quality': {'local_true': 1, 'quality': -1},
    'locality_pairing_gap': {'local_true': 1, 'local_shuffled': -1, 'scrambled_true': -1, 'scrambled_shuffled': 1},
    'local_minus_spatially_scrambled_true': {'local_true': 1, 'scrambled_true': -1},
    'same_head_pairing_gap': {'local_true': 1, 'local_true_on_shuffled': -1},
    'spatially_scrambled_same_head_pairing_gap': {'scrambled_true': 1, 'scrambled_true_on_shuffled': -1},
}


def evidence(h, x):
    return ((((x - h['standard_mean']) / h['standard_scale'] - h['pca_mean'])
             @ np.asarray(h['pca_components']).T) @ np.asarray(h['coef']).T + h['intercept']).ravel()


def fit_head(x, y, train, cal, base, gate, cfg):
    sc = StandardScaler().fit(x[train]); sx = sc.transform(x)
    pc = PCA(n_components=cfg['fusion']['pca_components'], svd_solver='full').fit(sx[train]); px = pc.transform(sx)
    cl = LogisticRegression(C=cfg['fusion']['C'], max_iter=1000, solver='lbfgs', random_state=cfg['seed']).fit(px[train], y[train])
    ev = cl.decision_function(px)
    choices = [{'alpha': a, 'ap': float(average_precision_score(y[cal], fuse(base, ev, gate, a)[cal]))}
               for a in cfg['fusion']['alpha_grid']]
    best = max(c['ap'] for c in choices); alpha = min(c['alpha'] for c in choices if c['ap'] == best)
    h = {'standard_mean': sc.mean_.tolist(), 'standard_scale': sc.scale_.tolist(), 'pca_mean': pc.mean_.tolist(),
         'pca_components': pc.components_.tolist(), 'coef': cl.coef_.tolist(), 'intercept': cl.intercept_.tolist(),
         'classes': cl.classes_.tolist(), 'alpha': alpha, 'calibration_grid': choices,
         'pca_explained_variance_ratio_sum': float(pc.explained_variance_ratio_.sum())}
    np.testing.assert_allclose(evidence(h, x), ev, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(sc.mean_, x[train].mean(0), atol=1e-12, rtol=1e-12)
    return h, fuse(base, ev, gate, alpha)


def ap_setup(y, z):
    order = np.argsort(-z, kind='stable'); ends = np.r_[np.flatnonzero(np.diff(z[order])), len(z)-1]
    return order, ends, y[order]


def weighted_ap(setup, weights):
    order, ends, y = setup; w = weights[order]
    tp = np.cumsum(w * y)[ends]; total = np.cumsum(w)[ends]
    if not tp[-1] or total[-1] == tp[-1]:
        return np.nan
    p = np.divide(tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0)
    return float(np.sum(np.diff(np.r_[0., tp]) * p) / tp[-1])


def contrasts(aps):
    return {k: float(sum(c * aps[n] for n, c in terms.items())) for k, terms in CONTRASTS.items()}


def bootstrap(rows, y, scores, masks, cfg):
    ids = sorted({r[k] for r in rows for k in ['a', 'b']}); lookup = {x: i for i, x in enumerate(ids)}
    ia = np.array([lookup[r['a']] for r in rows]); ib = np.array([lookup[r['b']] for r in rows])
    masks = {k: m for k, m in masks.items() if min(np.sum(m & (y == c)) for c in [0, 1]) >= 10}
    setups = {k: {n: ap_setup(y[m], z[m]) for n, z in scores.items()} for k, m in masks.items()}
    values = {k: {c: [] for c in CONTRASTS} for k in masks}; rng = np.random.default_rng(cfg['seed'])
    length_interaction = []
    for _ in range(cfg['fusion']['bootstrap_replicates']):
        count = rng.poisson(1, len(ids)); w = count[ia] * count[ib]; current = {}
        for k, m in masks.items():
            ap = {n: weighted_ap(s, w[m]) for n, s in setups[k].items()}
            current[k] = contrasts(ap)
            for c, v in current[k].items():
                if np.isfinite(v): values[k][c].append(v)
        if 'assessment/length_le512' in current and 'assessment/length_513_1536' in current:
            v = current['assessment/length_le512']['local_minus_shuffled'] - current['assessment/length_513_1536']['local_minus_shuffled']
            if np.isfinite(v): length_interaction.append(v)
    def interval(v):
        if len(v) < .95 * cfg['fusion']['bootstrap_replicates']:
            return {'status': 'insufficient_valid_replicates', 'replicates': len(v)}
        return {'low': float(np.quantile(v, .025)), 'high': float(np.quantile(v, .975)), 'replicates': len(v)}
    return {k: {c: interval(v) for c, v in vv.items()} for k, vv in values.items()}, interval(length_interaction)


def summarize_group(mask, y, scores, available):
    result = {'rows': int(mask.sum()), 'positive': int(y[mask].sum()), 'eligible': int((mask & available).sum()),
              'eligible_positive': int(y[mask & available].sum())}
    if len(np.unique(y[mask])) < 2:
        return {**result, 'status': 'insufficient_classes'}
    yy = y[mask]; counts = np.bincount(yy, minlength=2); weights = .5 / counts[yy]
    result['metrics'] = {k: {'ap': float(average_precision_score(yy, z[mask])),
        'auroc': float(roc_auc_score(yy, z[mask])),
        'ap_at_50_percent_prevalence': float(average_precision_score(yy, z[mask], sample_weight=weights))} for k, z in scores.items()}
    result['contrasts'] = contrasts({k: v['ap'] for k, v in result['metrics'].items()})
    return result


def main():
    started = time.monotonic(); cfg = config(); fp = verify_freeze()
    sample = read(ROOT / 'data/sample.json'); previous = read(ROOT / 'data/source_metadata.json')
    with ThreadPoolExecutor(max_workers=8) as ex:
        data = list(ex.map(lambda r: load_record(r['uid'], fp), sample))
    missing = [r['uid'] for r, d in zip(sample, data) if d is None]
    if missing:
        atomic(ROOT / 'results/decision.json', {'at_utc': now(), 'status': 'inconclusive_incomplete',
            'missing_pairs': missing, 'expected_pairs': len(sample), 'heads_fitted': False, 'test_accessed': False})
        (ROOT / 'results/REPORT.md').write_text(f'# Vx v2\n\nInconclusive: {len(missing)} of 8,000 computational records are missing. No head was fitted and no incomplete subset was evaluated.\n')
        return
    if (ROOT / 'results/heads.json').exists():
        raise RuntimeError('Already fitted; no automatic repeated analysis')
    source = np.load(ROOT / 'data/source.npz', allow_pickle=False)
    assert list(source['uids']) == [r['uid'] for r in sample]
    for r, d, p in zip(sample, data, previous):
        if sorted([r['a'], r['b']]) != [d['a'], d['b']] or d['available'] != p['available'] or d['gate'] != p['gate'] or d['reason'] != p['reason']:
            raise ValueError('Identity/eligibility changed: ' + r['uid'])
    y = np.array([r['label'] for r in sample]); gate = np.array([d['gate'] for d in data]); available = gate > 0
    train = np.array([r['split'] == 'train' for r in sample]); dev = ~train
    cal = np.array([r['role'] == 'calibration' for r in sample]); ass = np.array([r['role'] == 'assessment' for r in sample]); fit = train & available
    counts = {n: [int(np.sum(m & (y == c))) for c in [0, 1]] for n, m in [('fit', fit), ('calibration', cal), ('assessment', ass)]}
    assert counts == read(ROOT / 'data/source_metrics.json')['fit_counts']
    if min(counts['fit']) < 100 or min(counts['calibration'] + counts['assessment']) < 50:
        raise ValueError('Insufficient fitting/calibration/assessment population')
    baseline = read(ROOT / 'data/baseline.json')
    base = np.array([baseline[r['uid']]['score'] if r['split'] == 'val' else 0. for r in sample])
    for r in sample:
        if r['split'] == 'val':
            b = baseline[r['uid']]
            assert b['label'] == r['label'] and sorted([b['a'], b['b']]) == sorted([r['a'], r['b']])
    old_heads = read(ROOT / 'data/source_heads.json'); scores = {'baseline': base}
    for kind, name in [('true', 'global_true'), ('shuffled', 'global_shuffled'), ('quality', 'quality')]:
        h = old_heads[kind]; scores[name] = fuse(base, evidence(h, source[kind]), gate, h['alpha'])
    old_metrics = read(ROOT / 'data/source_metrics.json')
    for pop, mask in [('assessment', ass), ('fixed_dev_sample', dev), ('calibration', cal)]:
        for name, old_name in [('baseline', 'baseline'), ('global_true', 'true'), ('global_shuffled', 'shuffled'), ('quality', 'quality')]:
            assert abs(average_precision_score(y[mask], scores[name][mask]) - old_metrics['results'][pop]['metrics'][old_name]['ap']) < 1e-12
    warnings.simplefilter('error', ConvergenceWarning)
    heads = {}; features = {}
    for arm in cfg['arms']:
        features[arm] = np.array([d[arm] for d in data], dtype=float)
        heads[arm], scores[arm] = fit_head(features[arm], y, fit, cal, base, gate, cfg)
    for prefix in ['local', 'scrambled']:
        h = heads[prefix + '_true']
        scores[prefix + '_true_on_shuffled'] = fuse(base, evidence(h, features[prefix + '_shuffled']), gate, h['alpha'])
    for z in scores.values():
        assert np.isfinite(z).all() and np.array_equal(z[dev & ~available], base[dev & ~available])
    atomic(ROOT / 'results/heads.json', heads)
    length = np.array([r['length'] for r in sample])
    masks = {'assessment': ass, 'fixed_dev_sample': dev, 'calibration': cal,
             'crossing_descriptive': np.array([r['role'] == 'crossing' for r in sample])}
    for pop, rr in [('assessment', ass), ('fixed_dev_sample', dev)]:
        masks[pop + '/eligible'] = rr & available
        for name, ll in [('le512', length <= 512), ('513_1024', (length > 512) & (length <= 1024)),
                         ('1025_1536', (length > 1024) & (length <= 1536)),
                         ('513_1536', (length > 512) & (length <= 1536)), ('over1536', length > 1536)]:
            masks[pop + '/length_' + name] = rr & ll
    diagnostic_fields = {
        'neff80_paired': lambda d: d['paired_diversity']['neff80'],
        'neff80_per_residue': lambda d: d['paired_diversity']['neff_per_valid_residue'],
        'minimum_quality': lambda d: d['minimum_quality'], 'valid_interchain_cells': lambda d: d['valid_interchain_cells'],
        'joint_support_fraction': lambda d: d['true_support']['fraction_cells_at_least_half_rows'],
        'actual_null_token_change': lambda d: d['actual_token_changed_fraction'],
        'length_asymmetry': lambda d: d['length_asymmetry'],
        'candidate_depth': lambda d: d['candidates_after_coverage_and_conflict_filters']}
    diagnostic_summary = {}; cuts = {}
    for name, get in diagnostic_fields.items():
        a = np.array([get(d['diagnostics']) if d['available'] else np.nan for d in data])
        q = np.unique(np.quantile(a[fit], [.25, .5, .75])); cuts[name] = q.tolist()
        diagnostic_summary[name] = {n: dict(zip(['min', 'q25', 'median', 'q75', 'max'], map(float, np.quantile(a[m & available], [0, .25, .5, .75, 1]))))
            for n, m in [('train', train), ('dev', dev), ('assessment', ass)]}
        bin_id = np.searchsorted(q, a, side='right')
        for j in range(len(q) + 1):
            masks[f'assessment/{name}/train_quantile_bin_{j}'] = ass & available & (bin_id == j)
    ann = read(ROOT / 'data/dev_family_witnesses.json')
    family_count = np.full(len(sample), -1)
    families = []
    for i, r in enumerate(sample):
        aa, bb = ann.get(str(r['a'])), ann.get(str(r['b']))
        if aa is not None and bb is not None:
            family_count[i] = (aa['shared_family'] == 'True') + (bb['shared_family'] == 'True')
        families.append({a['family'] for a in [aa, bb] if a is not None and a['family']})
    for j in [0, 1, 2]:
        masks[f'assessment/train_shared_family_witness_count_{j}'] = ass & available & (family_count == j)
    groups = {k: summarize_group(m, y, scores, available) for k, m in masks.items()}
    dev_rows = [r for r, m in zip(sample, dev) if m]
    intervals, length_interval = bootstrap(dev_rows, y[dev], {k: z[dev] for k, z in scores.items()},
        {k: m[dev] for k, m in masks.items() if k not in ['calibration', 'crossing_descriptive']}, cfg)
    for k, ci in intervals.items(): groups[k]['ap_intervals'] = ci
    influence = {}
    for pop, rr in [('assessment', ass), ('fixed_dev_sample', dev)]:
        influence[pop] = {}
        for kind in ['protein', 'family_witness']:
            count = Counter(v for i in np.flatnonzero(rr) for v in ([sample[i]['a'], sample[i]['b']] if kind == 'protein' else families[i]))
            values = []
            for member, _ in count.most_common(10):
                member_mask = np.array([member in ([r['a'], r['b']] if kind == 'protein' else families[i]) for i, r in enumerate(sample)])
                m = rr & ~member_mask
                result = summarize_group(m, y, scores, available)
                values.append({'group': str(member), 'removed_pairs': int((rr & member_mask).sum()),
                    'remaining_pairs': int(m.sum()), 'contrasts': result.get('contrasts')})
            influence[pop][kind] = values
    mm = groups['assessment']; ci = mm['ap_intervals']; pts = mm['contrasts']
    needs = ['pairing_gap_improvement', 'local_minus_baseline', 'local_minus_shuffled', 'local_minus_quality']
    checked = {k: ci[k].get('low', -np.inf) > 0 for k in needs}
    checked['true_model_improves_over_original'] = pts['local_minus_global_true'] > 0
    checked['practical_baseline_gain'] = pts['local_minus_baseline'] >= cfg['fusion']['continuation_delta_ap']
    checked['auroc_noninferiority'] = mm['metrics']['local_true']['auroc'] >= mm['metrics']['baseline']['auroc'] - cfg['fusion']['maximum_auroc_decline']
    changes = np.array([d['diagnostics']['actual_sequence_changed_fraction'] for d, keep in zip(data, ass) if keep and d['available']])
    checked['actual_null_adequate'] = len(changes) > 0 and float(np.mean(changes >= .5)) >= .8
    passed = all(bool(v) for v in checked.values())
    locality = passed and ci['locality_pairing_gap'].get('low', -np.inf) > 0 and ci['same_head_pairing_gap'].get('low', -np.inf) > 0
    results = {'at_utc': now(), 'fingerprint': fp, 'fit_counts': counts, 'results': groups,
        'primary_contrast': 'pairing_gap_improvement', 'contrast_definitions': CONTRASTS,
        'alphas': {k: h['alpha'] for k, h in heads.items()}, 'diagnostic_summary': diagnostic_summary,
        'diagnostic_stratum_cuts_from_eligible_train': cuts, 'group_influence': influence,
        'assessment_short_minus_long_local_pairing_gap_interval': length_interval,
        'bootstrap': '1000 joint matched protein Poisson multipliers; pair weight endpoint product; two-sided 95% percentile intervals',
        'subgroup_intervals': 'exploratory, unadjusted; skipped when either class has fewer than 10 observations',
        'family_annotation': 'DEV-only TRAIN-shared witnesses; no complete-family independence claim',
        'assessment_previously_inspected': True, 'test_accessed': False, 'structural_heads_evaluated': False,
        'original_metrics_reproduced': True, 'exact_baseline_fallback_verified': True, 'seconds': time.monotonic() - started}
    atomic(ROOT / 'results/metrics.json', results)
    decision = {'at_utc': now(), 'status': 'pairing_statistics_met_family_breadth_unresolved' if passed else 'no_go_or_inconclusive',
        'checks': {k: bool(v) for k, v in checked.items()}, 'locality_statistics_met': bool(locality),
        'family_breadth_established': False, 'primary_point': pts['pairing_gap_improvement'],
        'primary_interval': ci['pairing_gap_improvement'], 'assessment_previously_inspected': True,
        'production_authorized': False, 'test_accessed': False, 'metrics_sha256': sha(ROOT / 'results/metrics.json')}
    atomic(ROOT / 'results/decision.json', decision)
    np.savez_compressed(ROOT / 'results/dev_predictions.npz', uids=np.array([r['uid'] for r in dev_rows]), labels=y[dev], **{k: z[dev] for k, z in scores.items()})
    lines = ['# Vx v2 local-pooling ablation', '', f"Decision: **{decision['status']}**.", '',
        'Fixed 4,000 TRAIN and 4,000 DEV pairs; original eligibility and baseline fallback preserved. No TEST access. Assessment was previously inspected, so this is exploratory.', '',
        '| Assessment model | AP | AUROC |', '|---|---:|---:|']
    for name, v in mm['metrics'].items(): lines.append(f"| {name} | {v['ap']:.6f} | {v['auroc']:.6f} |")
    lines += ['', '| Assessment AP contrast | Estimate | 95% matched protein interval |', '|---|---:|---|']
    for name, point in pts.items():
        c = ci[name]; lines.append(f"| {name} | {point:+.6f} | [{c.get('low', float('nan')):+.6f}, {c.get('high', float('nan')):+.6f}] |")
    lines += ['', 'The primary contrast is (local true − local shuffled) − (original global true − original global shuffled). A larger gap produced only by weakening the shuffled model is insufficient.', '',
        f"Selected fusion coefficients: `{json.dumps(results['alphas'], sort_keys=True)}`. All head transforms were fitted on the same 1,825 eligible TRAIN examples; alpha selection used calibration only.", '',
        'See metrics.json for full fixed DEV results, common-prevalence AP, length/depth/coverage strata, diagnostic cutpoints, fixed-head counterfactuals and protein/family-witness influence.', '',
        'Complete family annotations were unavailable within TRAIN/DEV-only inputs. Witness sensitivity is not complete-family independence. Pairing sensitivity would not by itself establish direct compensatory coevolution. No automatic continuation is enabled.']
    (ROOT / 'results/REPORT.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps(decision, indent=2), flush=True)


if __name__ == '__main__':
    main()
