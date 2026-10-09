"""Five predeclared TRAIN-only heads; calibration-only fusion and fixed DEV analysis."""
import time
from collections import Counter
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from study import ROOT, config, read, sha, atomic, atomic_npz, now, verify_freeze, load_feature, fuse, check_budget
from heads import fit_transform, transform, pair_features, fit_head, evidence, legacy_evidence
from monomers import pair_gate
from statistics_vy import CONTRASTS, contrasts, bootstrap, summarize


def load_complete(fp):
    sample = read(ROOT / 'data/sample.json'); monomers = read(ROOT / 'data/monomers.json')
    pairs = read(ROOT / 'data/pairs.json'); ids = read(ROOT / 'data/required-proteins.json')
    if len(pairs) != len(sample):
        raise ValueError('Wrong pair count')
    available = np.array([p['available'] for p in pairs]); gate = np.array([p['gate'] for p in pairs])
    expected_ids = sorted({str(r[k]) for r, p in zip(sample, pairs) if p['available'] for k in ('a', 'b')}, key=int)
    if ids != expected_ids:
        raise ValueError('Required encoding population changed')
    for r, p in zip(sample, pairs):
        if (r['uid'], r['a'], r['b']) != (p['uid'], p['a'], p['b']):
            raise ValueError('Pair identity mismatch')
        if p['gate'] != pair_gate(monomers[str(r['a'])], monomers[str(r['b'])]) or p['available'] != (p['gate'] > 0):
            raise ValueError('Pair availability/reliability mismatch')
    vectors = []
    for pid in ids:
        x = load_feature(pid, fp, monomers[pid])
        if x is None:
            raise RuntimeError('Incomplete computational coverage: required monomer ' + pid)
        vectors.append(x)
    raw = {k: np.array([x[k] for x in vectors], dtype=float) for k in ['M', 'S']}
    raw['P'] = np.array([monomers[p]['profile'] for p in ids], dtype=float)
    if any(not np.isfinite(a).all() for a in raw.values()):
        raise ValueError('Nonfinite raw descriptor')
    return sample, monomers, pairs, ids, available, gate, raw


def baseline_and_references(sample):
    baseline = read(ROOT / 'data/baseline.json'); dev = np.array([r['split'] == 'val' for r in sample])
    base = np.zeros(len(sample), dtype=float)
    for i in np.flatnonzero(dev):
        r = sample[i]; b = baseline[r['uid']]
        if b['label'] != r['label'] or sorted([b['a'], b['b']]) != sorted([r['a'], r['b']]):
            raise ValueError('Native baseline identity mismatch')
        base[i] = b['score']
    with np.load(ROOT / 'data/source.npz', allow_pickle=False) as f:
        if f['uids'].tolist() != [r['uid'] for r in sample]:
            raise ValueError('Historical feature row mismatch')
        old = {k: f[k].copy() for k in ['true', 'shuffled', 'quality', 'gate']}
    old_heads = read(ROOT / 'data/source_heads.json'); scores = {'baseline': base}
    for arm in ['true', 'shuffled', 'quality']:
        h = old_heads[arm]; scores['vx_' + arm] = fuse(base, legacy_evidence(h, old[arm]), old['gate'], h['alpha'])
    previous = read(ROOT / 'data/source_metrics.json')
    y = np.array([r['label'] for r in sample])
    for pop, mask in [('assessment', np.array([r['role'] == 'assessment' for r in sample])),
                      ('calibration', np.array([r['role'] == 'calibration' for r in sample])), ('fixed_dev_sample', dev)]:
        for name, source_name in [('baseline', 'baseline'), ('vx_true', 'true'), ('vx_shuffled', 'shuffled'), ('vx_quality', 'quality')]:
            ref = previous['results'][pop]['metrics'][source_name]
            if abs(average_precision_score(y[mask], scores[name][mask]) - ref['ap']) > 1e-12:
                raise ValueError('Historical AP failed to reproduce')
            if 'auroc' in ref and abs(roc_auc_score(y[mask], scores[name][mask]) - ref['auroc']) > 1e-12:
                raise ValueError('Historical AUROC failed to reproduce')
    return base, scores, old['gate'] > 0


def make_strata(sample, monomers, pairs, available, fit, cfg):
    dev = np.array([r['split'] == 'val' for r in sample]); ass = np.array([r['role'] == 'assessment' for r in sample])
    cal = np.array([r['role'] == 'calibration' for r in sample]); length = np.array([r['length'] for r in sample])
    old = np.array([p['vx_available'] for p in pairs])
    masks = {'assessment': ass, 'fixed_dev_sample': dev, 'calibration': cal,
             'crossing_descriptive': np.array([r['role'] == 'crossing' for r in sample])}
    for name, mask in [('assessment', ass), ('fixed_dev_sample', dev)]:
        masks[name + '/eligible'] = mask & available
        for title, band in [('le512', length <= 512), ('513_1024', (length > 512) & (length <= 1024)),
                            ('1025_1536', (length > 1024) & (length <= 1536)), ('over1536', length > 1536)]:
            masks[name + '/length_' + title] = mask & band
        for title, group in [('both', old & available), ('vy_only', ~old & available), ('vx_only', old & ~available), ('neither', ~old & ~available)]:
            masks[name + '/availability_' + title] = mask & group
    getters = {
        'minimum_neff80': lambda a, b: min(a['neff80'], b['neff80']),
        'minimum_neff80_per_valid_residue': lambda a, b: min(a['neff80'] / a['valid_residues'], b['neff80'] / b['valid_residues']),
        'minimum_quality': lambda a, b: min(a['quality_fraction'], b['quality_fraction']),
        'minimum_occupancy': lambda a, b: min(a['occupancy_mean'], b['occupancy_mean']),
        'minimum_effective_coverage': lambda a, b: min(a['fraction_good_positions_half_depth'], b['fraction_good_positions_half_depth']),
        'minimum_comparable_fraction': lambda a, b: min(a['fraction_comparable'], b['fraction_comparable']),
        'minimum_retained_depth': lambda a, b: min(a['retained_depth'], b['retained_depth']),
        'minimum_filtered_depth': lambda a, b: min(a['filtered_depth'], b['filtered_depth']),
        'minimum_taxonomic_effective_number': lambda a, b: min(a['taxonomic_effective_number'], b['taxonomic_effective_number']),
        'maximum_ambiguous_fraction': lambda a, b: max(a['ambiguous_fraction'], b['ambiguous_fraction']),
        'length_asymmetry': lambda a, b: max(a['length'], b['length']) / min(a['length'], b['length'])}
    cuts = {}; diagnostic_summary = {}; diagnostics = {}
    for name, fn in getters.items():
        values = np.array([fn(monomers[str(r['a'])], monomers[str(r['b'])]) if av else np.nan for r, av in zip(sample, available)])
        q = np.unique(np.quantile(values[fit], [.25, .5, .75])); cuts[name] = q.tolist()
        diagnostic_summary[name] = {pop: dict(zip(['min', 'q25', 'median', 'q75', 'max'], map(float, np.quantile(values[m & available], [0, .25, .5, .75, 1]))))
                                   for pop, m in [('train', np.array([r['split'] == 'train' for r in sample])), ('dev', dev), ('assessment', ass)]}
        bins = np.searchsorted(q, values, side='right'); diagnostics[name] = values
        for i in range(len(q) + 1):
            masks[f'assessment/{name}/train_bin_{i}'] = ass & available & (bins == i)
    ann = read(ROOT / 'data/dev_family_witnesses.json'); families = []
    count = np.full(len(sample), -1); adequate = np.full(len(sample), -1)
    for i, r in enumerate(sample):
        a, b = ann.get(str(r['a'])), ann.get(str(r['b']))
        if a is not None and b is not None:
            count[i] = int(a['shared_family'] == 'True') + int(b['shared_family'] == 'True')
            adequate[i] = int(a['adequate'] == 'True') + int(b['adequate'] == 'True')
        families.append({x['family'] for x in [a, b] if x is not None and x['family']})
    for i in [-1, 0, 1, 2]:
        masks[f'assessment/train_shared_family_witness_count_{i}'] = ass & (count == i)
        masks[f'assessment/adequate_annotation_count_{i}'] = ass & (adequate == i)
    return masks, cuts, diagnostic_summary, diagnostics, families


def main():
    started = time.monotonic(); cfg = config(); fp = verify_freeze(); check_budget()
    if (ROOT / 'results/heads.json').exists() or (ROOT / 'results/fit-start.json').exists():
        raise RuntimeError('No repeated outcome fitting allowed')
    sample, monomers, pairs, ids, available, gate, raw = load_complete(fp)
    y = np.array([r['label'] for r in sample]); train = np.array([r['split'] == 'train' for r in sample]); dev = ~train
    calibration = np.array([r['role'] == 'calibration' for r in sample]); assessment = np.array([r['role'] == 'assessment' for r in sample])
    fit = train & available
    counts = {name: [int(np.sum(mask & (y == c))) for c in (0, 1)] for name, mask in [('fit', fit), ('calibration', calibration), ('assessment', assessment)]}
    if min(counts['fit']) < cfg['fusion']['minimum_fit_per_class'] or min(counts['calibration'] + counts['assessment']) < cfg['fusion']['minimum_calibration_assessment_per_class']:
        raise ValueError('Insufficient population; no outcome model fitted')
    base, scores, old_available = baseline_and_references(sample)
    if not np.array_equal(old_available, [p['vx_available'] for p in pairs]):
        raise ValueError('Historical eligibility mismatch')
    fit_proteins = {str(r[k]) for r, keep in zip(sample, fit) if keep for k in ('a', 'b')}
    fit_indices = np.array([i for i, p in enumerate(ids) if p in fit_proteins])
    if any(monomers[p]['split'] != 'train' for p in fit_proteins):
        raise ValueError('Non-TRAIN protein in representation fit')
    atomic(ROOT / 'results/fit-start.json', {'at_utc': now(), 'fingerprint': fp, 'counts': counts,
        'complete_required_monomers': len(ids), 'distinct_fit_proteins': len(fit_indices)})
    transforms = {}; pair = {}; heads = {}; ev = {}
    for arm in ['M', 'S', 'P']:
        transforms[arm], z = fit_transform(raw[arm], fit_indices, ids, cfg['fusion']['protein_pca_components'])
        pair[arm] = pair_features(z, sample, ids, available)
    pair['SP'] = np.c_[pair['S'], pair['P']]
    pair['U'] = pair['M'][:, :cfg['fusion']['protein_pca_components']]
    for arm in cfg['arms']:
        heads[arm], ev[arm], scores[arm] = fit_head(pair[arm], y, fit, calibration, base, gate, cfg['fusion'])
        print({'fitted': arm, 'alpha': heads[arm]['alpha'], 'dimensions': pair[arm].shape[1]}, flush=True)
    counterfactual = pair_features(transform(transforms['M'], raw['S']), sample, ids, available)
    ev['M_on_S'] = evidence(heads['M'], counterfactual)
    scores['M_on_S'] = fuse(base, ev['M_on_S'], gate, heads['M']['alpha'])
    for arm in [*cfg['arms'], 'M_on_S']:
        if not np.array_equal(scores[arm][~available], base[~available]):
            raise ValueError('Exact native fallback violated')
    atomic(ROOT / 'results/heads.json', {'fingerprint': fp, 'transforms': transforms, 'heads': heads,
        'fit_pair_uids': [r['uid'] for r, keep in zip(sample, fit) if keep], 'TRAIN_baseline_used': False})
    atomic_npz(ROOT / 'results/dev_predictions.npz', uids=np.array([r['uid'] for r, keep in zip(sample, dev) if keep]),
               labels=y[dev], gate=gate[dev], **{name: z[dev] for name, z in scores.items()})
    atomic_npz(ROOT / 'results/dev_evidence.npz', **{name: z[dev] for name, z in ev.items()})
    masks, cuts, diag_summary, diagnostics, families = make_strata(sample, monomers, pairs, available, fit, cfg)
    groups = {name: summarize(mask, y, scores, available) for name, mask in masks.items()}
    dev_rows = [r for r, keep in zip(sample, dev) if keep]
    intervals = bootstrap(dev_rows, y[dev], {name: z[dev] for name, z in scores.items()},
        {name: mask[dev] for name, mask in masks.items() if name not in ['calibration', 'crossing_descriptive']}, cfg['fusion'], cfg['seed'])
    for name, ci in intervals.items():
        groups[name]['ap_intervals'] = ci
    influence = {}
    for pop in ['assessment', 'fixed_dev_sample']:
        scope = masks[pop]; influence[pop] = {}
        for kind in ['protein', 'family_witness']:
            members = [[r['a'], r['b']] if kind == 'protein' else families[i] for i, r in enumerate(sample)]
            degree = Counter(member for i in np.flatnonzero(scope) for member in set(members[i]))
            entries = []
            for member, _ in sorted(degree.items(), key=lambda item: (-item[1], str(item[0])))[:10]:
                included = np.array([member in m for m in members]); keep = scope & ~included
                result = summarize(keep, y, scores, available)
                entries.append({'group': str(member), 'removed_pairs': int(np.sum(scope & included)),
                                'remaining_pairs': int(keep.sum()), 'contrasts': result.get('contrasts')})
            influence[pop][kind] = entries
    common_alpha = {}
    for pop in ['assessment', 'fixed_dev_sample']:
        mask = masks[pop]; entries = []
        for alpha in cfg['fusion']['alpha_grid']:
            values = {arm: float(average_precision_score(y[mask], fuse(base, ev[arm], gate, alpha)[mask])) for arm in ['M', 'S', 'SP']}
            entries.append({'alpha': alpha, 'ap': values, 'M_minus_S': values['M'] - values['S'], 'M_minus_SP': values['M'] - values['SP']})
        common_alpha[pop] = entries
    calibration_diagnostics = {}; evidence_alone = {}
    for pop in ['assessment', 'fixed_dev_sample']:
        mask = masks[pop] & available
        evidence_alone[pop] = {arm: {'rows': int(mask.sum()), 'ap': float(average_precision_score(y[mask], ev[arm][mask])),
            'auroc': float(roc_auc_score(y[mask], ev[arm][mask]))} for arm in cfg['arms']}
        calibration_diagnostics[pop] = {}
        for arm in cfg['arms']:
            correction = scores[arm] - base
            calibration_diagnostics[pop][arm] = {'correction_quantiles_eligible': np.quantile(correction[mask], [0, .25, .5, .75, 1]).tolist(),
                'mean_absolute_correction_eligible': float(np.abs(correction[mask]).mean()),
                'brier': float(np.mean((1 / (1 + np.exp(-np.clip(scores[arm][masks[pop]], -700, 700))) - y[masks[pop]]) ** 2))}
    results = {'at_utc': now(), 'fingerprint': fp, 'fit_counts': counts, 'fit_distinct_proteins': len(fit_indices),
        'complete_required_monomers': len(ids), 'results': groups, 'primary_contrast': 'M_minus_SP', 'contrast_definitions': CONTRASTS,
        'alphas': {arm: h['alpha'] for arm, h in heads.items()}, 'common_alpha_descriptive': common_alpha,
        'evidence_alone_eligible_only': evidence_alone, 'correction_and_calibration': calibration_diagnostics,
        'diagnostic_summary': diag_summary, 'diagnostic_cuts_from_eligible_train': cuts, 'group_influence': influence,
        'bootstrap': '1000 joint protein Poisson multipliers; endpoint product weights; 95% percentile intervals conditional on frozen fitted heads',
        'subgroups': 'exploratory unadjusted; intervals require at least 10 observations in each class; no subgroup selection',
        'family_annotation': 'partial DEV-only TRAIN-shared family witnesses; absent witness is not established family novelty',
        'assessment_previously_inspected': True, 'original_metrics_reproduced': True, 'test_accessed': False,
        'exact_baseline_fallback_verified': True, 'structural_heads_evaluated': False, 'seconds': time.monotonic() - started}
    atomic(ROOT / 'results/metrics.json', results)
    mm = groups['assessment']; ci = mm['ap_intervals']; point = mm['contrasts']
    checks = {name: ci[name].get('low', -np.inf) > 0 for name in ['M_minus_SP', 'M_minus_baseline', 'M_minus_S', 'M_minus_P']}
    checks['practical_baseline_gain'] = point['M_minus_baseline'] >= cfg['fusion']['practical_delta_ap']
    checks['auroc_noninferiority'] = mm['metrics']['M']['auroc'] >= mm['metrics']['baseline']['auroc'] - cfg['fusion']['maximum_auroc_decline']
    checks['positive_after_frequent_group_removal'] = all(x['contrasts'] is not None and x['contrasts']['M_minus_SP'] > 0
        for entries in influence['assessment'].values() for x in entries)
    passed = all(checks.values())
    decision = {'at_utc': now(), 'fingerprint': fp,
        'status': 'promising_exploratory_family_breadth_unresolved' if passed else 'no_go_or_inconclusive',
        'primary_point': point['M_minus_SP'], 'primary_interval': ci['M_minus_SP'],
        'checks': {k: bool(v) for k, v in checks.items()}, 'pair_terms_supported': ci['M_minus_U'].get('low', -np.inf) > 0,
        'family_breadth_established': False, 'assessment_previously_inspected': True,
        'production_authorized': False, 'test_accessed': False, 'metrics_sha256': sha(ROOT / 'results/metrics.json')}
    atomic(ROOT / 'results/decision.json', decision)
    lines = ['# Original Bernett VY pilot', '', f"Decision: **{decision['status']}**.", '',
        'Independent monomer-MSA context, frozen native PLM-interact, and five small TRAIN-only heads. The reused assessment is exploratory. No TEST input or evaluation.', '',
        f"Available pairs: TRAIN {int(fit.sum())}/4000; DEV {int((available & dev).sum())}/4000; assessment {int((available & assessment).sum())}/958. Required monomers: {len(ids)}; distinct TRAIN proteins fitted once per representation: {len(fit_indices)}.", '',
        '| Assessment model | AP | AUROC |', '|---|---:|---:|']
    for name, metric in mm['metrics'].items():
        lines.append(f"| {name} | {metric['ap']:.6f} | {metric['auroc']:.6f} |")
    lines += ['', '| Assessment AP contrast | Estimate | 95% matched protein interval |', '|---|---:|---|']
    for name, value in point.items():
        c = ci[name]; lines.append(f"| {name} | {value:+.6f} | [{c.get('low', float('nan')):+.6f}, {c.get('high', float('nan')):+.6f}] |")
    lines += ['', 'The primary contrast is M minus S+P: learned MSA context versus the combined query-only and profile control. All VY arms share availability and reliability. SP has 97 supervised parameters, M/S/P 49 each, U 17.', '',
        'M = monomer-MSA; S = query only through the same encoder; P = fixed profile/quality; SP = concatenated S/P pair descriptors; U = additive unary MSA; M_on_S = M head evaluated on query-only descriptors without refitting.', '',
        f"Selected alpha values: `{results['alphas']}`. Native TRAIN logits were not used. Historical Vx predictions retain their original gate and heads and are descriptive references.", '',
        'See metrics.json for fixed full DEV, coverage groups, length/diversity/quality strata, common-alpha sensitivity, evidence-only metrics and protein/family influence. Missing complete-family annotations limit generalization claims. A monomer-context benefit would not establish inter-protein co-evolution. No automatic continuation is enabled.']
    (ROOT / 'results/REPORT.md').write_text('\n'.join(lines) + '\n')
    print(decision, flush=True)


if __name__ == '__main__':
    main()
