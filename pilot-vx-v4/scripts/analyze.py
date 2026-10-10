"""Exactly one TRAIN-only I head; immutable historical predictions and matched DEV."""
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from study import ROOT, config, read, atomic, atomic_npz, now, sha, verify_freeze, load_record, fuse, check_budget
from heads import fit_head, evidence, references
from statistics_v4 import CONTRASTS, bootstrap, summarize
from decision import decide
from input_report import aggregate


def load_complete(fp):
    sample = read(ROOT / 'data/sample.json'); pairs = read(ROOT / 'data/pairs.json')
    if len(sample) != config()['expected']['pairs'] or len(pairs) != len(sample):
        raise ValueError('Fixed population length changed')
    for r, p in zip(sample, pairs):
        if r['uid'] != p['uid'] or sorted([r['a'], r['b']]) != [p['a'], p['b']] or p['available'] != (p['gate'] > 0):
            raise ValueError('Pair identities or gate changed')
    with ThreadPoolExecutor(max_workers=8) as ex:
        records = list(ex.map(lambda p: load_record(p['uid'], fp, p), pairs))
    if any(r is None for r in records):
        raise RuntimeError('Incomplete computational coverage; no I fit allowed')
    for p, r in zip(pairs, records):
        timing = r['timing']
        if p['available']:
            if timing is None or timing['orientation_depths'] != [p['paired_depth']] * 2 or timing['orientation_layers'] != list(range(16)) * 2:
                raise ValueError('Missing or incorrect original-depth/layer execution audit')
            if timing['length'] != p['length'] or timing['breakpoint'] != p['breakpoint']:
                raise ValueError('Encoded coordinates differ from original input')
        elif timing is not None:
            raise ValueError('Originally unavailable pair was encoded')
    x = np.asarray([r['I'] for r in records], dtype=float)
    if not np.isfinite(x).all() or x.shape != (len(sample), 128):
        raise ValueError('Invalid complete I feature matrix')
    return sample, pairs, x


def strata(sample, pairs, cfg):
    y = np.array([r['label'] for r in sample]); available = np.array([p['available'] for p in pairs])
    train = np.array([r['split'] == 'train' for r in sample]); dev = ~train
    ass = np.array([r['role'] == 'assessment' for r in sample]); length = np.array([r['length'] for r in sample])
    masks = {'assessment': ass, 'fixed_dev_sample': dev, 'calibration': np.array([r['role'] == 'calibration' for r in sample]),
             'crossing_descriptive': np.array([r['role'] == 'crossing' for r in sample])}
    for name, mask in [('assessment', ass), ('fixed_dev_sample', dev)]:
        masks[name + '/eligible'] = mask & available; masks[name + '/fallback'] = mask & ~available
        for label, group in [('le512', length <= 512), ('513_1024', (length > 512) & (length <= 1024)),
                             ('1025_1536', (length > 1024) & (length <= 1536)), ('over1536', length > 1536)]:
            masks[name + '/length_' + label] = mask & group
    cuts = read(ROOT / 'data/v2_metrics.json')['diagnostic_stratum_cuts_from_eligible_train']
    diagnostics = read(ROOT / 'data/diagnostics.json')
    for name, cut in cuts.items():
        values = np.array([d[name] if d is not None else np.nan for d in diagnostics])
        expected = np.unique(np.quantile(values[train & available], [.25, .5, .75]))
        np.testing.assert_array_equal(expected, cut)
        bins = np.searchsorted(cut, values, side='right')
        for i in range(len(cut) + 1):
            masks[f'assessment/{name}/train_quantile_bin_{i}'] = ass & available & (bins == i)
    ann = read(ROOT / 'data/dev_family_witnesses.json'); families = []; shared = np.full(len(sample), -1)
    adequate = np.zeros(len(sample), dtype=bool)
    for i, r in enumerate(sample):
        a, b = ann.get(str(r['a'])), ann.get(str(r['b']))
        if a is not None and b is not None:
            shared[i] = int(a['shared_family'] == 'True') + int(b['shared_family'] == 'True')
            adequate[i] = a.get('adequate') == 'True' and b.get('adequate') == 'True'
        families.append({v['family'] for v in (a, b) if v is not None and v['family']})
    for i in (-1, 0, 1, 2):
        masks[f'assessment/family_witness_count_{i}'] = ass & available & (shared == i)
    masks['assessment/both_family_annotations_adequate'] = ass & available & adequate
    return masks, cuts, families


def main():
    started = time.monotonic(); cfg = config(); fp = verify_freeze(); check_budget()
    if (ROOT / 'results/fit-start.json').exists():
        raise RuntimeError('I fit already started; no automatic repeated analysis')
    sample, pairs, x = load_complete(fp)
    y = np.array([r['label'] for r in sample]); gate = np.array([p['gate'] for p in pairs]); available = gate > 0
    train = np.array([r['split'] == 'train' for r in sample]); dev = ~train
    cal = np.array([r['role'] == 'calibration' for r in sample]); ass = np.array([r['role'] == 'assessment' for r in sample]); fit = train & available
    counts = {name: [int(np.sum(mask & (y == c))) for c in (0, 1)] for name, mask in [('fit', fit), ('calibration', cal), ('assessment', ass)]}
    if counts != read(ROOT / 'data/source_metrics.json')['fit_counts']:
        raise ValueError('Fitting/calibration/assessment class counts changed')
    if min(counts['fit']) < cfg['fusion']['minimum_fit_per_class'] or min(counts['calibration'] + counts['assessment']) < cfg['fusion']['minimum_calibration_assessment_per_class']:
        raise ValueError('Insufficient class support')
    base, scores, ev, oldgate = references(ROOT, sample); np.testing.assert_array_equal(gate, oldgate)
    masks, cuts, families = strata(sample, pairs, cfg)
    input_diagnostics = aggregate(read(ROOT / 'data/input-audits.json'), sample, pairs, {'TRAIN': train, **masks})
    atomic(ROOT / 'results/input-diagnostics.json', input_diagnostics)
    atomic(ROOT / 'results/fit-start.json', {'at_utc': now(), 'fingerprint': fp, 'arms': ['I'], 'fit_counts': counts,
        'fit_pair_uids': [r['uid'] for r, keep in zip(sample, fit) if keep], 'TRAIN_baseline_used': False})
    head, ev['I'], scores['I'] = fit_head(x, y, fit, cal, base, gate, cfg)
    atomic(ROOT / 'results/heads.json', {'at_utc': now(), 'fingerprint': fp, 'I': head,
        'fit_pair_uids': [r['uid'] for r, keep in zip(sample, fit) if keep], 'TRAIN_baseline_used': False})
    for z in scores.values():
        np.testing.assert_array_equal(z[dev & ~available], base[dev & ~available])
    groups = {name: summarize(mask, y, scores, available) for name, mask in masks.items()}
    # Original Vx assessment-only protein universe and RNG preserve its historical interval.
    for population, mask in [('assessment', ass), ('fixed_dev_sample', dev)]:
        local = {name: m[mask] for name, m in masks.items() if name == population or name.startswith(population + '/')}
        ci = bootstrap([r for r, keep in zip(sample, mask) if keep], y[mask], {k: z[mask] for k, z in scores.items()}, local, cfg['fusion'], cfg['seed'])
        for name, intervals in ci.items():
            groups[name]['ap_intervals'] = intervals
    previous = read(ROOT / 'data/source_metrics.json')
    for pop in ('assessment', 'fixed_dev_sample'):
        for name, old in [('true_minus_baseline', 'true-minus-baseline'), ('true_minus_shuffled', 'true-minus-shuffled')]:
            c, ref = groups[pop]['ap_intervals'][name], previous['results'][pop]['ap_intervals'][old]
            np.testing.assert_allclose([c['low'], c['high']], [ref['low'], ref['high']], atol=1e-12, rtol=1e-12)
    vz = read(ROOT / 'data/vz_metrics.json')
    for pop in ('assessment', 'assessment/eligible', 'fixed_dev_sample'):
        for name, ref in vz['results'][pop]['ap_intervals'].items():
            c = groups[pop]['ap_intervals'][name]
            np.testing.assert_allclose([c['low'], c['high']], [ref['low'], ref['high']], atol=1e-12, rtol=1e-12)
            if c['replicates'] != ref['replicates']:
                raise ValueError('Historical bootstrap population differs')
    v3 = read(ROOT / 'data/v3_metrics.json')
    for pop in ('assessment', 'assessment/eligible', 'fixed_dev_sample'):
        for name, ref in v3['results'][pop]['ap_intervals'].items():
            c = groups[pop]['ap_intervals'][name]
            np.testing.assert_allclose([c['low'], c['high']], [ref['low'], ref['high']], atol=1e-12, rtol=1e-12)
            if c['replicates'] != ref['replicates']: raise ValueError('Historical C bootstrap population differs')
    common = {}; unfused = {}
    for pop, mask in [('assessment', ass), ('assessment/eligible', ass & available), ('fixed_dev_sample', dev), ('calibration', cal)]:
        common[pop] = {str(a): {arm: {'ap': float(average_precision_score(y[mask], fuse(base, ev[arm], gate, a)[mask])),
            'auroc': float(roc_auc_score(y[mask], fuse(base, ev[arm], gate, a)[mask]))} for arm in ('I', 'C', 'Q', 'true', 'shuffled', 'quality')}
            for a in cfg['fusion']['alpha_grid']}
        m = mask & available
        unfused[pop] = {arm: {'ap': float(average_precision_score(y[m], z[m])), 'auroc': float(roc_auc_score(y[m], z[m]))} for arm, z in ev.items()}
    influence = {}
    for kind in ('protein', 'family_witness'):
        members = [[r['a'], r['b']] if kind == 'protein' else sorted(families[i]) for i, r in enumerate(sample)]
        count = Counter(v for i in np.flatnonzero(ass) for v in members[i]); influence[kind] = []
        for member, _ in sorted(count.items(), key=lambda item: (-item[1], str(item[0])))[:10]:
            selected = np.array([member in values for values in members]); m = ass & ~selected
            group = summarize(m, y, scores, available)
            influence[kind].append({'group': str(member), 'removed_pairs': int(np.sum(ass & selected)),
                'remaining_pairs': int(m.sum()), 'contrasts': group.get('contrasts')})
    results = {'at_utc': now(), 'fingerprint': fp, 'primary_contrast': 'I_minus_shuffled', 'contrast_definitions': CONTRASTS,
        'results': groups, 'fit_counts': counts, 'alphas': {'I': head['alpha'], 'Q': read(ROOT / 'data/vz_heads.json')['Q']['alpha'], 'C': read(ROOT / 'data/v3_heads.json')['C']['alpha'], **{k: h['alpha'] for k, h in read(ROOT / 'data/source_heads.json').items()}},
        'common_alpha_descriptive': common, 'unfused_eligible_head_metrics': unfused, 'diagnostic_stratum_cuts_from_eligible_train': cuts,
        'assessment_group_influence': influence, 'bootstrap': 'Original Vx population-specific matched protein Poisson multipliers, 1000 draws, seed 20261007, endpoint product weights, 95% percentile intervals conditional on fitted heads and selected alpha',
        'subgroup_intervals': 'Exploratory, unadjusted; minimum 10 per class', 'assessment_previously_inspected': True,
        'original_metrics_and_intervals_reproduced': True, 'original_verified_predictions_reproduced': True, 'VZ_predictions_and_intervals_reproduced': True, 'V3_predictions_and_intervals_reproduced': True,
        'input_diagnostics_sha256': sha(ROOT / 'results/input-diagnostics.json'), 'R_evaluated': False,
        'exact_baseline_fallback': True, 'test_accessed': False, 'forbidden_heads_evaluated': False, 'seconds': time.monotonic() - started}
    atomic(ROOT / 'results/metrics.json', results)
    decision = {**decide(groups, cfg), 'at_utc': now(), 'fingerprint': fp, 'metrics_sha256': sha(ROOT / 'results/metrics.json')}
    atomic(ROOT / 'results/decision.json', decision)
    atomic_npz(ROOT / 'results/dev_predictions.npz', uids=np.array([r['uid'] for r, keep in zip(sample, dev) if keep]), labels=y[dev],
               gate=gate[dev], **{k: z[dev] for k, z in scores.items()})
    atomic_npz(ROOT / 'results/dev_evidence.npz', **{k: z[dev] for k, z in ev.items()})
    mm = groups['assessment']
    lines = ['# Vx v4 independent homolog sampling', '', f"Decision: **{decision['status']}**.", '',
        'One new I head; original Vx true/shuffled/quality, VZ Q and Vx v3 C predictions reused unchanged. Same masks, eligibility, depth, gate, and native PLM-interact. Assessment is exploratory; no TEST access. R is deferred.', '',
        '| Assessment model | AP | AUROC |', '|---|---:|---:|']
    for arm in ('baseline', 'Q', 'C', 'I', 'shuffled', 'true', 'quality'):
        v = mm['metrics'][arm]; lines.append(f"| {arm} | {v['ap']:.6f} | {v['auroc']:.6f} |")
    lines += ['', '| Assessment AP contrast | Estimate | 95% matched protein interval |', '|---|---:|---|']
    for name, value in mm['contrasts'].items():
        c = mm['ap_intervals'][name]; lines.append(f"| {name} | {value:+.6f} | [{c['low']:+.6f}, {c['high']:+.6f}] |")
    lines += ['', f"Selected I alpha: **{head['alpha']}**. Historical Q alpha remains 0.5; true/shuffled/C/quality alphas remain 1. All I transforms/head fitted on {fit.sum()} original eligible TRAIN pairs; assessment has {ass.sum()} rows, {(ass & available).sum()} eligible.", '',
        f"Useful I gain: {decision['useful_I_gain']}; recovery within 0.010 AP margin: {decision['independent_recovery_within_margin']}; recovery beyond Q: {decision['independent_homologs_recover_gain_beyond_Q']}; eligible-only noninferiority: {decision['eligible_only_noninferiority']}; three-way equivalence: {decision['all_I_true_shuffled_equivalent']}; I beyond quality/profile interval: {decision['I_exceeds_quality_interval']}.", '',
        'Noninferiority permits up to 0.010 AP loss (roughly 45% of the original shuffled gain). It is distinct from equivalence, which requires the whole interval inside ±0.010. True-minus-shuffled is unchanged; adding I cannot strengthen its evidence.', '',
        'I preserves query, original depth/masks/gate and intact homolog sequences. It selects A/B independently without the accession intersection. Profiles, diversity, coverage and matching change together; recovery supports practical sufficiency, not a unique biological mechanism. Additional availability is counted separately without new predictions. R remains deferred.', '',
        'See input-diagnostics.json for selection/diversity/coverage diagnostics and coverage.json for label-free potential availability, and metrics.json for eligible-only and length/depth/coverage/family diagnostics, common-alpha and unfused-head results. No subgroup rescue, refitting, production continuation, or TEST evaluation is enabled.']
    (ROOT / 'results/REPORT.md').write_text('\n'.join(lines) + '\n')
    print({'fitted': 'I', 'alpha': head['alpha'], 'decision': decision['status']}, flush=True)


if __name__ == '__main__':
    main()
