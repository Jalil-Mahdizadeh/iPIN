"""Label-free I execution checks, historical reproduction, and bounded timing."""
import time
import numpy as np
import torch
from study import ROOT, config, read, atomic, now, sha, token_hash, check_budget
from inputs import monomer_reader, independent_pair, prepared
from i_encoder import IndependentEncoder
from msa import shuffled
from common import text_hash


def main():
    started = time.monotonic(); cfg = config(); check_budget()
    if (ROOT / 'qualification/real.json').exists():
        raise RuntimeError('Qualification already completed; no automatic rerun')
    # Qualification uses audited prepared inputs; alternate draws are never fitted.
    if not read(ROOT / 'data/vz_verification.json')['passed']:
        raise ValueError('Unverified historical reference')
    pairs = read(ROOT / 'data/pairs.json'); previous = {r['uid']: r for r in read(ROOT / 'data/source_metadata.json')}
    pairs = [{**p, 'paired_depth': previous[p['uid']]['msa']['retained_homologs'] + 1} if p['available'] else p for p in pairs]
    mono = monomer_reader()
    lookup = {p['uid']: p for p in pairs}; eligible = [p for p in pairs if p['available']]
    maximum_depth = max(p['paired_depth'] for p in eligible)
    deepest = [p for p in eligible if p['paired_depth'] == maximum_depth]
    uids = list(cfg['qualification_uids'])
    for target in cfg['timing_length_targets']:
        uid = min(deepest, key=lambda p: (abs(p['length'] - target), p['uid']))['uid']
        if uid not in uids:
            uids.append(uid)
    with np.load(ROOT / 'data/source.npz', allow_pickle=False) as f:
        old = {k: f[k].copy() for k in ('uids', 'true', 'shuffled')}
    index = {uid: i for i, uid in enumerate(old['uids'])}
    enc = IndependentEncoder(); cases = []
    for uid in uids:
        check_budget(); p = lookup[uid]; before = time.monotonic()
        t, c = prepared(p); input_seconds = time.monotonic() - before
        errors = {}; alternate = []
        if uid in cfg['qualification_uids'][:3]:
            seed = int(text_hash(f"{cfg['seed']}:{uid}")[:16], 16)
            null, nullmeta = shuffled(t, p['breakpoint'], previous[uid]['msa']['taxonomy'], seed)
            if nullmeta != previous[uid]['null']:
                raise ValueError('Original row shuffle changed')
            for arm, tokens in [('true', t), ('shuffled', null)]:
                out = enc.symmetric(tokens, p['breakpoint']); reference = old[arm][index[uid]]
                errors[arm] = float(np.max(np.abs(out - reference)))
                np.testing.assert_array_equal(out, reference)
        a, ta = enc.independent(c, p['breakpoint'], p['paired_depth'])
        b, tb = enc.independent(c, p['breakpoint'], p['paired_depth'])
        np.testing.assert_array_equal(a, b)
        if uid in cfg['qualification_uids']:
            rev = np.concatenate([c[:, p['breakpoint']:], c[:, :p['breakpoint']]], axis=1)
            reverse, _ = enc.independent(rev, c.shape[1] - p['breakpoint'], p['paired_depth'])
            np.testing.assert_array_equal(a, reverse)
        if uid in cfg['qualification_uids'][:3]:
            for replicate in cfg['independent_sampling']['qualification_only_replicates']:
                cc, selected, _ = independent_pair(mono(p['a']), mono(p['b']), t, p, replicate)
                xx, timing = enc.independent(cc, p['breakpoint'], p['paired_depth'])
                good = t[0] != 26
                alternate.append({'replicate': replicate, 'tokens_sha256': token_hash(cc), 'selected_keys': selected,
                    'changed_fraction_vs_primary_I': float((cc[1:, good] != c[1:, good]).mean()),
                    'feature_rms_difference_vs_primary_I': float(np.sqrt(np.mean((xx - a)**2))),
                    'feature_max_difference_vs_primary_I': float(np.max(np.abs(xx - a))),
                    'primary_feature_rms': float(np.sqrt(np.mean(a**2))), 'timing': timing,
                    'outcomes_evaluated': False, 'eligible_for_primary_features': False})
        cases.append({'uid': uid, 'length': p['length'], 'paired_depth': p['paired_depth'],
            'i_tokens_sha256': token_hash(c), 'prepared_input_read_seconds': input_seconds,
            'timings': [ta, tb], 'deterministic_repeat_exact': True,
            'ab_ba_symmetry_checked': uid in cfg['qualification_uids'], 'original_max_error': errors,
            'qualification_only_alternate_replicates': alternate})
        print({'qualified': uid, 'length': p['length'], 'depth': p['paired_depth'], 'seconds': max(ta['seconds'], tb['seconds'])}, flush=True)
    buckets = []
    for c in sorted((c for c in cases if c['paired_depth'] == maximum_depth), key=lambda c: c['length']):
        seconds = max(v['prepared_input_read_seconds'] + max(t['seconds'] for t in v['timings'])
                      for v in cases if v['length'] <= c['length'])
        buckets.append((c['length'], seconds))
    if buckets[-1][0] < max(p['length'] for p in eligible):
        raise ValueError('Qualification misses maximum length at maximum retained depth')
    estimate = sum(next(seconds for length, seconds in buckets if length >= p['length']) for p in eligible)
    total = cfg['timing_safety_factor'] * estimate + cfg['budgets']['io_reserve_seconds'] + cfg['budgets']['analysis_reserve_seconds']
    if total > check_budget(cfg['budgets']['shutdown_reserve_seconds']):
        raise TimeoutError('Qualified workload does not fit remaining Vx v4 budget; do not launch')
    atomic(ROOT / 'qualification/real.json', {'at_utc': now(), 'passed': True, 'labels_used': False,
        'selection': 'original qualification pairs plus maximum length and fixed label-free length targets at maximum retained depth',
        'cases': cases, 'required_pairs': len(eligible), 'estimated_extraction_seconds': estimate,
        'conservative_total_seconds': total, 'timing_safety_factor': cfg['timing_safety_factor'],
        'timing_anchor_depth': maximum_depth, 'upper_length_timing_buckets': buckets,
        'timing_includes_prepared_input_read': True,
        'gpu': torch.cuda.get_device_name(0), 'forbidden_heads_evaluated': False, 'R_evaluated': False,
        'primary_replicate': 0, 'alternate_replicates_are_qualification_only': True,
        'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'config.json', ROOT / 'scripts/i_encoder.py',
            ROOT / 'scripts/inputs.py', ROOT / 'scripts/qualify.py', ROOT / 'scripts/study.py', ROOT / 'scripts/diagnostics.py']},
        'seconds': time.monotonic() - started})
    print({'qualified': True, 'estimated_hours': estimate / 3600, 'conservative_hours': total / 3600}, flush=True)


if __name__ == '__main__':
    main()
