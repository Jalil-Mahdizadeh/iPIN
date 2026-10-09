"""Label-free joint depth-one validation and bounded workload estimation."""
import time
import numpy as np
import torch
from study import ROOT, config, read, atomic, now, sha, check_budget
from inputs import monomer_reader, reconstruct
from q_encoder import QueryEncoder
from msa import shuffled
from common import text_hash


def main():
    started = time.monotonic(); cfg = config(); check_budget(); mono = monomer_reader()
    pairs = read(ROOT / 'data/pairs.json'); previous = {r['uid']: r for r in read(ROOT / 'data/source_metadata.json')}
    lookup = {p['uid']: p for p in pairs}; eligible = [p for p in pairs if p['available']]
    uids = list(cfg['qualification_uids'])
    for target in cfg['timing_length_targets']:
        uid = min(eligible, key=lambda p: (abs(p['length'] - target), p['uid']))['uid']
        if uid not in uids:
            uids.append(uid)
    with np.load(ROOT / 'data/source.npz', allow_pickle=False) as f:
        old_arrays = {k: f[k].copy() for k in ('uids', 'true', 'shuffled')}
    index = {uid: i for i, uid in enumerate(old_arrays['uids'])}
    enc = QueryEncoder(); cases = []
    for uid in uids:
        check_budget(); p = lookup[uid]; before = time.monotonic()
        tokens, q, meta = reconstruct(p, previous[uid], mono); reconstruction_seconds = time.monotonic() - before
        errors = {}
        if uid in cfg['qualification_uids'][:3]:
            seed = int(text_hash(f"{cfg['seed']}:{uid}")[:16], 16)
            null, nullmeta = shuffled(tokens, p['breakpoint'], meta['taxonomy'], seed)
            if nullmeta != previous[uid]['null']:
                raise ValueError('Original null changed')
            for arm, t in [('true', tokens), ('shuffled', null)]:
                out = enc.symmetric(t, p['breakpoint']); reference = old_arrays[arm][index[uid]]
                errors[arm] = float(np.max(np.abs(out - reference)))
                np.testing.assert_array_equal(out, reference)
        a, ta = enc.query(q, p['breakpoint']); b, tb = enc.query(q, p['breakpoint'])
        np.testing.assert_array_equal(a, b)
        if uid in cfg['qualification_uids']:
            rev = np.concatenate([q[:, p['breakpoint']:], q[:, :p['breakpoint']]], axis=1)
            c, _ = enc.query(rev, q.shape[1] - p['breakpoint'])
            np.testing.assert_array_equal(a, c)
        case = {'uid': uid, 'length': p['length'], 'q_tokens_sha256': p['q_tokens_sha256'],
            'reconstruction_seconds': reconstruction_seconds, 'timings': [ta, tb], 'deterministic_repeat_exact': True,
            'ab_ba_symmetry_checked': uid in cfg['qualification_uids'], 'original_max_error': errors}
        cases.append(case); print({'qualified': uid, 'length': p['length'], 'seconds': max(ta['seconds'], tb['seconds'])}, flush=True)
    buckets = []
    for c in sorted(cases, key=lambda c: c['length']):
        seconds = c['reconstruction_seconds'] + max(t['seconds'] for t in c['timings'])
        buckets.append((c['length'], max(seconds, buckets[-1][1] if buckets else 0)))
    if buckets[-1][0] < max(p['length'] for p in eligible):
        raise ValueError('Qualification misses maximum length')
    estimate = sum(next(seconds for length, seconds in buckets if length >= p['length']) for p in eligible)
    total = cfg['timing_safety_factor'] * estimate + cfg['budgets']['io_reserve_seconds'] + cfg['budgets']['analysis_reserve_seconds']
    atomic(ROOT / 'qualification/real.json', {'at_utc': now(), 'passed': True, 'labels_used': False,
        'selection': 'original qualification pairs plus maximum length and fixed label-free timing targets', 'cases': cases,
        'required_pairs': len(eligible), 'estimated_extraction_seconds': estimate, 'conservative_total_seconds': total,
        'timing_safety_factor': cfg['timing_safety_factor'], 'upper_length_timing_buckets': buckets,
        'gpu': torch.cuda.get_device_name(0), 'forbidden_heads_evaluated': False,
        'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'config.json', ROOT / 'scripts/q_encoder.py', ROOT / 'scripts/inputs.py', ROOT / 'scripts/qualify.py', ROOT / 'scripts/study.py']},
        'seconds': time.monotonic() - started})
    if total > check_budget(cfg['budgets']['shutdown_reserve_seconds']):
        raise TimeoutError('Qualified workload does not fit remaining VZ budget; do not launch')
    print({'qualified': True, 'estimated_hours': estimate / 3600, 'conservative_hours': total / 3600}, flush=True)


if __name__ == '__main__':
    main()
