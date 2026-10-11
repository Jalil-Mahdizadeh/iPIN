"""Outcome-free original feature reproduction and full-shard timing projection."""
import time
import numpy as np
from shared import ROOT, BASE, config, read, atomic, sha, now
from inputs import Inputs
from guarded_encoder import GuardedEncoder
from msa import pair, quality_features

GOLD = ['val-025136', 'val-057993', 'train-129443', 'train-073790', 'val-041603',
        'train-012468', 'val-001182', 'train-047928', 'train-027500']


def main():
    start = time.monotonic(); cfg = config(); preparation = read(ROOT / 'provenance/preparation.json')
    assert preparation['complete'] and preparation['rows'] == 222452
    if (ROOT / 'qualification/encoder.json').exists():
        raise RuntimeError('Encoder already qualified; no silent rerun')
    rows = read(ROOT / 'data/pairs.json'); by_uid = {r['uid']: r for r in rows}; inputs = Inputs(); encoder = GuardedEncoder()
    selected = [by_uid[uid] for uid in GOLD]
    for lo, hi in [(0, 512), (1024, 1536)]:
        candidates = [r for r in rows if lo < r['length'] <= hi and all(str(r[k]) in inputs.catalog for k in ['a', 'b'])
            and any(not inputs.catalog[str(r[k])]['reused_original_cache'] for k in ['a', 'b'])]
        candidates.sort(key=lambda r: r['uid'])
        for row in candidates:
            if inputs.materialize(row)[0] is not None:
                selected.append(row); break
        else:
            raise ValueError('No qualifying new-cache example in fixed length band')
    cases = []
    for row in selected:
        began = time.monotonic(); tokens, quality, info = inputs.materialize(row); input_seconds = time.monotonic() - began
        if tokens is None:
            raise ValueError('Qualification input is unexpectedly unavailable')
        a, b = inputs.monomer(info['a']), inputs.monomer(info['b'])
        np.testing.assert_array_equal(quality, quality_features(a, b, info['msa']))
        old = inputs.old_feature(row, info, quality)
        feature, timing = encoder.encode(tokens, info['msa']['length_a'])
        if old is not None:
            np.testing.assert_array_equal(feature, old)
        repeat = row['uid'] in ['val-057993', 'train-047928'] or old is None
        if repeat:
            again, _ = encoder.encode(tokens, info['msa']['length_a']); np.testing.assert_array_equal(feature, again)
            bp = info['msa']['length_a']; reverse = np.concatenate([tokens[:, bp:], tokens[:, :bp]], axis=1)
            swapped, _ = encoder.encode(reverse, tokens.shape[1] - bp); np.testing.assert_array_equal(feature, swapped)
        cases.append({'uid': row['uid'], 'length': row['length'], 'depth': len(tokens),
            'input_seconds': input_seconds, 'timing': timing, 'original_feature_exact': old is not None,
            'original_quality_exact': True, 'repeat_and_symmetry_checked': repeat, 'token_sha256': info['token_sha256']})
    fallback_cases = []
    old_rows = read(BASE / 'data/sample.json')
    for reason in ['long_query', 'low_quality_coverage', 'missing_exact_query']:
        for old_row in sorted(old_rows, key=lambda r: r['uid']):
            old = read(BASE / 'features' / (old_row['uid'] + '.json'))
            if old['reason'] != reason:
                continue
            row = by_uid[old_row['uid']]; tokens, quality, info = inputs.materialize(row)
            assert tokens is None; inputs.old_feature(row, info, quality)
            if info['msa'] is not None:
                a, b = inputs.monomer(info['a']), inputs.monomer(info['b'])
                expected_tokens, meta = pair(a, b, cfg); assert expected_tokens is None and meta == info['msa']
                np.testing.assert_array_equal(quality_features(a, b, meta), quality)
            fallback_cases.append(row['uid']); break
        else:
            raise ValueError('Missing original fallback qualification case')
    # Monotone upper-length buckets. Include CPU pairing and cold cache loading.
    buckets = []
    for case in sorted(cases, key=lambda c: c['length']):
        seconds = case['timing']['seconds'] + case['input_seconds']
        buckets.append([case['length'], max(seconds, max((v for _, v in buckets), default=0))])
    assert buckets[-1][0] == 1536
    estimated = []
    for row in rows:
        entries = [inputs.catalog.get(str(row[k])) for k in ['a', 'b']]
        candidate = row['length'] <= 1536 and all(e and e['quality_fraction'] >= .5 for e in entries)
        estimated.append(next(v for length, v in buckets if length >= row['length']) if candidate else .02)
    shards = read(ROOT / 'data/shards.json'); shard_seconds = [sum(estimated[i] for i in shard) for shard in shards]
    conservative = max(shard_seconds) * cfg['budgets']['timing_safety_factor'] + 600 + cfg['budgets']['analysis_reserve_seconds']
    amendment = ROOT / 'provenance/resource-amendment-1.json'
    if amendment.exists():
        conservative = max(conservative, read(amendment)['prior_conservative_estimate_floor_hours'] * 3600)
    atomic(ROOT / 'qualification/timing-estimate.json', {'at_utc': now(), 'cases': cases,
        'estimated_shard_seconds': shard_seconds, 'conservative_wall_hours': conservative / 3600,
        'registered_wall_hours': cfg['budgets']['batch_wall_hours'], 'labels_used': False})
    if conservative >= cfg['budgets']['batch_wall_hours'] * 3600:
        raise RuntimeError(f'Conservative walltime estimate {conservative / 3600:.2f}h exceeds the fixed allocation')
    code = {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'scripts/qualify.py', ROOT / 'scripts/inputs.py',
        ROOT / 'scripts/guarded_encoder.py', ROOT / 'scripts/shared.py', ROOT / 'config.json']}
    atomic(ROOT / 'qualification/encoder.json', {'at_utc': now(), 'passed': True, 'cases': cases,
        'fallback_cases': fallback_cases, 'upper_length_timing_buckets': buckets, 'estimated_shard_seconds': shard_seconds,
        'estimated_allocated_gpu_hours': max(shard_seconds) * 8 / 3600,
        'conservative_total_wall_hours': conservative / 3600, 'labels_used': False,
        'shuffle_called': False, 'forbidden_heads_called': False, 'code_sha256': code, 'seconds': time.monotonic() - start})
    print({'qualification_passed': True, 'cases': len(cases), 'conservative_wall_hours': conservative / 3600}, flush=True)


if __name__ == '__main__':
    main()
