"""Label-independent monomer/depth-one qualification and conservative timing."""
import hashlib
import time
import numpy as np
import torch
from study import ROOT, PROJECT, config, read, sha, atomic, atomic_npz, now, check_budget
from encoder import Encoder


def tokens_for(meta):
    path = ROOT / meta['selected_path']
    if sha(path) != meta['selected_sha256']:
        raise ValueError('Prepared token artifact changed')
    with np.load(path, allow_pickle=False) as f:
        tokens = f['tokens'].copy()
    if hashlib.sha256(tokens.tobytes()).hexdigest() != meta['tokens_sha256']:
        raise ValueError('Prepared tokens changed')
    return tokens


def main():
    cfg = config(); started = time.monotonic(); check_budget()
    if (ROOT / 'qualification/real.json').exists() or (ROOT / 'provenance/freeze.json').exists():
        raise RuntimeError('Qualification already finalized')
    inputs = read(ROOT / 'provenance/inputs.json')
    if sha(PROJECT / inputs['image']) != inputs['image_sha256']:
        raise ValueError('Image integrity failure')
    monomers = read(ROOT / 'data/monomers.json'); required = read(ROOT / 'data/required-proteins.json')
    deep = [p for p in required if monomers[p]['retained_depth'] >= 120] or required
    cases = []
    for target in cfg['qualification']['length_targets']:
        pid = min(deep, key=lambda p: (abs(monomers[p]['length'] - target), -monomers[p]['retained_depth'], int(p)))
        if pid not in cases:
            cases.append(pid)
    for pid in [max(required, key=lambda p: (monomers[p]['length'], monomers[p]['retained_depth'], -int(p))),
                min(required, key=lambda p: (monomers[p]['retained_depth'], monomers[p]['length'], int(p)))]:
        if pid not in cases:
            cases.append(pid)
    encoder = Encoder(); results = []
    for i, pid in enumerate(cases):
        check_budget(); meta = monomers[pid]; tokens = tokens_for(meta); repeated = []
        M, S, timing = encoder.both(tokens); repeated.append(timing)
        for _ in range(cfg['qualification']['timing_repetitions'] - 1):
            mm, ss, tm = encoder.both(tokens); repeated.append(tm)
            np.testing.assert_array_equal(M, mm); np.testing.assert_array_equal(S, ss)
        if np.array_equal(M, S):
            raise ValueError('Real MSA context had no effect on query representation')
        case = {'pid': pid, 'length': meta['length'], 'depth': meta['retained_depth'], 'timings': repeated,
                'context_rms_change': float(np.sqrt(np.mean((M.astype(float) - S) ** 2))),
                'deterministic_repeat_exact': True, 'query_sha256': meta['query_sha256'],
                'tokens_sha256': meta['tokens_sha256']}
        if i == 0:
            rng = np.random.default_rng(cfg['seed']); order = np.r_[0, rng.permutation(np.arange(1, len(tokens)))]
            shuffled, _ = encoder.one(tokens[order])
            relative = float(np.sqrt(np.mean((M.astype(float) - shuffled) ** 2)) / max(1e-12, np.sqrt(np.mean(M.astype(float) ** 2))))
            case['row_permutation_relative_rms'] = relative
            if relative > cfg['qualification']['row_permutation_max_relative_rms']:
                raise ValueError('Large unexpected homolog-row-order dependence')
            _, again_query, _ = encoder.both(tokens[order])
            np.testing.assert_array_equal(S, again_query)
        atomic_npz(ROOT / 'qualification' / f'monomer-{pid}.npz', M=M, S=S)
        results.append(case); print({'qualified': pid, 'length': meta['length'], 'depth': meta['retained_depth'], 'timing': timing}, flush=True)
    # Conservative upper-length-bucket estimates from deep MSAs; no outcome-derived workload choice.
    anchors = sorted((r['length'], max(t['M']['seconds'] + t['S']['seconds'] for t in r['timings'])) for r in results)
    high = 0.; upper = []
    for length, seconds in anchors:
        high = max(high, seconds); upper.append((length, high))
    def cost(pid):
        length = monomers[pid]['length']
        return next((seconds for cap, seconds in upper if cap >= length), upper[-1][1])
    estimated = sum(cost(pid) for pid in required)
    budgeted = cfg['qualification']['budget_safety_factor'] * estimated + 600 + cfg['budgets']['analysis_reserve_seconds']
    if budgeted > check_budget(cfg['budgets']['shutdown_reserve_seconds']):
        raise TimeoutError('Qualified workload does not fit separate VY budget; no outcome fitting or automatic scope reduction')
    atomic(ROOT / 'qualification/real.json', {'at_utc': now(), 'passed': True, 'labels_used': False,
        'selection': 'deep monomers nearest fixed length targets plus longest and shallowest required cases',
        'cases': results, 'required_monomers': len(required), 'estimated_extraction_seconds': estimated,
        'conservative_total_seconds': budgeted, 'timing_safety_factor': cfg['qualification']['budget_safety_factor'],
        'upper_length_timing_buckets': upper, 'seconds': time.monotonic() - started,
        'gpu': torch.cuda.get_device_name(), 'forbidden_heads_evaluated': False,
        'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [ROOT / 'config.json', ROOT / 'scripts/encoder.py',
            ROOT / 'scripts/monomers.py', ROOT / 'scripts/study.py', ROOT / 'scripts/qualify.py']}})
    print({'qualified': True, 'estimated_seconds': estimated, 'conservative_seconds': budgeted}, flush=True)


if __name__ == '__main__':
    main()
