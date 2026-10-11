"""Previously selected, label-free real cases; no head fitting or new outcomes."""
import time
import numpy as np
import torch
from study import ROOT, config, read, atomic, now, sha
from worker import inputs, extract
from readout import LocalEncoder
from msa import pair


def main():
    started = time.monotonic(); cfg = config()
    resource = read(ROOT / 'provenance/resource-start.json')
    if time.time() > resource['qualification_deadline_unix']:
        raise TimeoutError('Qualification reserve exhausted')
    sample, previous, arrays, mono = inputs(); lookup = {r['uid']: i for i, r in enumerate(sample)}
    enc = LocalEncoder(); results = []
    for uid in cfg['qualification_uids']:
        i = lookup[uid]; row = sample[i]; t = time.monotonic()
        out = extract(row, previous[i], {k: arrays[k][i] for k in ['true', 'shuffled']}, mono, enc)
        for arm in cfg['arms']:
            assert np.isfinite(out[arm]).all()
            np.testing.assert_array_equal(out[arm][:64], arrays['shuffled' if arm.endswith('shuffled') else 'true'][i][:64])
        assert not np.array_equal(out['local_true'][64:], out['scrambled_true'][64:])
        if not results:
            a, b = sorted([row['a'], row['b']]); ma, mb = mono(a), mono(b); tokens, _ = pair(ma, mb)
            reverse = np.c_[tokens[:, len(ma['query']):], tokens[:, :len(ma['query'])]]
            again, _ = enc.symmetric_views(reverse, len(mb['query']), uid, b, a)
            np.testing.assert_array_equal(again['local'], out['local_true'])
            np.testing.assert_array_equal(again['scrambled'], out['scrambled_true'])
        torch.cuda.synchronize()
        results.append({'uid': uid, 'length': row['length'], 'seconds': time.monotonic() - t,
            'original_global_max_error': out['global_feature_max_absolute_error'], 'pooling': out['pooling'],
            'diagnostics': out['diagnostics'], 'peak_cuda_bytes': torch.cuda.max_memory_allocated()})
        print({'qualified': uid, 'seconds': results[-1]['seconds']}, flush=True)
    atomic(ROOT / 'qualification/real.json', {'at_utc': now(), 'passed': True, 'cases': results,
        'selection': 'same pre-outcome real qualification cases as original pilot', 'labels_used': False,
        'code_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'scripts').glob('*.py'))},
        'seconds': time.monotonic() - started})


if __name__ == '__main__': main()
