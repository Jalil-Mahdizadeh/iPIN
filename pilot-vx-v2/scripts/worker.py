"""Bounded extraction with identical old features required for every usable pair."""
import argparse
import fcntl
import hashlib
import time
import traceback
from functools import lru_cache
import numpy as np
from study import ROOT, BASE, config, read, sha, text_hash, atomic, now, verify_freeze, load_record, save_record
from msa import load_monomer, pair, shuffled
from msa_diagnostics import diagnose


def inputs():
    sample = read(ROOT / 'data/sample.json')
    metadata = read(ROOT / 'data/source_metadata.json')
    arrays = np.load(ROOT / 'data/source.npz', allow_pickle=False)
    assert list(arrays['uids']) == [r['uid'] for r in sample]
    catalog = read(ROOT / 'data/monomer-catalog.json')
    proteins = read(ROOT / 'data/proteins.json')
    @lru_cache(maxsize=24)
    def mono(pid):
        item = catalog[str(pid)]; p = BASE / item['path']
        if sha(p) != item['sha256']:
            raise ValueError('Corrupt monomer: ' + str(pid))
        m = load_monomer(p)
        if m['query'] != proteins[str(pid)]['sequence'] or m['query_sha256'] != proteins[str(pid)]['sha256']:
            raise ValueError('Monomer query identity mismatch')
        return m
    return sample, metadata, arrays, mono


def extract(row, previous, old, mono, encoder):
    cfg = config(); a, b = sorted([row['a'], row['b']])
    out = {'uid': row['uid'], 'a': a, 'b': b, 'available': previous['available'], 'gate': previous['gate'],
           'reason': previous['reason'], **{k: [0.] * 128 for k in cfg['arms']}}
    if not previous['available']:
        if previous['gate'] != 0:
            raise ValueError('Inconsistent source eligibility')
        return out
    ma, mb = mono(a), mono(b)
    tokens, meta = pair(ma, mb)
    if tokens is None or meta != previous['msa']:
        raise ValueError('Paired MSA differs from frozen source: ' + row['uid'])
    seed = int(text_hash(f"{cfg['seed']}:{row['uid']}")[:16], 16)
    null, nm = shuffled(tokens, len(ma['query']), meta['taxonomy'], seed)
    if nm != previous['null']:
        raise ValueError('Pairing null differs from frozen source')
    errors = {}
    for label, data in [('true', tokens), ('shuffled', null)]:
        views, blockmeta = encoder.symmetric_views(data, len(ma['query']), row['uid'], a, b)
        error = float(np.max(np.abs(views['global'] - old[label])))
        if error > cfg['global_feature_max_absolute_error']:
            raise ValueError(f'Original global feature mismatch: {row["uid"]}:{label}:{error}')
        errors[label] = error
        out['local_' + label] = views['local'].tolist()
        out['scrambled_' + label] = views['scrambled'].tolist()
        out['pooling'] = blockmeta
    out.update({'diagnostics': diagnose(ma, mb, tokens, null, meta, nm), 'global_feature_max_absolute_error': errors,
                'tokens_sha256': hashlib.sha256(tokens.tobytes()).hexdigest(),
                'null_tokens_sha256': hashlib.sha256(null.tobytes()).hexdigest()})
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--rank', type=int, required=True); args = ap.parse_args()
    cfg = config(); rank = args.rank; world = cfg['budgets']['world_size']
    if not 0 <= rank < world:
        raise ValueError('Invalid worker rank')
    started = time.monotonic(); fingerprint = verify_freeze()
    lock = (ROOT / 'features' / f'rank-{rank}.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    sample, metadata, arrays, mono = inputs()
    # Frozen greedy assignment uses source runtime only, without labels.
    assignment = read(ROOT / 'data/worker_assignment.json')[str(rank)]
    encoder = None; completed = 0; current = None
    try:
        for i in assignment:
            if time.monotonic() - started >= cfg['budgets']['worker_wall_seconds']:
                break
            row = sample[i]; current = row['uid']
            saved = load_record(current, fingerprint)
            if saved is None:
                if metadata[i]['available'] and encoder is None:
                    from readout import LocalEncoder
                    encoder = LocalEncoder()
                t = time.monotonic()
                out = extract(row, metadata[i], {k: arrays[k][i] for k in ['true', 'shuffled']}, mono, encoder)
                out.update({'at_utc': now(), 'elapsed_seconds': time.monotonic() - t})
                save_record(current, out, fingerprint)
            else:
                if [saved['a'], saved['b']] != sorted([row['a'], row['b']]) or saved['gate'] != metadata[i]['gate']:
                    raise ValueError('Resumed identity/gate mismatch')
            completed += 1
            if completed % 20 == 0:
                status = {'at_utc': now(), 'rank': rank, 'completed': completed, 'total': len(assignment),
                          'seconds': time.monotonic() - started, 'fingerprint': fingerprint}
                atomic(ROOT / 'results' / f'worker-{rank}.json', status)
                print(status, flush=True)
    except BaseException:
        atomic(ROOT / 'results' / f'worker-{rank}-error.json', {'at_utc': now(), 'uid': current,
            'completed': completed, 'fingerprint': fingerprint, 'traceback': traceback.format_exc()})
        raise
    atomic(ROOT / 'results' / f'worker-{rank}.done.json', {'at_utc': now(), 'completed': completed,
        'total': len(assignment), 'seconds': time.monotonic() - started, 'fingerprint': fingerprint,
        'budget_stopped': completed != len(assignment)})


if __name__ == '__main__':
    main()
