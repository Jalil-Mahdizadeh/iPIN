"""One bounded GPU worker, with one shared feature artifact per required protein."""
import fcntl
import time
import traceback
from study import ROOT, config, read, atomic, now, check_budget, verify_freeze, save_feature, load_feature
from encoder import Encoder
from qualify import tokens_for


def main():
    started = time.monotonic(); cfg = config(); fp = verify_freeze()
    lock = (ROOT / 'features/worker.lock').open('a+'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    monomers = read(ROOT / 'data/monomers.json'); ids = read(ROOT / 'data/required-proteins.json')
    # Stable descending length catches expensive numerical cases early without using labels.
    order = sorted(ids, key=lambda p: (-monomers[p]['length'], int(p)))
    encoder = Encoder(); count = 0; current = None
    reserve = cfg['budgets']['analysis_reserve_seconds'] + cfg['budgets']['shutdown_reserve_seconds']
    try:
        for pid in order:
            current = pid; check_budget(reserve)
            if load_feature(pid, fp, monomers[pid]) is not None:
                raise RuntimeError('Unexpected existing production artifact; no automatic resumed extraction')
            meta = monomers[pid]; tokens = tokens_for(meta)
            M, S, timing = encoder.both(tokens)
            save_feature(pid, M, S, fp, {'query_sha256': meta['query_sha256'], 'tokens_sha256': meta['tokens_sha256'],
                'at_utc': now(), 'length': meta['length'], 'retained_depth': meta['retained_depth'], 'timings': timing})
            count += 1
            if count % 20 == 0 or count == len(order):
                status = {'at_utc': now(), 'completed': count, 'total': len(order), 'seconds': time.monotonic() - started,
                          'current_length': meta['length'], 'fingerprint': fp}
                atomic(ROOT / 'results/worker.json', status); print(status, flush=True)
        atomic(ROOT / 'results/worker.done.json', {'at_utc': now(), 'completed': count, 'total': len(order),
            'seconds': time.monotonic() - started, 'fingerprint': fp, 'budget_stopped': False})
    except BaseException:
        atomic(ROOT / 'results/worker-error.json', {'at_utc': now(), 'pid': current, 'completed': count,
            'total': len(order), 'fingerprint': fp, 'traceback': traceback.format_exc()})
        raise


if __name__ == '__main__':
    main()
