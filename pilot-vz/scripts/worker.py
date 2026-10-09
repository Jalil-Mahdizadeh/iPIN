"""One GPU, one new view; biological fallback is copied, never inferred from errors."""
import time
import traceback
import numpy as np
from study import ROOT, config, read, atomic, now, verify_freeze, check_budget, load_record, save_record
from inputs import monomer_reader, reconstruct


def main():
    started = time.monotonic(); cfg = config(); fp = verify_freeze()
    pairs = read(ROOT / 'data/pairs.json'); previous = read(ROOT / 'data/source_metadata.json')
    old = {r['uid']: r for r in previous}; mono = monomer_reader(); encoder = None; completed = 0; encoded = 0
    # Longest first catches maximum memory/resource failures before short examples dominate progress.
    ordered = sorted(pairs, key=lambda p: (not p['available'], -p['length'], p['uid']))
    try:
        for p in ordered:
            check_budget(cfg['budgets']['analysis_reserve_seconds'] + cfg['budgets']['shutdown_reserve_seconds'])
            saved = load_record(p['uid'], fp, p)
            if saved is not None:
                raise RuntimeError('Unexpected pre-existing worker output; no automatic retry')
            if p['available']:
                tokens, q, _ = reconstruct(p, old[p['uid']], mono)
                if encoder is None:
                    from q_encoder import QueryEncoder
                    encoder = QueryEncoder()
                feature, timing = encoder.query(q, p['breakpoint']); encoded += 1
            else:
                feature, timing = np.zeros(128), None
            save_record(p['uid'], feature, fp, p, timing); completed += 1
            if completed % 20 == 0:
                progress = {'at_utc': now(), 'completed': completed, 'total': len(pairs), 'encoded': encoded,
                    'required_encodings': 4453, 'current_length': p['length'], 'seconds': time.monotonic() - started, 'fingerprint': fp}
                atomic(ROOT / 'results/worker.json', progress); print(progress, flush=True)
    except BaseException:
        atomic(ROOT / 'results/worker-error.json', {'at_utc': now(), 'fingerprint': fp,
            'uid': p['uid'] if 'p' in locals() else None, 'completed': completed, 'encoded': encoded, 'traceback': traceback.format_exc()})
        raise
    atomic(ROOT / 'results/worker.done.json', {'at_utc': now(), 'completed': completed, 'total': len(pairs),
        'encoded': encoded, 'required_encodings': sum(p['available'] for p in pairs), 'seconds': time.monotonic() - started,
        'budget_stopped': False, 'fingerprint': fp})


if __name__ == '__main__':
    main()
