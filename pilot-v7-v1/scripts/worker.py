"""Bounded contact extraction; exact Vx global features checked for every encoded pair."""
import fcntl
import time
import traceback
import numpy as np
from study import ROOT, config, read, atomic, save_npz, sha, now, remaining, verify_freeze, load_record, save_record, owned_bytes
from inputs import Inputs
from contact_features import all_views


def extract(i, data, encoder, save_maps=False):
    started = time.monotonic(); row, old = data.sample[i], data.meta[i]
    a,b = sorted([row['a'],row['b']])
    value = {k: old[k] for k in ['uid','a','b','available','gate','reason']}
    value['features'] = {arm: [0.] * 6 for arm in config()['arms']}
    if not old['available']:
        if old['gate'] != 0:
            raise ValueError('Unavailable original pair has nonzero gate')
        value['seconds'] = time.monotonic()-started
        return value, None
    tokens, null, boundary = data.materialize(i)
    va,vb = tokens[0,:boundary] != 26, tokens[0,boundary:] != 26
    maps = {'valid_a':va, 'valid_b':vb}; value['diagnostics'] = {}; value['global_error'] = {}
    for arm,t in [('true',tokens),('shuffled',null)]:
        probability, original, asymmetry = encoder.symmetric_contacts(t,boundary)
        error = float(np.max(np.abs(original-data.arrays[arm][i])))
        if error > config()['encoder']['global_feature_atol']:
            raise ValueError(f'Original Vx feature mismatch {row["uid"]}:{arm}:{error}')
        features, details = all_views(probability,va,vb,row['uid'],a,b)
        value['features'][arm] = features['standard'].tolist()
        value['features']['coordinate_'+arm] = features['coordinate'].tolist()
        value['features']['boundary_'+arm] = features['boundary'].tolist()
        value['diagnostics'][arm] = {**details, 'orientation_mean_absolute_difference':asymmetry}
        value['global_error'][arm] = error
        maps[arm] = probability
    value['tokens_sha256'] = data.diag[i]['tokens_sha256']
    value['null_tokens_sha256'] = data.diag[i]['null_tokens_sha256']
    if save_maps:
        p = ROOT / 'cache/maps' / (row['uid']+'.npz')
        if p.exists():
            raise RuntimeError('Uncommitted map already exists; manual review required')
        save_npz(p, **maps)
        value.update({'map_path':str(p.relative_to(ROOT)), 'map_sha256':sha(p)})
    value['seconds'] = time.monotonic()-started
    return value, maps


def main():
    started = time.monotonic(); cfg = config(); fp = verify_freeze()
    lock = (ROOT/'features/worker.lock').open('a+')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    data = Inputs(); encoder = None; completed = 0; current = None
    try:
        for i,row in enumerate(data.sample):
            current = row['uid']; remaining(cfg['budgets']['analysis_reserve_seconds'])
            value = load_record(current,fp,maps=True)
            if value is not None:
                raise RuntimeError('Unexpected preexisting production feature; no automatic resume')
            if data.meta[i]['available'] and encoder is None:
                from contact_encoder import ContactEncoder
                encoder = ContactEncoder()
            value,_ = extract(i,data,encoder,save_maps=True)
            value['at_utc'] = now(); save_record(value,fp); completed += 1
            if completed % 25 == 0:
                if owned_bytes() >= cfg['budgets']['additional_bytes_ceiling']:
                    raise RuntimeError('V7 storage ceiling exceeded')
                report = {'at_utc':now(),'completed':completed,'total':len(data.sample),
                    'seconds':time.monotonic()-started,'fingerprint':fp,'uid':current}
                atomic(ROOT/'results/worker.json',report); print(report,flush=True)
    except BaseException:
        atomic(ROOT/'results/worker-error.json',{'at_utc':now(),'uid':current,'completed':completed,
            'fingerprint':fp,'traceback':traceback.format_exc()})
        raise
    atomic(ROOT/'results/worker.done.json',{'at_utc':now(),'completed':completed,'total':len(data.sample),
        'seconds':time.monotonic()-started,'fingerprint':fp})


if __name__ == '__main__':
    main()
