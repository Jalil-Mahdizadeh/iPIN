"""Frozen native TRAIN logits; separate process avoids historical common.py collisions."""
import argparse
import sys
import time
import numpy as np
from study import ROOT, PROJECT, read, atomic, save_npz, now, sha, verify_freeze, remaining
sys.path.insert(0, str(PROJECT / 'benchmark-v1/scripts'))
from common import configure, load_model, predict
from frozen_data import PairData


class SequenceData(PairData):
    def __init__(self, rows, proteins, vocab):
        self.rows = np.asarray(rows, dtype=np.int64)
        ids = set(self.rows[:, :2].ravel().tolist())
        self.sequences = {i: np.array([vocab[c] for c in proteins[str(i)]['sequence']], dtype=np.int64) for i in ids}
        self.lengths = np.array([len(self.sequences[int(a)]) + len(self.sequences[int(b)]) + 3 for a,b in self.rows[:,:2]])

    def sequence(self, pid):
        return self.sequences[int(pid)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--qualify', action='store_true'); args = ap.parse_args()
    started = time.monotonic(); fp = None if args.qualify else verify_freeze()
    proteins = read(ROOT / 'data/proteins.json')
    vocab = {c: i for i,c in enumerate((PROJECT / 'retrain-v1/assets/esm2/vocab.txt').read_text().splitlines())}
    device = configure(0); model = load_model('native-bernett', device)
    if args.qualify:
        checks = []
        for item in read(ROOT / 'data/native-qualification.json'):
            if any(proteins[str(p)]['split'] != 'val' for r in item['rows'] for p in r[:2]):
                raise ValueError('Non-VAL parity input')
            d = SequenceData(item['rows'], proteins, vocab)
            t = time.monotonic(); z = predict(model, d, list(range(len(d))), device)
            np.testing.assert_allclose(z[:, 2:], item['orientations'], atol=2e-4, rtol=2e-4)
            checks.append({'length': item['length'], 'pairs': len(d), 'seconds': time.monotonic()-t,
                'maximum_logit_error': float(np.max(np.abs(z[:,2:]-item['orientations'])))})
        atomic(ROOT / 'qualification/native.json', {'at_utc': now(), 'passed': True, 'checks': checks,
            'seconds': time.monotonic()-started, 'code_sha256': sha(__file__), 'test_accessed': False})
        print({'native_qualification': checks}, flush=True)
        return
    sample, meta = read(ROOT / 'data/sample.json'), read(ROOT / 'data/source_metadata.json')
    ids = [i for i,(r,m) in enumerate(zip(sample,meta)) if r['split']=='train' and m['available']]
    d = SequenceData([[sample[i]['a'],sample[i]['b'],sample[i]['label'],sample[i]['row']] for i in ids], proteins, vocab)
    order = sorted(range(len(d)), key=lambda i: (d.lengths[i], sample[ids[i]]['uid']))
    z = np.full((len(d), 2), np.nan)
    for batch in d.microbatches(order, 16384, 8):
        remaining(1800)
        result = predict(model, d, batch, device)
        for i, label, ab, ba in result:
            if label != d.rows[int(i),2]:
                raise ValueError('Native label/row mismatch')
            z[int(i)] = ab, ba
    if not np.isfinite(z).all() or len(ids) != 1825:
        raise ValueError('Incomplete native TRAIN calibration scores')
    save_npz(ROOT / 'cache/native_train.npz', indices=np.array(ids), orientations=z, scores=z.mean(1),
        uids=np.array([sample[i]['uid'] for i in ids]))
    atomic(ROOT / 'cache/native_train.json', {'at_utc': now(), 'fingerprint': fp, 'sha256': sha(ROOT / 'cache/native_train.npz'),
        'pairs': len(ids), 'seconds': time.monotonic()-started, 'test_accessed': False})
    print({'native_train_scored': len(ids), 'seconds': time.monotonic()-started}, flush=True)


if __name__ == '__main__':
    main()
