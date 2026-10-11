"""Fill only missing original VAL baseline rows; qualify against exact old batches."""
import sys
import time
import numpy as np
from shared import ROOT, PROJECT, read, atomic, save_npz, now, sha

# Native helpers must be imported in this separate process: the historical MSA
# implementation also has a module named common, with a different interface.
sys.path.insert(0, str(PROJECT / 'benchmark-v1/scripts'))
from common import configure, load_model, predict
from frozen_data import PairData


class SequenceData(PairData):
    def __init__(self, rows, proteins, vocab):
        self.rows = np.asarray(rows, dtype=np.int64)
        self.sequences = {int(i): np.asarray([vocab[a] for a in p['sequence']], dtype=np.int64)
                          for i, p in proteins.items() if p['split'] == 'val'}
        self.lengths = np.asarray([len(self.sequences[int(a)]) + len(self.sequences[int(b)]) + 3 for a, b in self.rows[:, :2]])

    def sequence(self, pid):
        return self.sequences[int(pid)]


def main():
    start = time.monotonic()
    if (ROOT / 'data/baseline.npz').exists():
        raise RuntimeError('Complete baseline already exists')
    reuse = read(ROOT / 'provenance/baseline-reuse.json'); missing = read(ROOT / 'data/baseline-missing.json')
    with np.load(ROOT / 'data/baseline-partial.npz', allow_pickle=False) as f:
        scores = f['orientations'].copy()
    assert np.flatnonzero(~np.isfinite(scores).all(1)).tolist() == [r['row'] for r in missing]
    checks = []
    if missing:
        proteins = read(ROOT / 'data/proteins.json')
        vocab = {s: i for i, s in enumerate((PROJECT / 'retrain-v1/assets/esm2/vocab.txt').read_text().splitlines())}
        source_rows = np.load(PROJECT / 'benchmark-v1/data/val.npy', allow_pickle=False)
        data = SequenceData(source_rows, proteins, vocab)
        device = configure(0); model = load_model('native-bernett', device)
        # Recreate original microbatches, including padding/order, for cached scores.
        selected = []
        rank_zero = [s for s in reuse['sources'] if '/rank-00-' in s['path']]
        for s in rank_zero:
            p = PROJECT / s['path']; assert sha(p) == s['sha256']
            with np.load(p, allow_pickle=False) as f:
                old = f['predictions']
            ids = old[:, 0].astype(int)
            expected = {int(v[0]): v for v in old}
            for batch in data.microbatches(ids, 16384, 8):
                length = max(int(data.lengths[i]) for i in batch)
                if (not selected and length >= 450) or (len(selected) == 1 and length >= 650) or (len(selected) == 2 and length >= 1024):
                    actual = predict(model, data, batch, device)
                    target = np.asarray([expected[i] for i in batch])
                    np.testing.assert_array_equal(actual[:, :2], target[:, :2])
                    np.testing.assert_allclose(actual[:, 2:], target[:, 2:], atol=2e-4, rtol=2e-4)
                    selected.append(batch)
                    checks.append({'length': length, 'pairs': len(batch), 'maximum_logit_difference': float(np.abs(actual[:, 2:] - target[:, 2:]).max())})
                if len(selected) == 3:
                    break
            if len(selected) == 3:
                break
        if len(selected) != 3:
            raise RuntimeError('Native baseline qualification coverage insufficient')
        new_data = SequenceData([[r['a'], r['b'], r['label'], r['row']] for r in missing], proteins, vocab)
        for batch in new_data.microbatches(list(range(len(missing))), 16384, 8):
            result = predict(model, new_data, batch, device)
            for i, label, ab, ba in result:
                row = missing[int(i)]; assert int(label) == row['label']
                scores[row['row']] = ab, ba
    if scores.shape != (59260, 2) or not np.isfinite(scores).all():
        raise ValueError('Full original VAL baseline is incomplete')
    save_npz(ROOT / 'data/baseline.npz', orientations=scores, baseline=scores.mean(1))
    atomic(ROOT / 'qualification/baseline.json', {'at_utc': now(), 'passed': True, 'rows': len(scores),
        'reused': reuse['rows_reused'], 'new_original_rows': [r['row'] for r in missing], 'checks': checks,
        'baseline_sha256': sha(ROOT / 'data/baseline.npz'), 'code_sha256': sha(__file__),
        'test_accessed': False, 'train_logits_generated': False, 'seconds': time.monotonic() - start})
    print({'baseline_complete': True, 'new_rows': len(missing), 'checks': checks}, flush=True)


if __name__ == '__main__':
    main()
