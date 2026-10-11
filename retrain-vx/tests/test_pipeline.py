import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import analyze
import chunks
from collect import collect
from shared import atomic, save_npz, read
from heads import fit_head


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup); self.root = Path(self.tmp.name)
        self.rng = np.random.default_rng(9); self.fp = 'fixture'; self.rows = []
        for role, count, pilot in [('train', 400, False), ('calibration', 200, False), ('assessment', 200, False), ('assessment', 80, True), ('crossing', 200, False)]:
            for j in range(count):
                i = len(self.rows); split = 'train' if role == 'train' else 'val'
                self.rows.append({'uid': f'{split}-{i:06d}', 'index': i, 'row': i if split == 'train' else i - 400,
                    'split': split, 'a': (i % 100) * 2, 'b': (i % 100) * 2 + 1, 'label': j % 2,
                    'length': 128, 'role': role, 'pilot_row': pilot})
        self.shards = [list(range(i, len(self.rows), 8)) for i in range(8)]
        self.cfg = {'train_pairs': 400, 'dev_pairs': 680, 'chunk_rows': 32, 'seed': 20261007,
            'primary': 'assessment_new_rows', 'fusion': {'alpha_grid': [0, .1, .25, .5, 1], 'bootstrap_replicates': 20}}
        atomic(self.root / 'data/pairs.json', self.rows); atomic(self.root / 'data/shards.json', self.shards)
        save_npz(self.root / 'data/baseline.npz', baseline=self.rng.normal(size=680))
        for rank, ids in enumerate(self.shards):
            for number, begin in enumerate(range(0, len(ids), 32)):
                part = [self.rows[i] for i in ids[begin:begin + 32]]; available = np.asarray([r['index'] % 11 != 0 for r in part])
                true = self.rng.normal(size=(len(part), 128)); true[~available] = 0
                quality = self.rng.normal(size=(len(part), 68)); records = []
                for r, active in zip(part, available):
                    records.append({'uid': r['uid'], 'index': r['index'], 'a': r['a'], 'b': r['b'],
                        'reason': 'available' if active else 'missing_exact_query', 'length': 128,
                        'available': bool(active), 'gate': float(active), 'depth': 128 if active else 0,
                        'token_sha256': 'a' * 64 if active else None, 'kind': 'encoded' if active else 'fallback',
                        'timing': {'orientation_depths': [128, 128], 'layer_indices': list(range(16)) * 2} if active else None})
                arrays = {'indices': np.asarray([r['index'] for r in part]), 'true': true, 'quality': quality,
                    'gate': available.astype(float), 'available': available}
                chunks.save(self.root / 'features' / f'rank-{rank:02d}' / f'chunk-{number:04d}', arrays, records, part, self.fp)
            atomic(self.root / 'results' / f'worker-{rank}.done.json', {'fingerprint': self.fp, 'rank': rank, 'world': 8,
                'rows': len(ids), 'gpu_uuid': f'gpu-{rank}', 'host': f'node-{rank // 4}', 'test_accessed': False, 'shuffle_encoded': False})

    def analysis(self):
        stack = ExitStack(); self.addCleanup(stack.close)
        stack.enter_context(patch.object(analyze, 'ROOT', self.root))
        stack.enter_context(patch.object(analyze, 'config', lambda: self.cfg))
        stack.enter_context(patch.object(analyze, 'verify_freeze', lambda: self.fp))
        return stack.enter_context(patch.object(analyze, 'fit_head', wraps=fit_head))

    def test_complete_analysis_two_fits_and_independent_reconstruction(self):
        fitted = self.analysis(); analyze.main(); self.assertEqual(fitted.call_count, 2)
        features, _, _ = collect(self.root, self.rows, self.shards, self.fp, 32)
        heads = read(self.root / 'results/heads.json')
        with np.load(self.root / 'results/val_predictions.npz') as f:
            scores = {k: f[k] for k in f.files}
        with np.load(self.root / 'data/baseline.npz') as f:
            base = f['baseline']
        for arm, h in heads.items():
            standardized = (features[arm] - h['standard_mean']) / h['standard_scale']
            ev = ((standardized - h['pca_mean']) @ np.asarray(h['pca_components']).T @ np.asarray(h['coef']).T).ravel() + h['intercept'][0]
            active = features['available'][400:]; z = base.copy(); z[active] += h['alpha'] * features['gate'][400:][active] * ev[400:][active]
            np.testing.assert_allclose(z, scores[arm], atol=1e-12)
            np.testing.assert_array_equal(z[~active], base[~active])
        with self.assertRaises(RuntimeError):
            analyze.main()
        self.assertEqual(fitted.call_count, 2)

    def test_incomplete_or_corrupt_chunk_prevents_any_fit(self):
        fitted = self.analysis(); path = self.root / 'features/rank-00/chunk-0000.npz'
        path.write_bytes(path.read_bytes() + b'corrupt')
        with self.assertRaises(ValueError):
            analyze.main()
        fitted.assert_not_called(); self.assertFalse((self.root / 'results/fit-start.json').exists())
        path.unlink()
        with self.assertRaises(ValueError):
            analyze.main()
        fitted.assert_not_called()

    def test_duplicate_shards_and_wrong_gpu_topology_rejected(self):
        bad = [v.copy() for v in self.shards]; bad[1][0] = bad[0][0]
        with self.assertRaises(ValueError):
            collect(self.root, self.rows, bad, self.fp, 32)
        p = self.root / 'results/worker-1.done.json'; w = read(p); w['gpu_uuid'] = 'gpu-0'; atomic(p, w)
        with self.assertRaises(ValueError):
            collect(self.root, self.rows, self.shards, self.fp, 32)

    def test_extra_shuffled_arm_rejected(self):
        part = [self.rows[i] for i in self.shards[0][:32]]
        saved = chunks.load(self.root / 'features/rank-00/chunk-0000', part, self.fp)
        arrays, records = saved; arrays['shuffled'] = arrays['true'].copy()
        with self.assertRaises(ValueError):
            chunks.validate(arrays, records, part)


if __name__ == '__main__':
    unittest.main()
