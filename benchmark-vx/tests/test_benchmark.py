"""Inference-only invariants and complete-cohort integration without TEST outcomes."""
import copy
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import shared
import chunks
import analyze
from collect import collect
from predict import score, prohibit_training
from runtime import topology
from verify import independent_evidence
from msa import cached_monomer


def heads():
    result = {}
    for arm, dim in shared.ARMS.items():
        coef = np.zeros(32); coef[[0, 2, 4]] = [1, -2, .5]
        result[arm] = {'standard_mean': np.ones(dim).tolist(), 'standard_scale': np.full(dim, 2.).tolist(),
            'pca_mean': np.full(dim, .25).tolist(), 'pca_components': np.eye(32, dim).tolist(),
            'coef': coef[None, :].tolist(), 'intercept': [.3], 'classes': [0, 1], 'alpha': 1,
            'fit_rows': 74576}
    return result


def fixture(root, n=160):
    (root / 'data').mkdir(); (root / 'provenance').mkdir(); (root / 'results').mkdir()
    rng = np.random.default_rng(731); shards = [list(range(i, n, 4)) for i in range(4)]
    rows = [{'uid': f'test-{i:06d}', 'index': i, 'row': i, 'split': 'test', 'a': i % 31,
             'b': 32 + (i * 7) % 29, 'length': 300 + i % 5} for i in range(n)]
    y = np.arange(n) % 2; base = rng.normal(size=n); available = np.arange(n) % 3 != 0
    gate = available.astype(float) * .8
    x = {'true': rng.normal(size=(n, 128)), 'quality': rng.normal(size=(n, 68)), 'gate': gate, 'available': available}
    x['true'][~available] = 0
    devices = [{'rank': i, 'local_rank': i, 'world': 4, 'host': 'node', 'gpu_uuid': f'gpu-{i}',
                'gpu_name': 'NVIDIA GH200', 'job_id': 'unit', 'fingerprint': 'fp'} for i in range(4)]
    for rank, ids in enumerate(shards):
        shared.atomic(root / 'results' / f'worker-{rank}.done.json', {**devices[rank], 'rows': len(ids),
            'test_authorized': True, 'shuffle_encoded': False, 'heads_fitted': False})
        for number, begin in enumerate(range(0, len(ids), 16)):
            part = [rows[i] for i in ids[begin:begin + 16]]; indices = [r['index'] for r in part]; records = []
            for row in part:
                i = row['index']; records.append({'uid': row['uid'], 'index': i, 'a': row['a'], 'b': row['b'],
                    'length': row['length'], 'available': bool(available[i]), 'gate': float(gate[i]),
                    'reason': 'available' if available[i] else 'missing_exact_query',
                    'depth': 16 if available[i] else 0, 'token_sha256': 'tokens' if available[i] else None,
                    'kind': 'encoded' if available[i] else 'fallback',
                    'msa': {'quality_a': .8, 'quality_b': 1., 'neff': 15} if available[i] else None,
                    'timing': {'orientation_depths': [16, 16], 'layer_indices': list(range(16)) * 2} if available[i] else None})
            arrays = {k: v[indices] for k, v in x.items()}; arrays['indices'] = np.asarray(indices)
            chunks.save(root / 'features' / f'rank-{rank:02d}' / f'chunk-{number:04d}', arrays, records, part, 'fp')
    shared.atomic(root / 'provenance/execution.json', {'job_id': 'unit', 'fingerprint': 'fp'})
    shared.atomic(root / 'provenance/topology-approved.json', {'job_id': 'unit', 'fingerprint': 'fp', 'workers': devices})
    shared.atomic(root / 'data/pairs.json', rows); shared.atomic(root / 'data/shards.json', shards)
    uids = np.asarray([r['uid'] for r in rows]); shared.save_npz(root / 'data/labels.npz', uid=uids, y=y)
    shared.save_npz(root / 'data/baseline.npz', uid=uids, baseline=base, orientations=np.column_stack([base, base]))
    return rows, shards, x, base, y, devices


class BenchmarkTests(unittest.TestCase):
    def test_frozen_arithmetic_and_native_fallback(self):
        h = heads(); rng = np.random.default_rng(7)
        f = {k: rng.normal(size=(5, d)) for k, d in shared.ARMS.items()}
        f.update(gate=np.array([0., .2, 1., .8, 0.]), available=np.array([False, True, True, True, False]))
        baseline = np.arange(5, dtype=float); snapshot = copy.deepcopy(h)
        prohibit_training(); z, evidence = score(f, baseline, h, {'true': 1, 'quality': 1})
        for arm in shared.ARMS:
            v = (f[arm] - 1) / 2 - .25
            oracle = v[:, 0] - 2 * v[:, 2] + .5 * v[:, 4] + .3
            np.testing.assert_allclose(evidence[arm], oracle, atol=1e-12, rtol=0)
            np.testing.assert_allclose(independent_evidence(f[arm], h[arm]), oracle, atol=1e-12, rtol=0)
            np.testing.assert_array_equal(z[arm][~f['available']], baseline[~f['available']])
        self.assertEqual(h, snapshot)
        from sklearn.linear_model import LogisticRegression
        with self.assertRaisesRegex(RuntimeError, 'forbidden'):
            LogisticRegression().fit(np.ones((4, 2)), [0, 1, 0, 1])

    def test_invalid_inputs_cannot_become_fallback(self):
        f = {'true': np.zeros((2, 128)), 'quality': np.zeros((2, 68)), 'gate': np.array([0., 1.]), 'available': np.array([False, True])}
        h = heads()
        for baseline in [np.array([np.nan, 0.]), np.array([0.])]:
            with self.assertRaises(ValueError): score(f, baseline, h, {'true': 1, 'quality': 1})
        f['true'][1, 0] = np.inf
        with self.assertRaises(ValueError): score(f, np.zeros(2), h, {'true': 1, 'quality': 1})
        f['true'][1, 0] = 0; h['true']['alpha'] = .5
        with self.assertRaises(ValueError): score(f, np.zeros(2), h, {'true': 1, 'quality': 1})

    def test_context_exclusion_removes_both_splits(self):
        forbidden = {hashlib.sha256(s.encode()).hexdigest() for s in ['ARNDCEQG', 'ARCEQG']}
        mono = cached_monomer(iter([('train genus:family', 'ARNDCEQG'),
            ('val genus:family', 'AR--CEQG'), ('allowed genus:family', 'VRNDCEQG')]), 'ARNDCEQG', '*' * 8, forbidden)
        self.assertEqual([r[0] for r in mono['rows']], ['allowed'])
        self.assertEqual(mono['stats']['excluded_human_context'], 2)

    def test_topology_rejects_shared_devices_and_multiple_nodes(self):
        records = [{'rank': i, 'local_rank': i, 'world': 4, 'host': 'node', 'gpu_uuid': str(i),
                    'gpu_name': 'NVIDIA GH200', 'job_id': 'j', 'fingerprint': 'f'} for i in range(4)]
        topology(records, 'j', 'f')
        for key, value in [('gpu_uuid', '0'), ('host', 'other'), ('world', 8), ('job_id', 'stale'), ('local_rank', 0)]:
            changed = copy.deepcopy(records); changed[1][key] = value
            with self.assertRaises(ValueError): topology(changed, 'j', 'f')

    def test_complete_cohort_and_corrupt_chunk(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows, shards, expected, base, y, _ = fixture(root)
            features, records, workers = collect(root, rows, shards, 'fp', 16)
            for k, v in expected.items(): np.testing.assert_array_equal(features[k], v)
            self.assertEqual(len(workers), 4)
            wrong = copy.deepcopy(shards); wrong[0][0] = wrong[1][0]
            with self.assertRaises(ValueError): collect(root, rows, wrong, 'fp', 16)
            p = root / 'features/rank-00/chunk-0000.npz'; p.write_bytes(b'corruption')
            with self.assertRaisesRegex(ValueError, 'Corrupt'): collect(root, rows, shards, 'fp', 16)

    def test_missing_worker_or_labels_block_evaluation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows, shards, *_ = fixture(root)
            bad = copy.deepcopy(rows); bad[0]['label'] = 1
            with self.assertRaises(ValueError): collect(root, bad, shards, 'fp', 16)
            (root / 'results/worker-3.done.json').unlink()
            cfg = {'chunk_rows': 16}
            with mock.patch.multiple(analyze, ROOT=root, config=lambda: cfg, verify_freeze=lambda: 'fp', source_contract=heads):
                with self.assertRaisesRegex(ValueError, 'Worker not completed'): analyze.main()
            self.assertFalse((root / 'results/evaluation-start.json').exists())
            self.assertFalse((root / 'results/test_predictions.npz').exists())

    def test_end_to_end_frozen_scoring_and_second_attempt_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); rows, shards, x, base, y, _ = fixture(root)
            cfg = {'test_pairs': len(rows), 'chunk_rows': 16, 'heads_sha256': 'synthetic',
                'fixed_alphas': {'true': 1, 'quality': 1}, 'bootstrap_replicates': 40, 'seed': 731,
                'predictive_criteria': {'minimum_delta_ap': .01, 'minimum_ci_lower': 0, 'maximum_auroc_loss': .005}}
            with mock.patch.multiple(analyze, ROOT=root, config=lambda: cfg, verify_freeze=lambda: 'fp', source_contract=heads):
                analyze.main()
                with self.assertRaisesRegex(RuntimeError, 'already attempted'): analyze.main()
            with np.load(root / 'results/test_predictions.npz') as z:
                for arm in shared.ARMS:
                    oracle = base + x['gate'] * independent_evidence(x[arm], heads()[arm])
                    np.testing.assert_allclose(z[arm], oracle, rtol=0, atol=1e-12)
            self.assertFalse((root / 'results/fit-start.json').exists())
            self.assertFalse((root / 'results/heads.json').exists())
            self.assertFalse(shared.read(root / 'results/metrics.json')['heads_fitted'])


if __name__ == '__main__':
    unittest.main()
