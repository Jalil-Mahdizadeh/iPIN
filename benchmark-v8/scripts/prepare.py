"""Verify immutable sources, identities and fusion; prepare sequence controls."""
from collections import Counter
import hashlib
import numpy as np
from scipy import sparse
from io_v8 import ROOT, INPUT, arrays, guard, read, save, sha, write, now


def sequence_controls(proteins, a, b):
    n = max(map(int, proteins)) + 1
    aac = np.zeros((n, 26), float)
    lengths = np.zeros(n, dtype=np.int64)
    rr, cc, vv = [], [], []
    for pid in sorted(set(a.tolist()) | set(b.tolist())):
        seq = proteins[str(pid)]['sequence']
        token = np.frombuffer(seq.encode('ascii'), dtype=np.uint8).astype(np.int32) - 65
        if np.any((token < 0) | (token >= 26)):
            raise ValueError('Nonletter sequence')
        lengths[pid] = len(seq)
        counts = np.bincount(token, minlength=26).astype(float)
        aac[pid] = counts / np.linalg.norm(counts)
        kmers = token[:-2] * 26**2 + token[1:-1] * 26 + token[2:]
        keys, values = np.unique(kmers, return_counts=True)
        if len(values):
            values = values / np.linalg.norm(values)
            rr.extend([pid] * len(keys)); cc.extend(keys.tolist()); vv.extend(values.tolist())
    matrix = sparse.csr_matrix((vv, (rr, cc)), shape=(n, 26**3))
    kmer = np.zeros(len(a))
    for start in range(0, len(a), 1024):
        stop = min(start + 1024, len(a))
        kmer[start:stop] = np.asarray(matrix[a[start:stop]].multiply(matrix[b[start:stop]]).sum(1)).ravel()
    return np.column_stack([(aac[a] * aac[b]).sum(1), kmer,
                            np.minimum(lengths[a], lengths[b]) / np.maximum(lengths[a], lengths[b])])


def main():
    freeze = guard()
    complete = read(INPUT / 'vx_complete.json')
    if complete['verified'] is not True:
        raise RuntimeError('Parent run is not verified')
    for original, alias in [('val_predictions.npz', 'vx_predictions.npz'),
                             ('val_evidence.npz', 'vx_evidence.npz'), ('heads.json', 'vx_heads.json')]:
        if complete['result_hashes'][original] != sha(INPUT / alias):
            raise RuntimeError('Parent completed result changed: ' + original)
    parent = read(INPUT / 'vx_freeze.json')
    for alias in ['vx_pairs.json', 'vx_proteins.json', 'vx_groups.json', 'vx_catalog.json']:
        record = freeze['inputs'][alias]
        if parent['files'][record['project_relative']] != record['sha256']:
            raise RuntimeError('Parent frozen input changed: ' + alias)
    original = read(INPUT / 'pilot_freeze.json')
    cached = freeze['inputs']['test_sequence_hashes.json']
    expected = original['files'].get(cached['project_relative'])
    # The original pilot freeze uses project-relative paths from before relocation.
    if expected is None:
        expected = original['files'].get(cached['project_relative'].replace('pilot-vx-series/', ''))
    if expected != cached['sha256']:
        raise RuntimeError('Cached TEST identity exclusion hash does not match the original freeze')
    source = read(INPUT / 'assay_source_freeze.json')
    for alias, rel in [('assay_candidates.npz', 'data/candidates.npz'),
                       ('assay_truth.npz', 'private/truth.npz'),
                       ('assay_exposure.npz', 'private/exposure.npz'),
                       ('assay_sequences.json', 'data/sequences.json')]:
        item = next(x for x in source['files'] if x['path'] == rel)
        if sha(INPUT / alias) != item['sha256']:
            raise RuntimeError('Curated assay input changed: ' + rel)
    rows = read(INPUT / 'vx_pairs.json')
    proteins = read(INPUT / 'vx_proteins.json')
    groups = read(INPUT / 'vx_groups.json')
    for p in proteins.values():
        if hashlib.sha256(p['sequence'].encode()).hexdigest() != p['sha256']:
            raise RuntimeError('Sequence identity mismatch')
    val = [r for r in rows if r['split'] == 'val']
    pred, evidence = arrays(INPUT / 'vx_predictions.npz'), arrays(INPUT / 'vx_evidence.npz')
    heads = read(INPUT / 'vx_heads.json')
    np.testing.assert_array_equal(pred['uid'], [r['uid'] for r in val])
    np.testing.assert_array_equal(pred['y'], [r['label'] for r in val])
    errors = {}
    for arm in ['true', 'quality']:
        expected = pred['baseline'].copy()
        active = pred['gate'] > 0
        expected[active] += heads[arm]['alpha'] * pred['gate'][active] * evidence[arm][active]
        np.testing.assert_allclose(expected, pred[arm], atol=1e-12, rtol=0)
        np.testing.assert_array_equal(pred[arm][~active], pred['baseline'][~active])
        errors[arm] = float(np.max(np.abs(expected - pred[arm])))
    if not all(np.isfinite(pred[k]).all() for k in ['baseline', 'true', 'quality', 'gate']):
        raise RuntimeError('Incomplete native or evidence score coverage')
    a = np.array([r['a'] for r in val]); b = np.array([r['b'] for r in val])
    component = np.full(max(map(int, proteins)) + 1, -1, dtype=np.int64)
    for pid, c in groups['component'].items():
        component[int(pid)] = int(c)
    if np.any(component[a] < 0) or np.any(component[b] < 0):
        raise RuntimeError('Missing validation component')
    for row in val:
        left, right = groups['assignments'][str(row['a'])], groups['assignments'][str(row['b'])]
        if row['role'] != (left if left == right else 'crossing'):
            raise RuntimeError('Changed calibration/assessment assignment')
    controls = sequence_controls(proteins, a, b)
    scores = np.column_stack([pred['baseline'], pred['true'], pred['quality'], controls])
    save(ROOT / 'data/val.npz', a=a, b=b, y=pred['y'].astype(bool), uid=pred['uid'],
         role=np.array([r['role'] for r in val]), pilot=np.array([r['pilot_row'] for r in val]),
         gate=pred['gate'], scores=scores, component=component)
    write(ROOT / 'provenance/preparation.json', {
        'at_utc': now(), 'source_freezes_verified': True, 'train_rows': len(rows) - len(val),
        'val_rows': len(val), 'sequence_hashes_verified': len(proteins), 'fusion_max_errors': errors,
        'native_fallback_exact': True, 'all_component_assignments_replayed': True,
        'val_sha256': sha(ROOT / 'data/val.npz'), 'new_models_fitted': 0,
        'new_encoder_calls': 0, 'test_files_opened': False,
        'cached_test_sequence_hash_list_reused': True,
        'controls': 'A-Z amino-acid counts and ordered 3-mer counts, L2 cosine; min/max length ratio; no fitting'})
    print({'preparation_complete': True, 'val_rows': len(val), 'fusion_max_errors': errors}, flush=True)


if __name__ == '__main__':
    main()
