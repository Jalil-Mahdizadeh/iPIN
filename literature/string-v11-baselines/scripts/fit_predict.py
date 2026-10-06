"""Fit once on human TRAIN; generate all scores without reading evaluation labels."""
import csv
import gzip
import math
import platform
import sys
import time
import joblib
import numpy as np
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score
from threadpoolctl import threadpool_limits
from common import *

AA = 'ACDEFGHIKLMNPQRSTVWY'

def features(sequences):
    code = np.full(256, 20, np.int16)
    for i, a in enumerate(AA):
        code[ord(a)] = i
    result = np.zeros((len(sequences), 422), np.float32)
    for i, sequence in enumerate(sequences):
        x = code[np.frombuffer(sequence.encode('ascii'), np.uint8)]
        valid = x < 20
        result[i, 0] = math.log(len(x))
        result[i, 1:21] = np.bincount(x[valid], minlength=20) / len(x)
        pairs = valid[:-1] & valid[1:]
        result[i, 21:421] = np.bincount((20*x[:-1]+x[1:])[pairs], minlength=400) / max(1, len(x)-1)
        result[i, 421] = 1 - valid.mean()
    assert np.isfinite(result).all()
    assert np.allclose(result[:, 1:21].sum(1) + result[:, 421], 1, atol=1e-6)
    assert np.all(result[:, 21:421].sum(1) <= 1.000001)
    return result

def homologs(meta, train_mask):
    count = len(meta['sequence'])
    hitlists = [{} for _ in range(count)]
    search = read(ROOT / 'provenance/search.json')
    path = ROOT / 'work/alignments.tsv'
    assert sha(path) == search['output_sha256']
    assert search['identity']['query_sha256'] == sha(ROOT / 'data/queries.fasta')
    assert search['identity']['target_sha256'] == sha(ROOT / 'data/human-train.fasta')
    observed = 0
    with path.open() as inp:
        for row in csv.reader(inp, delimiter='\t'):
            assert len(row) == 9
            q, t = int(row[0][1:]), int(row[1][1:])
            identity, bits, evalue, qcov, tcov = map(float, row[2:7])
            assert 0 <= q < count and 0 <= t < count and train_mask[t]
            assert int(row[7]) == len(meta['sequence'][q]) and int(row[8]) == len(meta['sequence'][t])
            assert 0.249 <= identity <= 1 and 0.499 <= qcov <= 1 and 0.499 <= tcov <= 1
            assert evalue <= 1.01e-5
            weight = identity * math.sqrt(qcov * tcov)
            item = (t, weight, bits, evalue, identity, qcov, tcov, q == t)
            if t not in hitlists[q] or bits > hitlists[q][t][2]:
                hitlists[q][t] = item
            observed += 1
    k = CONFIG['search']['top_k']
    indices, weights = np.full((count, k), -1, np.int64), np.zeros((count, k), np.float64)
    exact_inserted = 0
    with gzip.open(ROOT / 'models/homologs.csv.gz', 'wt', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['query_index', 'query_sha256', 'rank', 'train_index', 'train_sha256',
                         'weight', 'bit_score', 'evalue', 'identity', 'query_coverage', 'target_coverage', 'exact'])
        for q, hits in enumerate(hitlists):
            if train_mask[q]:
                exact_inserted += int(q not in hits)
                bits = hits[q][2] if q in hits else -1.0
                hits[q] = (q, 1.0, bits, 0.0, 1.0, 1.0, 1.0, True)
            ranked = sorted(hits.values(), key=lambda r: (not r[7], -r[2], -r[1], meta['sha256'][r[0]]))[:k]
            for j, item in enumerate(ranked):
                t, weight, bits, evalue, identity, qcov, tcov, exact = item
                indices[q, j], weights[q, j] = t, weight
                writer.writerow([q, meta['sha256'][q], j+1, t, meta['sha256'][t], repr(weight),
                                 bits, evalue, identity, qcov, tcov, int(exact)])
    assert (indices[train_mask, 0] == np.flatnonzero(train_mask)).all()
    assert (weights[train_mask, 0] == 1).all()
    return indices, weights, {'raw_hits': observed, 'exact_matches_inserted_missing_from_search': exact_inserted,
                              'query_proteins_with_hit': int((indices[:, 0] >= 0).sum()),
                              'query_proteins_without_hit': int((indices[:, 0] < 0).sum())}

def interolog_scores(pairs, indices, weights, graph, count):
    scores = np.zeros(len(pairs), np.float64)
    support = np.zeros(len(pairs), np.int16)
    for i in range(indices.shape[1]):
        a, wa = indices[pairs[:, 0], i], weights[pairs[:, 0], i]
        for j in range(indices.shape[1]):
            b, wb = indices[pairs[:, 1], j], weights[pairs[:, 1], j]
            valid = (a >= 0) & (b >= 0)
            ids = np.flatnonzero(valid)
            code = np.minimum(a[ids], b[ids]) * count + np.maximum(a[ids], b[ids])
            loc = np.searchsorted(graph, code)
            found = (loc < len(graph)) & (graph[np.minimum(loc, len(graph)-1)] == code)
            ids = ids[found]
            scores[ids] = np.maximum(scores[ids], wa[ids] * wb[ids])
            support[ids] += 1
    return scores, support

def main():
    start = time.monotonic()
    verify_protocol()
    marker = ROOT / 'provenance/predictions-frozen.json'
    script_ids = {n: sha(ROOT / 'scripts' / n) for n in ['common.py', 'fit_predict.py']}
    if marker.exists():
        previous = read(marker)
        assert previous['scripts'] == script_ids
        for item in previous['artifacts']:
            verify(item)
        print('Existing fitted model and frozen predictions verified; no refitting.', flush=True)
        return
    meta = read(ROOT / 'data/sequences.json')
    count = len(meta['sequence'])
    train = np.load(ROOT / 'data/human-train-pairs.npy', allow_pickle=False)
    ytrain = np.load(ROOT / 'data/human-train-labels.npy', allow_pickle=False)
    train_ids = np.array(meta['train_indices'], np.int64)
    assert np.array_equal(train_ids, np.unique(train))
    train_mask = np.zeros(count, bool)
    train_mask[train_ids] = True
    positive, negative = degrees(train, ytrain, count)
    r = np.log((positive+1.0)/(negative+1.0))
    assert ((positive + negative > 0) == train_mask).all()
    # Independent loop audit checks row multiplicity and single counting of self-pair endpoints.
    audit = np.zeros((2, count), np.int64)
    for (a, b), y in zip(train, ytrain):
        for q in set((a, b)):
            audit[int(y), q] += 1
    assert np.array_equal(audit[1], positive) and np.array_equal(audit[0], negative)
    xstart = time.monotonic()
    x = features(meta['sequence'])
    feature_seconds = time.monotonic() - xstart
    xcheck = features(['ACXDA'])
    assert np.isclose(xcheck[0, 1], .4) and np.isclose(xcheck[0, 421], .2)
    assert np.isclose(xcheck[0, 21 + AA.index('A')*20 + AA.index('C')], .25)
    assert np.isclose(xcheck[0, 21:421].sum(), .5)
    regressor = HistGradientBoostingRegressor(**CONFIG['regressor'])
    fstart = time.monotonic()
    with threadpool_limits(CONFIG['threads'], user_api='openmp'), threadpool_limits(1, user_api='blas'):
        regressor.fit(x[train_ids], r[train_ids])
        fitted_seconds = time.monotonic() - fstart
        pstart = time.monotonic()
        sequence_values = regressor.predict(x)
        inference_seconds = time.monotonic() - pstart
    assert regressor.n_iter_ == 300 and np.isfinite(sequence_values).all()
    joblib.dump(regressor, ROOT / 'models/sequence-propensity.joblib', compress=3)
    print({'regressor_fitted_seconds': fitted_seconds, 'sequence_prediction_seconds': inference_seconds,
           'train_proteins': len(train_ids), 'features': x.shape[1]}, flush=True)
    hstart = time.monotonic()
    hits, weights, hitinfo = homologs(meta, train_mask)
    fallback = float(r[train_ids].mean())
    sums = weights.sum(1)
    homology_values = np.full(count, fallback, np.float64)
    hit_rows = sums > 0
    homology_values[hit_rows] = ((r[np.maximum(hits[hit_rows], 0)] * weights[hit_rows]).sum(1) / sums[hit_rows])
    homology_values[train_ids] = r[train_ids]
    graph_pairs = np.sort(train[ytrain == 1], axis=1)
    graph = np.unique(graph_pairs[:, 0] * count + graph_pairs[:, 1])
    assert len(graph) > 0
    npz(ROOT / 'models/protein-values.npz', positive_degree=positive, negative_degree=negative,
        exact_degree=r, homology_degree=homology_values, sequence_propensity=sequence_values,
        train_mask=train_mask, homolog_indices=hits, homolog_weights=weights)
    np.save(ROOT / 'models/train-positive-graph.npy', graph)
    graph_set = set(graph.tolist())
    score_info = {}
    # Evaluation label files are deliberately not opened anywhere in this stage.
    for dataset in DATASETS:
        pairs = np.load(ROOT / 'data' / (dataset + '-pairs.npy'), allow_pickle=False)
        scores = {'exact-degree': r[pairs].sum(1), 'homology-degree': homology_values[pairs].sum(1),
                  'sequence-propensity': sequence_values[pairs].sum(1)}
        t0 = time.monotonic()
        scores['interolog'], support = interolog_scores(pairs, hits, weights, graph, count)
        backward, backward_support = interolog_scores(pairs[:, ::-1], hits, weights, graph, count)
        assert np.array_equal(backward, scores['interolog']) and np.array_equal(backward_support, support)
        for name, values in [('exact-degree', r), ('homology-degree', homology_values), ('sequence-propensity', sequence_values)]:
            assert np.array_equal(scores[name], values[pairs[:, ::-1]].sum(1))
        # Independent scalar edge lookup for a fixed, label-blind sample of pairs.
        sample = np.random.default_rng(47).choice(len(pairs), min(100, len(pairs)), replace=False)
        for row in sample:
            a, b = pairs[row]
            scalar = 0.0
            for i, u in enumerate(hits[a]):
                for j, v in enumerate(hits[b]):
                    if u >= 0 and v >= 0 and min(u, v)*count+max(u, v) in graph_set:
                        scalar = max(scalar, weights[a, i]*weights[b, j])
            assert scalar == scores['interolog'][row]
        assert all(np.isfinite(v).all() and len(v) == len(pairs) for v in scores.values())
        npz(ROOT / 'results' / (dataset + '-baseline-scores.npz'), **scores,
            interolog_support_count=support)
        score_info[dataset] = {'rows': len(pairs), 'scoring_and_symmetry_audit_seconds': time.monotonic()-t0}
    artifacts = [record(p) for p in sorted((ROOT / 'models').iterdir()) if p.is_file()]
    artifacts += [record(ROOT / 'results' / (s + '-baseline-scores.npz')) for s in DATASETS]
    info = {'at_utc': now(), 'protocol_sha256': sha(ROOT / 'PROTOCOL.md'), 'scripts': script_ids,
            'artifacts': artifacts, 'training_pairs': record(ROOT / 'data/human-train-pairs.npy'),
            'training_labels': record(ROOT / 'data/human-train-labels.npy'),
            'all_query_sequences': record(ROOT / 'data/sequences.json'),
            'regressor_parameters': regressor.get_params(), 'feature_count': 422,
            'train_protein_target_r2': float(r2_score(r[train_ids], sequence_values[train_ids])),
            'feature_seconds': feature_seconds, 'regressor_fit_seconds': fitted_seconds,
            'regressor_prediction_seconds': inference_seconds,
            'homolog_parse_and_all_pair_scoring_seconds': time.monotonic()-hstart,
            'homology_no_hit_value': fallback, 'homologs': hitinfo, 'unique_positive_train_edges': len(graph),
            'datasets': score_info, 'resource_use': stage_stat(start),
            'checks': {'no_evaluation_label_files_read_by_fit_predict': True, 'all_scores_finite': True,
                       'all_scores_symmetric': True, 'vectorized_interolog_matches_independent_scalar_lookup': True,
                       'train_degree_counts_match_independent_loop': True},
            'environment': {'python': sys.version, 'numpy': np.__version__, 'sklearn': sklearn.__version__,
                            'machine': platform.machine(), 'hostname': platform.node(), 'gpu_used': False}}
    atomic(marker, info)
    print({'frozen_predictions': str(marker), 'wall_seconds': time.monotonic()-start, 'homologs': hitinfo}, flush=True)

if __name__ == '__main__':
    main()
