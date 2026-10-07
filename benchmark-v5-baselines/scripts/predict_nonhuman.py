"""Reuse v5 fits, fit Bernett once, and freeze scores without loading species labels."""
import csv, gzip, math, joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits
from nonhuman_common import *
from fit_predict import features, interolog_scores

def parse_hits(meta, train_ids, path):
    count = len(meta['sequence']); mask = np.zeros(count, bool); mask[train_ids] = True
    hitlists = [{} for _ in range(count)]; raw = 0; precision = {'integer_diagnostics_rows': 0,
        'integer_below_requested_identity': 0, 'integer_below_requested_bilateral_coverage': 0,
        'maximum_absolute_weight_rounding_difference': 0.0}
    with Path(path).open() as inp:
        for row in csv.reader(inp, delimiter='\t'):
            assert len(row) in [9, 15]
            q, t = int(row[0][1:]), int(row[1][1:]); fident, bits, e, qc, tc = map(float, row[2:7])
            assert mask[t] and int(row[7]) == len(meta['sequence'][q]) and int(row[8]) == len(meta['sequence'][t])
            assert .249 <= fident <= 1 and .499 <= qc <= 1 and .499 <= tc <= 1 and e <= 1.01e-5
            weight = fident * math.sqrt(qc*tc)
            item = (t, weight, bits, e, fident, qc, tc, q == t)
            if t not in hitlists[q] or bits > hitlists[q][t][2]: hitlists[q][t] = item
            if len(row) == 15:
                ni, alen, qs, qe, ts, te = map(int, row[9:]); qlen, tlen = int(row[7]), int(row[8])
                assert 0 <= ni <= alen and 1 <= qs <= qe <= qlen and 1 <= ts <= te <= tlen
                precise = (ni/alen) * math.sqrt((qe-qs+1)/qlen * (te-ts+1)/tlen)
                precision['integer_diagnostics_rows'] += 1
                precision['integer_below_requested_identity'] += 4*ni < alen
                precision['integer_below_requested_bilateral_coverage'] += 2*(qe-qs+1) < qlen or 2*(te-ts+1) < tlen
                precision['maximum_absolute_weight_rounding_difference'] = max(precision['maximum_absolute_weight_rounding_difference'], abs(weight-precise))
            raw += 1
    k = CONFIG['search']['top_k']; ids = np.full((count, k), -1, np.int64); weights = np.zeros((count,k), np.float64)
    missing_exact = 0; retained = []
    for q, found in enumerate(hitlists):
        if mask[q]:
            missing_exact += q not in found; bits = found[q][2] if q in found else -1.
            found[q] = (q, 1., bits, 0., 1., 1., 1., True)
        ranked = sorted(found.values(), key=lambda r:(not r[7],-r[2],-r[1],meta['sha256'][r[0]]))[:k]
        for j, item in enumerate(ranked):
            ids[q,j], weights[q,j] = item[:2]; retained.append((q,j+1,*item))
    assert np.array_equal(ids[train_ids,0], train_ids) and np.all(weights[train_ids,0] == 1)
    return ids, weights, retained, {'raw_hits': raw, 'missing_exact_inserted': int(missing_exact), **precision}

def main():
    start = time.monotonic(); protocol = freeze_extension(); original = verify_legacy_models()
    marker = ROOT / 'provenance/nonhuman/predictions-frozen.json'
    identity = {'protocol': protocol, 'prepared_sha256': sha(ROOT / 'provenance/nonhuman/prepared.json'),
        'original_predictions_sha256': sha(ROOT / 'provenance/predictions-frozen.json'),
        'script_sha256': sha(__file__), 'searches': {r: sha(ROOT / 'provenance/nonhuman' / ('search-'+r+'.json')) for r in TRAINING}}
    if marker.exists():
        old = read(marker); assert old['identity'] == identity
        for item in old['artifacts']: verify(item)
        print('Nonhuman frozen predictions verified; no refitting.', flush=True); return
    meta = read(NDATA / 'sequences.json'); count = len(meta['sequence'])
    legacy_meta = read(ROOT / 'data/sequences.json'); old_count = len(legacy_meta['sequence'])
    legacy = np.load(ROOT / 'models/protein-values.npz')
    # Full replay qualifies the parser/tie breaking before the new queries are scored.
    legacy_search = read(ROOT / 'provenance/search.json')
    assert sha(ROOT / 'work/alignments.tsv') == legacy_search['output_sha256']
    replay_i, replay_w, _, _ = parse_hits(legacy_meta, np.asarray(legacy_meta['train_indices']), ROOT / 'work/alignments.tsv')
    assert np.array_equal(replay_i, legacy['homolog_indices']) and np.array_equal(replay_w, legacy['homolog_weights'])
    x = features(meta['sequence']); refs = {}; artifacts = []
    for ref in TRAINING:
        fit_start = time.monotonic(); train = np.load(NDATA / (ref + '-train.npz'))
        p, y = train['pairs'], train['labels']; ids = np.asarray(meta['train_indices'][ref], np.int64)
        assert np.array_equal(np.unique(p), ids)
        positive, negative = degrees(p, y, count); ratio = np.log((positive+1.)/(negative+1.))
        # Independent source-row count reconstruction, including self-pair convention.
        independent = np.zeros((2,count), np.int64)
        for (a,b), label in zip(p,y):
            for protein in set((a,b)): independent[int(label),protein] += 1
        assert np.array_equal(independent[1],positive) and np.array_equal(independent[0],negative)
        mask = np.zeros(count,bool); mask[ids] = True
        with threadpool_limits(16, user_api='openmp'), threadpool_limits(1, user_api='blas'):
            if ref == 'v5':
                assert np.array_equal(positive[:old_count], legacy['positive_degree'])
                assert np.array_equal(negative[:old_count], legacy['negative_degree'])
                assert np.array_equal(ratio[:old_count], legacy['exact_degree'])
                model_file = ROOT / 'models/sequence-propensity.joblib'; model = joblib.load(model_file)
                sequence_values = model.predict(x)
                assert np.array_equal(sequence_values[:old_count], legacy['sequence_propensity'])
                fit_seconds = 0.0
            else:
                model = HistGradientBoostingRegressor(**CONFIG['regressor']); t0 = time.monotonic()
                model.fit(x[ids], ratio[ids]); fit_seconds = time.monotonic()-t0
                assert model.n_iter_ == 300
                sequence_values = model.predict(x)
                model_file = NMODELS / 'bernett-sequence-propensity.joblib'; joblib.dump(model, model_file, compress=3)
        assert model.get_params() == original['regressor_parameters']
        pairs = np.sort(p[y == 1], axis=1); graph = np.unique(pairs[:,0]*count+pairs[:,1])
        if ref == 'v5':
            original_graph = np.load(ROOT / 'models/train-positive-graph.npy')
            recoded = (original_graph//old_count)*count + original_graph%old_count
            assert np.array_equal(graph,recoded)
        search = read(ROOT / 'provenance/nonhuman' / ('search-'+ref+'.json'))
        alignment = NWORK / ('alignments-'+ref+'.tsv'); assert sha(alignment) == search['output_sha256']
        hits, weights, retained, hitinfo = parse_hits(meta, ids, alignment)
        with gzip.open(NMODELS / (ref+'-homologs.csv.gz'),'wt',newline='') as handle:
            writer = csv.writer(handle); writer.writerow(['query_index','rank','training_index','weight','bits','evalue','identity','qcov','tcov','exact'])
            writer.writerows(retained)
        fallback = float(ratio[ids].mean()); denom = weights.sum(1); homology = np.full(count, fallback)
        has = denom > 0
        homology[has] = (ratio[np.maximum(hits[has],0)]*weights[has]).sum(1)/denom[has]
        homology[ids] = ratio[ids]
        npz(NMODELS / (ref+'-protein-values.npz'), positive_degree=positive, negative_degree=negative,
            exact_degree=ratio, homology_degree=homology, sequence_propensity=sequence_values, train_mask=mask,
            homolog_indices=hits, homolog_weights=weights)
        np.save(NMODELS / (ref+'-train-positive-graph.npy'),graph)
        graph_set = set(graph.tolist()); stage = {}
        for species in SPECIES:
            # Only the pair array is loaded; held-out labels never enter fitting or prediction.
            q = np.load(NDATA / (species+'-rows.npz'))['pairs']
            values = {'exact-degree': ratio[q].sum(1), 'homology-degree': homology[q].sum(1),
                      'sequence-propensity': sequence_values[q].sum(1)}
            values['interolog'], support = interolog_scores(q, hits, weights, graph, count)
            backward, bs = interolog_scores(q[:,::-1],hits,weights,graph,count)
            assert np.array_equal(backward,values['interolog']) and np.array_equal(bs,support)
            for name, v in [('exact-degree',ratio),('homology-degree',homology),('sequence-propensity',sequence_values)]:
                assert np.array_equal(values[name],v[q[:,::-1]].sum(1))
            sample = np.random.default_rng(47).choice(len(q),100,replace=False)
            for row in sample:
                a,b = q[row]; scalar = 0.; ns = 0
                for i,u in enumerate(hits[a]):
                    for j,v in enumerate(hits[b]):
                        if u >= 0 and v >= 0 and min(u,v)*count+max(u,v) in graph_set:
                            scalar = max(scalar, weights[a,i]*weights[b,j]); ns += 1
                assert scalar == values['interolog'][row] and ns == support[row]
            assert all(np.isfinite(v).all() and len(v) == len(q) for v in values.values())
            path = ROOT / 'results' / (species+'-'+ref+'-baseline-scores.npz')
            npz(path, **values, interolog_support_count=support); artifacts.append(record(path))
            stage[species] = {'rows':len(q),'all_scores_finite_and_symmetric':True,'scalar_interolog_rows_checked':100}
        target = ratio[ids]
        refs[ref] = {'train_rows':len(p),'positive_rows':int(y.sum()),'negative_rows':int((1-y).sum()),
            'train_sequences':len(ids),'unique_positive_edges':len(graph),'fitted_regressor':record(model_file),
            'new_regressor_fit_seconds':fit_seconds,'reused_existing_model':ref=='v5',
            'degree_target':{'equal_degrees':int((positive[ids]==negative[ids]).sum()),'std':float(target.std()),
                            'min':float(target.min()),'max':float(target.max()),'unique_values':len(np.unique(target))},
            'no_hit_propensity':fallback,'homologs':hitinfo,'scored':stage,'stage_seconds':time.monotonic()-fit_start}
        print('Frozen predictor',ref,'fit seconds',fit_seconds,'degree target',refs[ref]['degree_target'],flush=True)
    artifacts += [record(p) for p in sorted(NMODELS.iterdir()) if p.is_file()]
    verify_legacy_models()
    atomic(marker, {'at_utc':now(),'identity':identity,'references':refs,'artifacts':artifacts,
        'checks':{'legacy_v5_parser_replay_exact':True,'legacy_v5_all_protein_predictions_exact':True,
                  'v5_degrees_and_graph_reused_and_independently_reconstructed':True,'no_species_label_arrays_loaded':True,
                  'all_scores_symmetric_and_finite':True,'scalar_interolog_verified':True,'v5_original_artifacts_unchanged':True},
        'resource_use':stage_stat(start)})
    print('Both reference predictions frozen, ready for evaluation',flush=True)

if __name__ == '__main__': main()
