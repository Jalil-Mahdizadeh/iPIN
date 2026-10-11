"""Independent frozen-head arithmetic, raw identity, bootstrap and source replay."""
import csv
import hashlib
import time
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from shared import ROOT, PROJECT, TRAIN, config, read, now, atomic, sha, verify_freeze, source_contract, ARMS
from collect import collect
from inputs import Inputs
from msa import pair, quality_features, load_monomer
from statistics_rx import intervals
from predict import prohibit_training


def independent_evidence(x, h):
    centered = (x - np.asarray(h['standard_mean'])) / np.asarray(h['standard_scale']) - np.asarray(h['pca_mean'])
    # Directly compose the fixed PCA and logistic coefficients; no estimator object.
    weights = np.asarray(h['coef'])[0] @ np.asarray(h['pca_components'])
    return np.sum(centered * weights[None, :], axis=1) + h['intercept'][0]


def replay_ids(rows, records, seed):
    groups = defaultdict(list)
    for row, record in zip(rows, records):
        length = int(np.digitize(row['length'], [512, 1024, 1536, 2048], right=True))
        depth = int(np.digitize(record['depth'], [0, 32, 64, 127], right=True))
        groups[(record['reason'], length, depth)].append(row)
    chosen = []
    for key in sorted(groups):
        chosen.extend(sorted(groups[key], key=lambda r: hashlib.sha256(f"{seed}:verify:{r['uid']}".encode()).hexdigest())[:4])
    return chosen


def main():
    started = time.monotonic(); cfg = config(); fp = verify_freeze(); heads = source_contract(); prohibit_training()
    rows = read(ROOT / 'data/pairs.json'); features, records, workers = collect(
        ROOT, rows, read(ROOT / 'data/shards.json'), fp, cfg['chunk_rows'])
    assert len(rows) == cfg['test_pairs'] and all('label' not in r for r in rows)
    with np.load(ROOT / 'data/labels.npz', allow_pickle=False) as f:
        y = f['y']; uids = f['uid']
    with np.load(ROOT / 'data/baseline.npz', allow_pickle=False) as f:
        base = f['baseline']; orientations = f['orientations']
        np.testing.assert_array_equal(f['uid'], uids)
    raw = np.load(PROJECT / 'retrain-v1/data/prepared/test.raw.npy', allow_pickle=False)
    proteins = read(ROOT / 'data/proteins.json'); count = 0
    with (PROJECT / 'retrain-v1/data/raw/pairs_uniprot_seqs_test.csv').open() as f:
        for i, source in enumerate(csv.DictReader(f)):
            a, b, label, original = map(int, raw[i]); row = rows[i]
            assert (a, b, original) == (row['a'], row['b'], row['row']) and original == i
            assert label == y[i] == int(source['label'])
            assert row['uid'] == uids[i] == f'test-{i:06d}' and row['split'] == 'test'
            sa, sb = proteins[str(a)]['sequence'], proteins[str(b)]['sequence']
            assert (source['query'].strip(), source['text'].strip()) == (sa, sb)
            assert row['length'] == len(sa) + len(sb)
            count += 1
    assert count == len(rows) and int(y.sum()) == len(rows) // 2
    # Independently rebuild every native score from its original audited chunk.
    native_rows = np.load(PROJECT / 'benchmark-v1/data/test.npy', allow_pickle=False)
    np.testing.assert_array_equal(native_rows, raw)
    rebuilt = np.full_like(orientations, np.nan); seen = np.zeros(len(rows), dtype=bool)
    reuse = read(ROOT / 'provenance/baseline-reuse.json')
    for source in reuse['sources']:
        p = PROJECT / source['path']; meta = read(p.with_suffix('.json'))
        assert sha(p) == source['sha256'] == meta['sha256'] and meta['fingerprint'] == reuse['source_fingerprint']
        with np.load(p, allow_pickle=False) as f:
            v = f['predictions']
        ids = v[:, 0].astype(int); np.testing.assert_array_equal(v[:, 0], ids)
        assert not seen[ids].any() and len(np.unique(ids)) == len(ids)
        np.testing.assert_array_equal(v[:, 1], y[ids]); rebuilt[ids] = v[:, 2:4]; seen[ids] = True
    assert seen.all(); np.testing.assert_array_equal(rebuilt, orientations)
    np.testing.assert_array_equal(base, rebuilt.mean(axis=1))
    with np.load(ROOT / 'results/test_predictions.npz', allow_pickle=False) as f:
        saved = {k: f[k] for k in f.files}
    assert set(saved) == {'uid', 'y', 'gate', 'available', 'baseline', *ARMS}
    for k, wanted in [('uid', uids), ('y', y), ('gate', features['gate']), ('available', features['available']), ('baseline', base)]:
        np.testing.assert_array_equal(saved[k], wanted)
    with np.load(ROOT / 'results/test_evidence.npz', allow_pickle=False) as f:
        saved_evidence = {k: f[k] for k in f.files}
    assert set(saved_evidence) == set(ARMS)
    scores = {'baseline': base}; max_error = 0.; active = features['available']
    for arm in ARMS:
        z = independent_evidence(features[arm], heads[arm])
        np.testing.assert_allclose(z, saved_evidence[arm], rtol=0, atol=1e-10)
        rebuilt_score = base.copy(); rebuilt_score[active] = base[active] + heads[arm]['alpha'] * features['gate'][active] * z[active]
        np.testing.assert_allclose(rebuilt_score, saved[arm], rtol=0, atol=1e-10)
        np.testing.assert_array_equal(saved[arm][~active], base[~active])
        max_error = max(max_error, float(np.max(np.abs(rebuilt_score - saved[arm]))))
        scores[arm] = saved[arm]
    metrics = read(ROOT / 'results/metrics.json'); primary = metrics['full_test']
    assert metrics['fingerprint'] == fp and primary['rows'] == len(rows) and primary['eligible'] == int(active.sum())
    assert metrics['alphas'] == cfg['fixed_alphas'] and metrics['heads_sha256'] == cfg['heads_sha256']
    assert not metrics['heads_fitted'] and not metrics['calibration_performed'] and not metrics['shuffled_arm']
    for arm, z in scores.items():
        np.testing.assert_allclose(primary['metrics'][arm]['ap'], average_precision_score(y, z), rtol=0, atol=1e-12)
        np.testing.assert_allclose(primary['metrics'][arm]['auroc'], roc_auc_score(y, z), rtol=0, atol=1e-12)
    independent_ci = intervals(rows, y, scores, cfg['bootstrap_replicates'], cfg['seed'], independent=True)
    for name, ci in independent_ci.items():
        for k in ['low', 'high']:
            np.testing.assert_allclose(primary['intervals'][name][k], ci[k], rtol=0, atol=1e-12)
        assert primary['intervals'][name]['replicates'] == ci['replicates']
    # Recompute the descriptive strata separately from the analysis mask builder.
    length = np.asarray([r['length'] for r in rows]); depth = np.asarray([r['depth'] for r in records])
    minimum_quality = np.asarray([min(r['msa']['quality_a'], r['msa']['quality_b']) if r.get('msa') else 0. for r in records])
    for name, result in metrics['strata_descriptive'].items():
        if name == 'eligible': mask = active
        elif name == 'fallback': mask = ~active
        elif name.startswith('length_'):
            lo, hi = map(int, name.split('_')[1:]); mask = (length >= lo) & (length <= hi)
        elif name.startswith('depth_'):
            lo, hi = map(int, name.split('_')[1:]); mask = (depth >= lo) & (depth <= hi)
        elif name.startswith('quality_mask_fraction_'):
            lo, hi = map(float, name.removeprefix('quality_mask_fraction_').split('_'))
            mask = active & (minimum_quality >= lo) & (minimum_quality < hi)
        else:
            raise ValueError('Unexpected post-hoc stratum: ' + name)
        assert result['rows'] == int(mask.sum()) and result['positive'] == int(y[mask].sum())
        assert result['eligible'] == int((active & mask).sum())
        if len(np.unique(y[mask])) == 2:
            counts = np.bincount(y[mask], minlength=2); weights = .5 / counts[y[mask]]
            for arm, z in scores.items():
                expected = [average_precision_score(y[mask], z[mask]), roc_auc_score(y[mask], z[mask]),
                            average_precision_score(y[mask], z[mask], sample_weight=weights)]
                got = [result['metrics'][arm][k] for k in ['ap', 'auroc', 'ap_at_50_percent_prevalence']]
                np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)
        else:
            assert result['status'] == 'insufficient_classes'
    for name, values in [('depth_eligible', [r['depth'] for r in records if r['available']]),
                         ('neff_eligible', [r['msa']['neff'] for r in records if r['available']])]:
        assert metrics[name]['count'] == len(values)
        if values:
            np.testing.assert_array_equal(metrics[name]['quantiles_0_25_50_75_100'], np.quantile(values, [0, .25, .5, .75, 1]))
    assert metrics['availability_reasons'] == dict(Counter(r['reason'] for r in records))
    assert metrics['feature_source_counts'] == dict(Counter(r['kind'] for r in records))
    # Check exact query identity and exclusion on every retained monomer context.
    forbidden = set(read(ROOT / 'data/context-exclusion-hashes.json'))
    assert forbidden == {p['sha256'] for p in read(TRAIN / 'data/proteins.json').values()}
    assert not forbidden.intersection(p['sha256'] for p in proteins.values())
    catalog = read(ROOT / 'data/monomer-catalog.json'); retained = 0
    for pid, entry in catalog.items():
        p = PROJECT / entry['path']; assert sha(p) == entry['sha256']; m = load_monomer(p)
        assert m['query'] == proteins[pid]['sequence'] and m['query_sha256'] == proteins[pid]['sha256']
        assert len(m['rows']) <= cfg['maximum_cached_homologs_per_protein']
        assert len({r[0] for r in m['rows']}) == len(m['rows'])
        for key, sequence, taxonomy in m['rows']:
            assert len(sequence) == len(m['query'])
            assert hashlib.sha256(sequence.replace('-', '').encode()).hexdigest() not in forbidden
        retained += len(m['rows'])
    inputs = Inputs(reuse=False); replays = replay_ids(rows, records, cfg['seed'])
    for row in replays:
        tokens, quality, info = inputs.materialize(row); record = records[row['index']]
        for k, v in info.items():
            assert record[k] == v, (row['uid'], k)
        np.testing.assert_array_equal(quality, features['quality'][row['index']])
        a, b = sorted([row['a'], row['b']])
        if str(a) in catalog and str(b) in catalog:
            ma, mb = inputs.monomer(a), inputs.monomer(b)
            direct, meta = pair(ma, mb, cfg)
            assert meta == info['msa']
            np.testing.assert_array_equal(quality, quality_features(ma, mb, meta))
            if direct is not None:
                np.testing.assert_array_equal(tokens, direct)
            else:
                assert tokens is None
    for row, record in zip(rows, records):
        if record['available']:
            m = record['msa']; assert record['reason'] == 'available' and row['length'] <= 1536
            assert len(m['genome_keys']) == m['retained_homologs'] == record['depth'] - 1
            assert len(m['taxonomy']) == len(m['genome_keys']) and 8 <= m['retained_homologs'] <= 127
            assert record['gate'] == min(1, m['neff'] / 32) * min(m['quality_a'], m['quality_b'])
        else:
            assert record['gate'] == 0 and record['depth'] == 0 and not features['true'][row['index']].any()
    d = read(ROOT / 'results/decision.json'); c = cfg['predictive_criteria']; m = primary['metrics']
    delta = m['true']['ap'] - m['baseline']['ap']
    native_ok = delta >= c['minimum_delta_ap'] and independent_ci['true_minus_baseline']['low'] > 0
    quality_ok = independent_ci['true_minus_quality']['low'] > 0
    auroc_ok = m['true']['auroc'] >= m['baseline']['auroc'] - c['maximum_auroc_loss']
    assert d['status'] == ('useful_predictive_gain_supported' if native_ok and quality_ok and auroc_ok else 'no_go_or_inconclusive')
    assert d['native_effect_size_and_interval_pass'] == native_ok and d['beyond_quality_supported'] == quality_ok
    assert d['auroc_noninferiority'] == auroc_ok and d['test_authorized'] and d['test_accessed']
    assert not d['automatic_followup'] and not d['pairing_specific_claim']
    assert sha(ROOT / 'results/metrics.json') == d['metrics_sha256'] and d['fingerprint'] == fp
    assert not (ROOT / 'results/fit-start.json').exists() and not (ROOT / 'results/heads.json').exists()
    files = ['test_predictions.npz', 'test_evidence.npz', 'metrics.json', 'decision.json', 'REPORT.md']
    result = {'at_utc': now(), 'passed': True, 'fingerprint': fp, 'rows_verified': len(rows),
        'native_source_rows_verified': int(seen.sum()), 'eligible_rows': int(active.sum()),
        'maximum_prediction_error': max_error, 'native_fallback_exact': True,
        'frozen_heads_sha256': cfg['heads_sha256'], 'alphas': cfg['fixed_alphas'],
        'heads_fitted': False, 'calibration_performed': False, 'shuffled_arm': False,
        'test_authorized': True, 'test_accessed': True, 'independent_intervals': independent_ci,
        'monomer_queries_and_context_exclusions_verified': len(catalog), 'retained_homolog_rows_checked': retained,
        'source_replay_uids': [r['uid'] for r in replays],
        'source_replay_policy': 'Four deterministic rows per reason/length/depth bin; no outcome selection',
        'result_hashes': {name: sha(ROOT / 'results' / name) for name in files}, 'seconds': time.monotonic() - started}
    atomic(ROOT / 'results/verification.json', result)
    print({'verified': True, 'rows': len(rows), 'max_prediction_error': max_error}, flush=True)


if __name__ == '__main__':
    main()
