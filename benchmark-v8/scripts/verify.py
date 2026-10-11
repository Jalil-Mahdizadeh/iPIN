"""Independent arithmetic and membership audit; no production metric imports."""
from collections import Counter, defaultdict
import hashlib
import math
import numpy as np
from io_v8 import ROOT, INPUT, arrays, read, write, guard, sha, now


def quartet_reference(a, b, y, uid, salt, maximum, edge_cap, endpoint_cap):
    negative = defaultdict(dict); positive = defaultdict(list)
    for i, (x, z, state) in enumerate(zip(a, b, y)):
        if state:
            positive[int(x)].append((int(z), i)); positive[int(z)].append((int(x), i))
        else:
            negative[int(x)][int(z)] = i; negative[int(z)][int(x)] = i
    candidates = []
    for first in np.flatnonzero(y):
        aa, bb = int(a[first]), int(b[first])
        for cc in negative[bb]:
            for dd, second in positive[cc]:
                if second <= first or len({aa, bb, cc, dd}) != 4 or dd not in negative[aa]:
                    continue
                flip = int(a[second] != cc)
                key = hashlib.sha256(f'{salt}:{uid[first]}:{uid[second]}:{flip}'.encode()).digest()
                candidates.append((key, (aa, bb, cc, dd),
                                   (int(first), second, negative[aa][dd], negative[bb][cc])))
    candidates.sort()
    edge, protein = Counter(), Counter(); rows, eps = [], []
    for key, ep, rr in candidates:
        if len(rows) == maximum:
            break
        if max(edge[x] for x in rr[:2]) >= edge_cap or max(protein[x] for x in ep) >= endpoint_cap:
            continue
        rows.append(rr); eps.append(ep); edge.update(rr[:2]); protein.update(ep)
    return np.array(rows, dtype=np.int64).reshape(-1, 4), np.array(eps, dtype=np.int64).reshape(-1, 4), len(candidates)


def expected_rank_reference(s, y, ks):
    total = int(y.sum()); out = []
    for k in ks:
        hits = 0.; dcg = 0.
        for i in np.flatnonzero(y):
            higher = int(np.sum(s > s[i])); tied = int(np.sum(s == s[i]))
            positions = range(higher + 1, min(higher + tied, k) + 1)
            count = max(0, min(higher + tied, k) - higher)
            hits += count / tied
            dcg += sum(1 / math.log2(rank + 1) for rank in positions) / tied
        ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, total) + 1))
        out.extend([hits / total, dcg / ideal])
    return np.array(out)


def main():
    guard(); cfg = read(ROOT / 'config.json'); report = read(ROOT / 'results/validation_diagnostics.json')
    checks = []; max_error = 0.
    val = arrays(ROOT / 'data/val.npz'); source = arrays(INPUT / 'vx_predictions.npz')
    np.testing.assert_array_equal(val['uid'], source['uid'])
    np.testing.assert_array_equal(val['y'], source['y'])
    for j, key in enumerate(['baseline', 'true', 'quality']):
        np.testing.assert_array_equal(val['scores'][:, j], source[key])
    checks.append('all_native_vx_quality_logits_byte_value_identical')
    for view in cfg['views']:
        f = arrays(ROOT / 'data' / f'{view}_bootstrap.npz')
        mask = val['role'] == 'assessment'
        if view == 'assessment_new_rows':
            mask &= ~val['pilot']
        for key in ['a', 'b', 'y', 'uid', 'scores']:
            np.testing.assert_array_equal(f[key], val[key][mask])
        a, b, y, s = [f[k] for k in ['a', 'b', 'y', 'scores']]
        rows, endpoints, candidate_count = quartet_reference(a, b, y, f['uid'], f'{cfg["seed"]}:{view}',
            cfg['maximum_quartets'], cfg['positive_edge_cap'], cfg['endpoint_cap'])
        np.testing.assert_array_equal(rows, f['quartet_rows'])
        np.testing.assert_array_equal(endpoints, f['quartet_endpoints'])
        assert candidate_count == report['views'][view]['support']['eligible_quartet_candidates']
        checks.append(view + ':independent_graph_enumeration_all_quartets_and_caps')
        q = defaultdict(list)
        for i in range(len(y)):
            q[int(a[i])].append((i, int(b[i]))); q[int(b[i])].append((i, int(a[i])))
        selected = [(anchor, rows_) for anchor, rows_ in sorted(q.items())
                    if any(y[i] for i, p in rows_) and any(not y[i] for i, p in rows_)]
        ci, counts = f['component'], f['counts']
        repids = np.linspace(0, len(counts) - 1, 12, dtype=int)
        total = np.zeros((len(repids), s.shape[1])); mass = np.zeros(len(repids))
        points = []; retrieved = []; retanchors = []
        for anchor, members in selected:
            positives = [(i, p) for i, p in members if y[i]]
            negatives = [(i, p) for i, p in members if not y[i]]
            pi = np.array([i for i, p in positives]); ui = np.array([i for i, p in negatives])
            credit = (s[pi, None, :] > s[ui, :]).astype(float) + .5 * (s[pi, None, :] == s[ui, :])
            points.append(credit.mean(axis=(0, 1)))
            for k, rep in enumerate(repids):
                c = counts[rep]; ac = ci[anchor]
                pw = np.array([1 if ci[p] == ac else c[ci[p]] for i, p in positives], float)
                uw = np.array([1 if ci[p] == ac else c[ci[p]] for i, p in negatives], float)
                if pw.sum() and uw.sum():
                    mass[k] += c[ac]
                    total[k] += c[ac] * np.einsum('i,ijm,j->m', pw, credit, uw) / (pw.sum() * uw.sum())
            if len(ui) >= cfg['retrieval_min_negative']:
                ids = np.r_[pi, ui]
                retrieved.append(np.stack([expected_rank_reference(s[ids, j], y[ids], cfg['retrieval_k']) for j in range(s.shape[1])]))
                retanchors.append(anchor)
        points = np.array(points)
        np.testing.assert_allclose(points, f['anchor_values'], atol=2e-13, rtol=0)
        if len(points):
            np.testing.assert_allclose(points.mean(0), f['within_points'], atol=2e-13, rtol=0)
        oracle = np.divide(total, mass[:, None], out=np.full_like(total, np.nan), where=mass[:, None] > 0)
        np.testing.assert_allclose(oracle, f['within_draws'][repids], atol=2e-12, rtol=0)
        finite = np.isfinite(oracle)
        if finite.any():
            max_error = max(max_error, float(np.max(np.abs(oracle[finite] - f['within_draws'][repids][finite]))))
        checks.append(view + ':all_anchor_points_and_12_full_component_draws_bruteforce')
        if retrieved:
            retrieved = np.stack(retrieved)
            np.testing.assert_allclose(retrieved, f['retrieval_values'], atol=2e-13, rtol=0)
            np.testing.assert_allclose(retrieved.mean(0), f['retrieval_points'], atol=2e-13, rtol=0)
            for rep in repids:
                w = counts[rep, ci[retanchors]]
                if w.sum():
                    np.testing.assert_allclose(np.average(retrieved, weights=w, axis=0), f['retrieval_draws'][rep], atol=2e-12, rtol=0)
            checks.append(view + ':all_retrieval_points_individual_rank_oracle')
        if len(rows):
            credits = []
            for rr in rows:
                positive_sum = s[rr[0]] + s[rr[1]]; alternate_sum = s[rr[2]] + s[rr[3]]
                difference = positive_sum - alternate_sum
                epsilon = 1e-10 * (1 + abs(positive_sum) + abs(alternate_sum))
                credits.append(np.where(abs(difference) <= epsilon, .5, (difference > 0).astype(float)))
            credits = np.stack(credits)
            np.testing.assert_array_equal(credits, f['swap_credits'])
            for rep in repids:
                weights = np.array([math.prod(int(counts[rep, c]) for c in set(ci[ep].tolist())) for ep in endpoints], float)
                if weights.sum():
                    np.testing.assert_allclose(np.average(credits, weights=weights, axis=0), f['swap_draws'][rep], atol=2e-12, rtol=0)
            checks.append(view + ':all_swap_credits_and_12_component_draws')
        estimates = {'within_anchor': (f['within_points'], f['within_draws']),
                     'swap_preference': (f['swap_points'], f['swap_draws'])}
        for i, k in enumerate(cfg['retrieval_k']):
            estimates[f'recall@{k}'] = (f['retrieval_points'][:, 2*i], f['retrieval_draws'][:, :, 2*i])
            estimates[f'ndcg@{k}'] = (f['retrieval_points'][:, 2*i+1], f['retrieval_draws'][:, :, 2*i+1])
        for name, (point, boot) in estimates.items():
            for j, model in enumerate(cfg['models']):
                entry = report['views'][view]['metrics'][name][model]
                if entry['point'] is not None:
                    assert abs(point[j] - entry['point']) < 1e-14
                if entry['ci95'] is not None:
                    np.testing.assert_allclose(entry['ci95'], np.nanpercentile(boot[:, j], [2.5, 97.5]), atol=2e-13, rtol=0)
                if j != 1:
                    entry = report['views'][view]['contrasts'][name]['vx_minus_' + model]
                    if entry['point'] is not None:
                        assert abs(point[1] - point[j] - entry['point']) < 1e-14
                    if entry['ci95'] is not None:
                        np.testing.assert_allclose(entry['ci95'], np.nanpercentile(boot[:, 1] - boot[:, j], [2.5, 97.5]), atol=2e-13, rtol=0)
        checks.append(view + ':all_2000_draw_intervals_and_paired_differences_replayed')
    # Sequence statistics use an independent string Counter calculation.
    proteins = read(INPUT / 'vx_proteins.json')
    for i in np.linspace(0, len(val['a']) - 1, 32, dtype=int):
        ss = [proteins[str(val[k][i])]['sequence'] for k in ['a', 'b']]
        for j, width in enumerate([1, 3]):
            c = [Counter(seq[z:z+width] for z in range(len(seq)-width+1)) for seq in ss]
            den = math.sqrt(sum(v*v for v in c[0].values()) * sum(v*v for v in c[1].values()))
            value = sum(v*c[1][word] for word, v in c[0].items()) / den if den else 0.
            assert abs(value - val['scores'][i, j+3]) < 2e-14
    checks.append('32_sequence_control_pairs_independent_string_counters')
    # Rebuild external exclusions with hashed sequence-pair strings, independent of tuple joins.
    ext = arrays(ROOT / 'data/external_audit.npz'); meta = read(INPUT / 'assay_sequences.json')
    truth = arrays(INPUT / 'assay_truth.npz'); result = read(ROOT / 'results/external_feasibility.json')
    def identity(s1, s2):
        return hashlib.sha256(('\0'.join(sorted((s1, s2)))).encode()).hexdigest()
    source_rows = read(INPUT / 'vx_pairs.json')
    keys = {s: set() for s in ['train', 'val']}
    for row in source_rows:
        keys[row['split']].add(identity(proteins[str(row['a'])]['sequence'], proteins[str(row['b'])]['sequence']))
    test_hashes = set(read(INPUT / 'test_sequence_hashes.json'))
    for i, (a, b) in enumerate(zip(ext['a'], ext['b'])):
        k = identity(meta['sequence'][a], meta['sequence'][b])
        assert (k in keys['train']) == ext['exposure_train_pair'][i]
        assert (k in keys['val']) == ext['exposure_val_pair'][i]
        test_hit = meta['sha256'][a] in test_hashes or meta['sha256'][b] in test_hashes
        assert test_hit == ext['exposure_test_hash_any_endpoint'][i]
        primary = truth['primary_unseen_pairs'][i] and k not in keys['train'] and k not in keys['val'] and not test_hit
        assert primary == ext['view_primary'][i]
    for name, r in result['views'].items():
        selected = ext['view_' + name]
        assert r == {'pairs': int(selected.sum()), 'positive': int(ext['positive'][selected].sum()),
                     'assay_nondetection': int((~ext['positive'][selected]).sum())}
    checks.append('every_external_pair_exposure_and_all_attrition_counts_independently_rebuilt')
    expected_failures = []
    if result['views']['primary']['positive'] < cfg['external_min_positive']:
        expected_failures.append('primary_positive_support_below_40')
    if result['views']['primary']['assay_nondetection'] < cfg['external_min_nondetection']:
        expected_failures.append('primary_nondetection_support_below_100')
    if result['views']['primary_length_eligible']['positive'] < cfg['external_min_msa_positive']:
        expected_failures.append('even_all_length_eligible_positives_below_40_MSA_support_floor')
    assert expected_failures == result['failures']
    checks.append('fixed_external_gate_replayed_without_relaxation')
    # Readout decisions are derived from the frozen conjunction, not report prose.
    primary = report['views'][cfg['primary']]['contrasts']; decisions = {}
    for metric in ['within_anchor', 'swap_preference']:
        for model in ['native', 'quality']:
            d = primary[metric]['vx_minus_' + model]
            decisions[metric + '_above_' + model] = bool(d['ci95'] and d['ci95'][0] > 0)
            if model == 'native':
                decisions[metric + '_practical_gain'] = d['point'] is not None and d['point'] >= cfg['minimum_partner_gain']
    for model in ['aac_cosine', 'kmer3_cosine']:
        d = primary['swap_preference']['vx_minus_' + model]
        decisions['swap_above_' + model] = bool(d['ci95'] and d['ci95'][0] > 0)
    assert decisions == report['research_screen']['checks']
    checks.append('partner_research_conjunction_independently_replayed')
    write(ROOT / 'results/verification.json', {'at_utc': now(), 'passed': True, 'checks': checks,
          'maximum_sampled_bootstrap_oracle_error': max_error,
          'scope': 'All point metrics, quartet selection, reported intervals and external exposure counts; 12 raw bootstrap replicates per view independently rebuilt.',
          'test_accessed': False})
    print({'verification_passed': True, 'checks': len(checks), 'maximum_error': max_error}, flush=True)


if __name__ == '__main__':
    main()
