"""Retrospective VAL diagnostics with immutable predictors and candidate sets."""
import hashlib
import time
import numpy as np
from io_v8 import ROOT, arrays, guard, read, save, write, now, sha
from metrics import (queries, component_draws, within_anchor, early_retrieval,
                     select_quartets, quartet_metrics, summary)


def main():
    started = time.monotonic(); guard(); cfg = read(ROOT / 'config.json')
    prepared = read(ROOT / 'provenance/preparation.json')
    if sha(ROOT / 'data/val.npz') != prepared['val_sha256']:
        raise RuntimeError('Prepared validation input changed')
    data = arrays(ROOT / 'data/val.npz'); models = cfg['models']; results = {}
    for view in cfg['views']:
        mask = data['role'] == 'assessment'
        if view == 'assessment_new_rows':
            mask &= ~data['pilot']
        a, b, y, uid, scores = [data[k][mask] for k in ['a', 'b', 'y', 'uid', 'scores']]
        if np.any(a == b) or not np.isfinite(scores).all():
            raise RuntimeError('Invalid candidate/score coverage')
        used = np.unique(np.r_[a, b])
        ci, counts, labels = component_draws(data['component'][used], cfg['seed'], cfg['replicates'])
        comp = np.full(len(data['component']), -1, dtype=np.int64); comp[used] = ci
        qs = queries(a, b, y)
        qp, qb, qv = within_anchor(scores, qs, comp, counts)
        rp, rb, rv, rqs = early_retrieval(scores, y, qs, comp, counts,
                                         cfg['retrieval_k'], cfg['retrieval_min_negative'])
        quartet_rows, endpoints, candidates = select_quartets(
            a, b, y, uid, f'{cfg["seed"]}:{view}', cfg['maximum_quartets'],
            cfg['positive_edge_cap'], cfg['endpoint_cap'])
        sp, sb, sv = quartet_metrics(scores, quartet_rows, endpoints, comp, counts)
        # A nonconstant endpoint score must cancel before interpreting the real swaps.
        unary = np.sin(np.arange(len(comp), dtype=float))
        additive = (unary[a] + unary[b])[:, None]
        additive_point, additive_draws, additive_credit = quartet_metrics(
            additive, quartet_rows, endpoints, comp, counts)
        if len(quartet_rows):
            np.testing.assert_array_equal(additive_credit, .5)
            np.testing.assert_allclose(additive_draws[np.isfinite(additive_draws)], .5, rtol=0, atol=1e-15)
        anchor_components = len(set(comp[[q[0] for q in qs]].tolist()))
        retrieval_components = len(set(comp[[q[0] for q in rqs]].tolist()))
        quartet_components = len(np.unique(comp[endpoints])) if len(endpoints) else 0
        qs_ok = len(qs) >= cfg['minimum_anchors'] and anchor_components >= cfg['minimum_components']
        rs_ok = len(rqs) >= cfg['minimum_anchors'] and retrieval_components >= cfg['minimum_components']
        ss_ok = len(quartet_rows) >= cfg['minimum_quartets'] and quartet_components >= cfg['minimum_components']
        estimates = {'within_anchor': (qp, qb, qs_ok), 'swap_preference': (sp, sb, ss_ok)}
        for i, k in enumerate(cfg['retrieval_k']):
            estimates[f'recall@{k}'] = (rp[:, 2*i], rb[:, :, 2*i], rs_ok)
            estimates[f'ndcg@{k}'] = (rp[:, 2*i+1], rb[:, :, 2*i+1], rs_ok)
        metrics, contrasts = {}, {}
        for metric, (point, boot, valid_support) in estimates.items():
            metrics[metric] = {m: summary(point[j], boot[:, j], valid_support, cfg['minimum_finite_fraction'])
                               for j, m in enumerate(models)}
            contrasts[metric] = {'vx_minus_' + m: summary(point[1] - point[j], boot[:, 1] - boot[:, j],
                                  valid_support, cfg['minimum_finite_fraction']) for j, m in enumerate(models) if j != 1}
        sizes = [len(q[1]) + len(q[2]) for q in rqs]
        support = {'rows': len(a), 'positive': int(y.sum()), 'negative_labelled': int((~y).sum()),
                   'active_evidence_rows': int((data['gate'][mask] > 0).sum()),
                   'participating_proteins': len(used), 'participating_components': len(labels),
                   'anchors': len(qs), 'anchor_components': anchor_components,
                   'retrieval_anchors': len(rqs), 'retrieval_anchor_components': retrieval_components,
                   'retrieval_candidate_size_quantiles': np.quantile(sizes, [0, .25, .5, .75, 1]).tolist() if sizes else [],
                   'eligible_quartet_candidates': candidates, 'selected_quartets': len(quartet_rows),
                   'quartet_components': quartet_components,
                   'anchor_support_pass': qs_ok, 'retrieval_support_pass': rs_ok, 'swap_support_pass': ss_ok}
        results[view] = {'support': support, 'metrics': metrics, 'contrasts': contrasts,
                         'additive_cancellation_checked': True,
                         'counts_sha256': hashlib.sha256(counts.astype('<i4').tobytes()).hexdigest()}
        save(ROOT / 'data' / f'{view}_bootstrap.npz',
             a=a, b=b, y=y, uid=uid, scores=scores, component=comp, counts=counts,
             within_points=qp, within_draws=qb, anchor_values=qv,
             retrieval_points=rp, retrieval_draws=rb, retrieval_values=rv,
             swap_points=sp, swap_draws=sb, swap_credits=sv,
             quartet_rows=quartet_rows, quartet_endpoints=endpoints)
        print({'view_complete': view, 'support': support,
               'within_anchor_vx_minus_native': contrasts['within_anchor']['vx_minus_native'],
               'swap_vx_minus_native': contrasts['swap_preference']['vx_minus_native']}, flush=True)
    primary = results[cfg['primary']]
    checks = {}
    for metric in ['within_anchor', 'swap_preference']:
        for comparator in ['native', 'quality']:
            r = primary['contrasts'][metric]['vx_minus_' + comparator]
            checks[metric + '_above_' + comparator] = bool(r['ci95'] and r['ci95'][0] > 0)
            if comparator == 'native':
                checks[metric + '_practical_gain'] = r['point'] is not None and r['point'] >= cfg['minimum_partner_gain']
    for comparator in ['aac_cosine', 'kmer3_cosine']:
        r = primary['contrasts']['swap_preference']['vx_minus_' + comparator]
        checks['swap_above_' + comparator] = bool(r['ci95'] and r['ci95'][0] > 0)
    write(ROOT / 'results/validation_diagnostics.json', {
        'at_utc': now(), 'models': models, 'primary': cfg['primary'], 'views': results,
        'research_screen': {'status': 'supports_further_targeted_research' if all(checks.values()) else 'no_go_or_inconclusive',
                            'checks': checks}, 'seconds': time.monotonic() - started,
        'bootstrap_replicates': cfg['replicates'], 'seed': cfg['seed'],
        'sequence_component_rule': 'inherited MMseqs >=40% identity, >=80% coverage both endpoints, E<=1e-3',
        'interpretation': 'Retrospective fixed-model diagnostics, not untouched confirmation or physical binding specificity.',
        'retrieval_uncertainty': 'Anchor-component resampling, conditional on fixed lists; shared-partner dependence remains.',
        'training_uncertainty_included': False, 'test_accessed': False, 'new_encoder_calls': 0})


if __name__ == '__main__':
    main()
