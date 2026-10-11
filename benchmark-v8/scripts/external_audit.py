"""Assay support and exact exposure audit; never import historical model scores."""
from collections import Counter
import hashlib
import numpy as np
from io_v8 import ROOT, INPUT, arrays, guard, read, save, write, now


def pair_key(a, b):
    return tuple(sorted((a, b)))


def census(mask, positive):
    return {'pairs': int(mask.sum()), 'positive': int((mask & positive).sum()),
            'assay_nondetection': int((mask & ~positive).sum())}


def main():
    guard(); cfg = read(ROOT / 'config.json')
    pair = arrays(INPUT / 'assay_candidates.npz')
    truth = arrays(INPUT / 'assay_truth.npz')
    oldex = arrays(INPUT / 'assay_exposure.npz')
    meta = read(INPUT / 'assay_sequences.json')
    proteins = read(INPUT / 'vx_proteins.json')
    rows = read(INPUT / 'vx_pairs.json')
    a, b, positive = pair['a'], pair['b'], truth['positive']
    if len(a) != 3313 or positive.dtype != bool or np.any(a == b):
        raise RuntimeError('Changed curated source census')
    for i in set(a.tolist()) | set(b.tolist()):
        if hashlib.sha256(meta['sequence'][i].encode()).hexdigest() != meta['sha256'][i]:
            raise RuntimeError('External sequence identity mismatch')
    source = read(INPUT / 'assay_source_freeze.json')
    for name, expected in source['views'].items():
        if census(truth[name], positive) != expected:
            raise RuntimeError('Historical source view does not replay: ' + name)
    historical = truth['primary_unseen_pairs']
    if np.any(historical & (oldex['training_P'] | oldex['training_U'] |
                           oldex['development_candidate'] | oldex['previous_test_candidate'])):
        raise RuntimeError('Historical exposure exclusion failed')
    keys = [pair_key(meta['sha256'][i], meta['sha256'][j]) for i, j in zip(a, b)]
    if len(set(keys)) != len(keys):
        raise RuntimeError('Duplicate curated unordered sequence pairs')
    current = {split: {pair_key(proteins[str(r['a'])]['sha256'], proteins[str(r['b'])]['sha256'])
                      for r in rows if r['split'] == split} for split in ['train', 'val']}
    endpoints = {split: {p['sha256'] for p in proteins.values() if p['split'] == split}
                 for split in ['train', 'val']}
    tests = set(read(INPUT / 'test_sequence_hashes.json'))
    if len(tests) != 3022 or any(tests & endpoints[s] for s in endpoints):
        raise RuntimeError('Cached TEST hash exclusion universe changed')
    exposure = {split + '_pair': np.array([k in current[split] for k in keys]) for split in current}
    for split, hashes in [*endpoints.items(), ('test_hash', tests)]:
        exposure[split + '_any_endpoint'] = np.array([x in hashes or y in hashes for x, y in keys])
        exposure[split + '_both_endpoints'] = np.array([x in hashes and y in hashes for x, y in keys])
    pair_free = historical & ~exposure['train_pair'] & ~exposure['val_pair']
    primary = pair_free & ~exposure['test_hash_any_endpoint']
    catalog = read(INPUT / 'vx_catalog.json')
    cached = {r['query_sha256'] for r in catalog.values()}
    cached_both = np.array([x in cached and y in cached for x, y in keys])
    lengths = np.array(meta['length'])[a] + np.array(meta['length'])[b]
    length_ok = lengths <= cfg['maximum_combined_residues']
    views = {'eligible_final_source': truth['all_eligible_final'], 'historical_primary': historical,
             'current_train_val_pair_disjoint': pair_free, 'primary': primary,
             'primary_length_eligible': primary & length_ok,
             'primary_both_cached_and_length_eligible': primary & cached_both & length_ok}
    result = {name: census(mask, positive) for name, mask in views.items()}
    required = result['primary']
    failures = []
    if required['positive'] < cfg['external_min_positive']:
        failures.append('primary_positive_support_below_40')
    if required['assay_nondetection'] < cfg['external_min_nondetection']:
        failures.append('primary_nondetection_support_below_100')
    if result['primary_length_eligible']['positive'] < cfg['external_min_msa_positive']:
        failures.append('even_all_length_eligible_positives_below_40_MSA_support_floor')
    # Length is a genuine upper bound on Vx availability. Cache absence is not.
    status = 'no_go_insufficient_panel_support' if failures else 'further_label_free_MSA_and_component_audit_required'
    save(ROOT / 'data/external_audit.npz', a=a, b=b, positive=positive, lengths=lengths,
         **{'view_' + k: v for k, v in views.items()}, **{'exposure_' + k: v for k, v in exposure.items()})
    write(ROOT / 'results/external_feasibility.json', {
        'at_utc': now(), 'status': status, 'views': result,
        'exposures_in_historical_primary': {k: census(historical & v, positive) for k, v in exposure.items()},
        'failures': failures, 'source_replayed': True,
        'exact_pair_identity': 'unordered SHA256 of full sequences',
        'test_exclusion': 'any endpoint in pre-existing 3022 TEST sequence hashes; no TEST files opened',
        'native_train_exclusion': 'conservative all raw TRAIN rows, covering the cleaned native subset',
        'cache_support_is_only_a_known_input_inventory': True,
        'uncached_external_sequences_may_exist_in_archive': True,
        'MSA_eligibility_not_yet_established': True,
        'homology_to_current_training_not_yet_audited': True,
        'component_floor_not_yet_evaluated': True,
        'component_and_MSA_audit_skipped_due_to_earlier_floor': bool(failures),
        'external_model_predictions_computed': False, 'external_truth_read_for_feasibility_only': True,
        'inference_authorized_by_gate': False, 'test_accessed': False,
        'interpretation': 'Panel-support failure is not evidence against Vx or a negative experimental result.'})
    print({'external_feasibility': status, 'views': result, 'failures': failures}, flush=True)


if __name__ == '__main__':
    main()
