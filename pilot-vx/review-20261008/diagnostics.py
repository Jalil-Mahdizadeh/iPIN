"""Posthoc TRAIN/DEV review of cached pilot outputs; no fitting or GPU use.

Run from the repository root with the existing CPU container:
  apptainer exec images/msa-pairformer/msa-pairformer-arm64-v1.sif \
    python -B pilot-vx/review-20261008/diagnostics.py

The optional family evidence is a DEV-only table from an existing local audit.
Its `family` field is one shared-family witness, NOT a complete Pfam annotation.
No TEST file, sequence, label, prediction, or exclusion-hash file is opened.
"""
import csv
import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
from common import ROOT, PROJECT, atomic, now, sha
from features import load_record


def quantiles(values):
    a = np.asarray(values, dtype=float)
    a = a[np.isfinite(a)]
    return dict(zip(['min', 'q10', 'q25', 'median', 'q75', 'q90', 'max'],
                    map(float, np.quantile(a, [0, .1, .25, .5, .75, .9, 1])))) if len(a) else {}


def ap_setup(y, scores):
    order = np.argsort(-scores, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(scores[order])), len(order) - 1]
    return order, ends, y[order]


def weighted_ap(setup, weights):
    order, ends, y = setup
    w = weights[order]
    tp = np.cumsum(w * y)[ends]
    total = np.cumsum(w)[ends]
    if tp[-1] == 0 or total[-1] == tp[-1]:
        return np.nan
    precision = np.divide(tp, total, out=np.zeros_like(tp, dtype=float), where=total > 0)
    return float(np.sum(np.diff(np.r_[0., tp]) * precision) / tp[-1])


def main():
    started = time.monotonic()
    fingerprint = sha(ROOT / 'provenance/freeze.json')
    frozen = json.loads((ROOT / 'provenance/freeze.json').read_text())['files']
    sources = ['data/sample.json', 'data/proteins.json', 'data/dev-groups.json',
               'data/baseline.json', 'results/heads.json', 'results/metrics.json',
               'scripts/features.py', 'scripts/common.py', 'scripts/msa.py',
               'scripts/encoder.py', 'scripts/analyze.py', 'config.json']
    hashes = {}
    for name in sources:
        digest = sha(ROOT / name)
        hashes[name] = digest
        key = 'pilot-vx/' + name
        if key in frozen:
            assert digest == frozen[key], ('Frozen input changed', name)
    rows = json.loads((ROOT / 'data/sample.json').read_text())
    proteins = json.loads((ROOT / 'data/proteins.json').read_text())
    assert set(r['split'] for r in rows) == {'train', 'val'}
    assert set(r['split'] for r in proteins.values()) == {'train', 'val'}
    data = []
    for i, r in enumerate(rows):
        x = load_record(ROOT / 'features', r['uid'], fingerprint)
        assert x is not None and x['uid'] == r['uid']
        assert sorted([x['a'], x['b']]) == sorted([r['a'], r['b']])
        data.append(x)
        if (i + 1) % 2000 == 0:
            print(json.dumps({'verified_records': i + 1}), flush=True)
    y = np.array([r['label'] for r in rows])
    train = np.array([r['split'] == 'train' for r in rows])
    dev = ~train
    role = np.array([r['role'] for r in rows])
    length = np.array([r['length'] for r in rows])
    gate = np.array([x['gate'] for x in data])
    available = gate > 0
    baseline = json.loads((ROOT / 'data/baseline.json').read_text())
    base = np.array([baseline[r['uid']]['score'] if r['split'] == 'val' else 0. for r in rows])
    heads = json.loads((ROOT / 'results/heads.json').read_text())
    scores = {'baseline': base}
    for name in ['true', 'shuffled', 'quality', 'true_head_shuffled_input']:
        h = heads['true' if name == 'true_head_shuffled_input' else name]
        field = 'shuffled' if name == 'true_head_shuffled_input' else name
        x = np.array([d[field] for d in data])
        standardized = (x - h['standard_mean']) / h['standard_scale']
        ev = (((standardized - h['pca_mean']) @ np.asarray(h['pca_components']).T)
              @ np.asarray(h['coef']).T + h['intercept']).ravel()
        z = base.copy()
        z[available] += h['alpha'] * gate[available] * ev[available]
        assert np.isfinite(z).all()
        assert np.array_equal(z[dev & ~available], base[dev & ~available])
        scores[name] = z
    recorded = json.loads((ROOT / 'results/metrics.json').read_text())
    assert recorded['fingerprint'] == fingerprint
    for rname, mask in [('assessment', role == 'assessment'), ('fixed_dev_sample', dev)]:
        for name in ['baseline', 'true', 'shuffled', 'quality']:
            assert abs(average_precision_score(y[mask], scores[name][mask]) -
                       recorded['results'][rname]['metrics'][name]['ap']) < 1e-12

    metadata_names = ['common_keys', 'retained_homologs', 'neff', 'taxonomic_families',
                      'quality_a', 'quality_b', 'length_a', 'length_b']
    meta = {k: np.array([d.get('msa', {}).get(k, np.nan) for d in data], float)
            for k in metadata_names}
    meta['gate'] = gate
    meta['minimum_quality'] = np.minimum(meta['quality_a'], meta['quality_b'])
    valid_a = np.rint(meta['quality_a'] * meta['length_a'])
    valid_b = np.rint(meta['quality_b'] * meta['length_b'])
    meta['valid_interchain_cells'] = valid_a * valid_b
    meta['valid_query_residues'] = valid_a + valid_b
    meta['length_asymmetry'] = np.maximum(meta['length_a'], meta['length_b']) / np.minimum(meta['length_a'], meta['length_b'])
    meta['null_changed_row_fraction'] = np.array([d.get('null', {}).get('changed_fraction', np.nan) for d in data])
    meta['depth_per_valid_query_residue'] = meta['retained_homologs'] / meta['valid_query_residues']
    profiles = np.array([d['quality'] for d in data])
    meta['mean_monomer_gap_fraction'] = profiles[:, 3]
    meta['mean_monomer_entropy'] = profiles[:, 4]
    delta = (np.array([d['true'] for d in data]) - np.array([d['shuffled'] for d in data])) / heads['true']['standard_scale']
    for j, name in enumerate(['mean', 'std', 'max', 'q95']):
        meta['standardized_true_null_rms_' + name] = np.sqrt(np.mean(delta[:, 32*j:32*(j+1)] ** 2, axis=1))

    family_path = PROJECT / 'FADI-benchmark-v1/results/per-protein/bernett-train__bernett-validation.csv.gz'
    family_by_sha = {}
    if family_path.exists():
        hashes[str(family_path.relative_to(PROJECT))] = sha(family_path)
        with gzip.open(family_path, 'rt') as handle:
            for r in csv.DictReader(handle):
                assert r['sequence_sha256'] not in family_by_sha
                family_by_sha[r['sequence_sha256']] = r
    annotations = {}
    for pid, p in proteins.items():
        if p['split'] == 'val' and p['sha256'] in family_by_sha:
            ann = family_by_sha[p['sha256']]
            assert len(p['sequence']) == int(ann['length'])
            annotations[int(pid)] = ann
    family_count = np.full(len(rows), -1)
    meta['mean_pfam_coverage'] = np.full(len(rows), np.nan)
    for i, r in enumerate(rows):
        if r['a'] in annotations and r['b'] in annotations:
            aa, bb = annotations[r['a']], annotations[r['b']]
            family_count[i] = int(aa['shared_family'] == 'True') + int(bb['shared_family'] == 'True')
            meta['mean_pfam_coverage'][i] = (float(aa['pfam_coverage']) + float(bb['pfam_coverage'])) / 2

    def describe(mask):
        ix = np.flatnonzero(mask)
        eligible = mask & available
        result = {'rows': int(mask.sum()), 'positive': int(y[mask].sum()),
                  'eligible': int(eligible.sum()), 'eligible_positive': int(y[eligible].sum()),
                  'unique_proteins': len({rows[i][k] for i in ix for k in ['a', 'b']})}
        if len(np.unique(y[mask])) == 2:
            result['metrics'] = {name: {'ap': float(average_precision_score(y[mask], z[mask])),
                                        'auroc': float(roc_auc_score(y[mask], z[mask]))}
                                 for name, z in scores.items()}
            result['true_minus_shuffled_ap'] = result['metrics']['true']['ap'] - result['metrics']['shuffled']['ap']
            result['true_minus_same_head_shuffled_ap'] = result['metrics']['true']['ap'] - result['metrics']['true_head_shuffled_input']['ap']
        result['eligible_diagnostics'] = {k: quantiles(v[eligible]) for k, v in meta.items()}
        result['eligible_diagnostics']['fraction_at_depth_cap'] = float(np.mean(meta['retained_homologs'][eligible] == 127)) if eligible.any() else None
        result['eligible_family_overlap_with_train'] = dict(Counter(map(str, family_count[eligible])))
        return result

    masks = {}
    lengths = [('le512', length <= 512), ('513_1024', (length > 512) & (length <= 1024)),
               ('1025_1536', (length > 1024) & (length <= 1536)),
               ('513_1536', (length > 512) & (length <= 1536))]
    for name, rr in [('all_dev', dev), ('assessment', role == 'assessment'),
                     ('calibration', role == 'calibration'), ('crossing', role == 'crossing')]:
        masks[name + '/all'] = rr
        for label, ll in lengths:
            masks[name + '/' + label] = rr & ll
            masks[name + '/' + label + '/eligible'] = rr & ll & available
    report = {'at_utc': now(), 'fingerprint': fingerprint, 'source_sha256': hashes,
              'script_sha256': sha(__file__), 'verified_records': len(data),
              'heads_refitted': False, 'gpu_used': False, 'test_accessed': False,
              'interpretation': 'Posthoc exploratory strata; no multiple-comparison correction. Assessment has already been examined.',
              'family_annotation': {'dev_proteins_matched_by_sha256': len(annotations),
                  'field_limitation': 'family is a single shared-with-TRAIN Pfam witness, not full family/domain annotations; no witness does not imply novel family.'},
              'depth_audit': {name: {'eligible': int(m.sum()),
                    'neff_equals_retained_depth': int(np.sum(meta['neff'][m] == meta['retained_homologs'][m])),
                    'depth_127': int(np.sum(meta['retained_homologs'][m] == 127)),
                    'gate_depth_term_saturated': int(np.sum(meta['neff'][m] >= 32))}
                    for name, m in [('train', train & available), ('dev', dev & available)]},
              'groups': {name: describe(m) for name, m in masks.items()}}
    # Joint protein multipliers preserve covariance when comparing length groups.
    dev_ix = np.flatnonzero(dev)
    pids = sorted({rows[i][k] for i in dev_ix for k in ['a', 'b']})
    lookup = {p: i for i, p in enumerate(pids)}
    ia = np.array([lookup[rows[i]['a']] for i in dev_ix])
    ib = np.array([lookup[rows[i]['b']] for i in dev_ix])
    bnames = [k for k in masks if k.startswith(('all_dev/', 'assessment/'))]
    setups = {k: {name: ap_setup(y[masks[k]], z[masks[k]]) for name, z in scores.items()} for k in bnames}
    for k in bnames:
        for name in scores:
            assert abs(weighted_ap(setups[k][name], np.ones(masks[k].sum())) -
                       report['groups'][k]['metrics'][name]['ap']) < 1e-12
    comparisons = [('true', 'baseline'), ('true', 'shuffled'), ('true', 'quality'),
                   ('true', 'true_head_shuffled_input')]
    boot = {k: {a + '-minus-' + b: [] for a, b in comparisons} for k in bnames}
    interactions = {role_name + suffix: [] for role_name in ['all_dev', 'assessment'] for suffix in ['', '/eligible']}
    rng = np.random.default_rng(20261007)
    for rep in range(1000):
        counts = rng.poisson(1, len(pids))
        w = counts[ia] * counts[ib]
        gaps = {}
        for k in bnames:
            ww = w[masks[k][dev]]
            aps = {name: weighted_ap(setup, ww) for name, setup in setups[k].items()}
            for a, b in comparisons:
                boot[k][a + '-minus-' + b].append(aps[a] - aps[b])
            gaps[k] = aps['true'] - aps['shuffled']
        for key in interactions:
            role_name, _, suffix = key.partition('/')
            ending = '/' + suffix if suffix else ''
            interactions[key].append(gaps[role_name + '/le512' + ending] - gaps[role_name + '/513_1536' + ending])
    def interval(v):
        a = np.asarray(v); a = a[np.isfinite(a)]
        return {'low': float(np.quantile(a, .025)), 'high': float(np.quantile(a, .975)), 'replicates': len(a)}
    for k in bnames:
        report['groups'][k]['ap_intervals'] = {name: interval(v) for name, v in boot[k].items()}
    report['short_minus_long_pairing_gap_intervals'] = {k: interval(v) for k, v in interactions.items()}
    report['bootstrap'] = '1000 joint matched protein Poisson multipliers across DEV; pair weight is endpoint product; two-sided percentile 95% intervals.'
    sensitivities = {
        'depth127': meta['retained_homologs'] == 127,
        'minimum_quality_ge075': meta['minimum_quality'] >= .75,
        'null_changed_ge090': meta['null_changed_row_fraction'] >= .9,
        'both_train_shared_family_witness': family_count == 2,
        'fewer_than_two_train_shared_family_witnesses': (family_count >= 0) & (family_count < 2),
    }
    report['conditional_descriptive_only'] = {
        role_name + '/' + ln + '/' + name: describe(rr & ll & available & condition)
        for role_name, rr in [('all_dev', dev), ('assessment', role == 'assessment')]
        for ln, ll in [('le512', length <= 512), ('513_1536', (length > 512) & (length <= 1536))]
        for name, condition in sensitivities.items()}
    report['short_group_influence'] = {}
    for role_name, rr in [('all_dev', dev), ('assessment', role == 'assessment')]:
        m = rr & (length <= 512)
        count_protein = Counter(rows[i][k] for i in np.flatnonzero(m) for k in ['a', 'b'])
        fams_per_row = [{annotations[r[k]]['family'] for k in ['a', 'b']
                         if r[k] in annotations and annotations[r[k]]['family']} for r in rows]
        count_family = Counter(f for i in np.flatnonzero(m) for f in fams_per_row[i])
        result = {}
        for kind, counts in [('protein', count_protein), ('shared_family_witness', count_family)]:
            values = []
            for group, n in counts.most_common(10):
                member = np.array([group in [r['a'], r['b']] if kind == 'protein' else group in fams_per_row[i] for i, r in enumerate(rows)])
                keep = m & ~member
                values.append({'group': str(group), 'excluded_pairs': int((m & member).sum()),
                               'remaining_pairs': int(keep.sum()), 'true_minus_shuffled_ap':
                    float(average_precision_score(y[keep], scores['true'][keep]) - average_precision_score(y[keep], scores['shuffled'][keep]))})
            result[kind] = values
        report['short_group_influence'][role_name] = result
    # Catch concurrent mutation of any reviewed source before publishing the report.
    for name in sources:
        assert hashes[name] == sha(ROOT / name), ('Source changed during review', name)
    report['seconds'] = time.monotonic() - started
    atomic(HERE / 'diagnostics.json', report)
    print(json.dumps({'report': str(HERE / 'diagnostics.json'), 'seconds': report['seconds'],
                      'depth_audit': report['depth_audit'],
                      'short_minus_long_pairing_gap_intervals': report['short_minus_long_pairing_gap_intervals']}, indent=2), flush=True)


if __name__ == '__main__':
    main()
