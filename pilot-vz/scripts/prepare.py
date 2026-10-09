"""Allowlisted verified TRAIN/DEV snapshots; no archive or TEST preparation."""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from study import ROOT, PROJECT, BASE, SOURCE, config, read, sha, token_hash, atomic, now, check_budget
from msa import load_monomer, encode

FIELDS = {
    'neff80_paired': lambda d: d['paired_diversity']['neff80'],
    'neff80_per_residue': lambda d: d['paired_diversity']['neff_per_valid_residue'],
    'minimum_quality': lambda d: d['minimum_quality'],
    'valid_interchain_cells': lambda d: d['valid_interchain_cells'],
    'joint_support_fraction': lambda d: d['true_support']['fraction_cells_at_least_half_rows'],
    'actual_null_token_change': lambda d: d['actual_token_changed_fraction'],
    'length_asymmetry': lambda d: d['length_asymmetry'],
    'candidate_depth': lambda d: d['candidates_after_coverage_and_conflict_filters']}


def main():
    if (ROOT / 'provenance/inputs.json').exists():
        raise RuntimeError('Inputs already prepared')
    started = time.monotonic(); cfg = config(); check_budget()
    sources = {}
    manifests = {}
    for folder, key in [(BASE, 'source_fingerprint'), (SOURCE, 'v2_fingerprint'), (PROJECT / 'pilot-vy', 'vy_fingerprint')]:
        p = folder / 'provenance/freeze.json'
        if sha(p) != cfg[key]:
            raise ValueError('Wrong source freeze: ' + str(folder))
        manifests[key] = read(p); sources[str(p.relative_to(PROJECT))] = sha(p)
    def snapshot(p, name, expected):
        digest = sha(p)
        if digest != expected:
            raise ValueError('Source checksum mismatch: ' + str(p))
        sources[str(p.relative_to(PROJECT))] = digest
        (ROOT / 'data' / name).write_bytes(p.read_bytes())
    for name in ['sample.json', 'proteins.json', 'baseline.json', 'dev-groups.json', 'monomer-catalog.json',
                 'source.npz', 'source_metadata.json', 'source_heads.json', 'source_metrics.json', 'dev_family_witnesses.json']:
        p = SOURCE / 'data' / name
        snapshot(p, name, manifests['v2_fingerprint']['files'][str(p.relative_to(PROJECT))])
    snapshot(BASE / 'config.json', 'original_config.json', manifests['source_fingerprint']['files']['pilot-vx/config.json'])
    for name in ['common.py', 'msa.py', 'encoder.py', 'analyze.py']:
        p = BASE / 'scripts' / name; digest = sha(p)
        if digest != manifests['source_fingerprint']['files'][str(p.relative_to(PROJECT))]:
            raise ValueError('Vx code changed')
        sources[str(p.relative_to(PROJECT))] = digest
    vy = PROJECT / 'pilot-vy'; complete = read(vy / 'results/COMPLETE.json')
    if not complete['verified'] or complete['fingerprint'] != cfg['vy_fingerprint']:
        raise ValueError('VY reference predictions not verified')
    for name in ['dev_predictions.npz', 'verification.json']:
        snapshot(vy / 'results' / name, 'vy_' + name, complete['result_hashes'][name])
    sources['pilot-vy/results/COMPLETE.json'] = sha(vy / 'results/COMPLETE.json')
    v2decision = read(SOURCE / 'results/decision.json')
    snapshot(SOURCE / 'results/metrics.json', 'v2_metrics.json', v2decision['metrics_sha256'])
    sample = read(ROOT / 'data/sample.json'); previous = read(ROOT / 'data/source_metadata.json')
    proteins = read(ROOT / 'data/proteins.json'); catalog = read(ROOT / 'data/monomer-catalog.json')
    if len(sample) != cfg['expected']['pairs'] or len(previous) != len(sample) or len({r['uid'] for r in sample}) != len(sample):
        raise ValueError('Invalid sample population')
    with np.load(ROOT / 'data/source.npz', allow_pickle=False) as f:
        if f['uids'].tolist() != [r['uid'] for r in sample]:
            raise ValueError('Source feature ordering differs')
        gate = f['gate'].copy()
    for r, p, g in zip(sample, previous, gate):
        if r['split'] not in ('train', 'val') or any(proteins[str(r[k])]['split'] != r['split'] for k in ('a', 'b')):
            raise ValueError('Unexpected cohort')
        if p['uid'] != r['uid'] or [p['a'], p['b']] != sorted([r['a'], r['b']]) or p['gate'] != g or p['available'] != (g > 0):
            raise ValueError('Source pair identity/eligibility mismatch')
        if r['length'] != sum(len(proteins[str(r[k])]['sequence']) for k in ('a', 'b')):
            raise ValueError('Source length differs')
    required = sorted({p[k] for p in previous if p['available'] for k in ('a', 'b')})
    def monomer(pid):
        rec = catalog[str(pid)]; path = BASE / rec['path']; digest = sha(path)
        if digest != rec['sha256']:
            raise ValueError('Changed source monomer')
        m = load_monomer(path)
        if m['query'] != proteins[str(pid)]['sequence'] or m['query_sha256'] != proteins[str(pid)]['sha256']:
            raise ValueError('Wrong monomer query')
        q = encode(m['query']); q[np.array(list(m['mask'])) != '*'] = 26
        return str(pid), digest, q
    with ThreadPoolExecutor(max_workers=8) as ex:
        monomers = list(ex.map(monomer, required))
    query = {p: q for p, _, q in monomers}; digests = {p: h for p, h, _ in monomers}
    atomic(ROOT / 'data/monomer-digests.json', digests)
    def old_record(r):
        path = SOURCE / 'features' / (r['uid'] + '.json'); side = read(path.with_suffix('.sha.json'))
        if side['fingerprint'] != cfg['v2_fingerprint'] or sha(path) != side['sha256']:
            raise ValueError('Corrupt V2 feature source')
        x = read(path)
        if x['fingerprint'] != cfg['v2_fingerprint'] or x['uid'] != r['uid']:
            raise ValueError('Wrong V2 feature identity')
        return x, side['sha256']
    with ThreadPoolExecutor(max_workers=8) as ex:
        old = list(ex.map(old_record, sample))
    pairs = []; diagnostics = []
    for r, p, (v2, _) in zip(sample, previous, old):
        if any(v2[k] != p[k] for k in ('uid', 'a', 'b', 'available', 'gate', 'reason')):
            raise ValueError('V2 and Vx eligibility/identity differ')
        e = {k: p[k] for k in ('uid', 'a', 'b', 'available', 'gate', 'reason')}
        e['length'] = r['length']; e['breakpoint'] = len(proteins[str(p['a'])]['sequence'])
        if p['available']:
            q = np.concatenate([query[str(p['a'])], query[str(p['b'])]])[None]
            e.update(q_tokens_sha256=token_hash(q), paired_tokens_sha256=v2['tokens_sha256'])
            diagnostics.append({k: float(fn(v2['diagnostics'])) for k, fn in FIELDS.items()})
        else:
            e.update(q_tokens_sha256=None, paired_tokens_sha256=None); diagnostics.append(None)
        pairs.append(e)
    counts = {'pairs': len(sample), 'train': sum(r['split'] == 'train' for r in sample), 'dev': sum(r['split'] == 'val' for r in sample),
              'eligible_train': sum(p['available'] and r['split'] == 'train' for r, p in zip(sample, pairs)),
              'eligible_dev': sum(p['available'] and r['split'] == 'val' for r, p in zip(sample, pairs)),
              **{k: sum(r['role'] == k for r in sample) for k in ('calibration', 'assessment', 'crossing')}}
    if counts != cfg['expected']:
        raise ValueError('Fixed cohort counts differ')
    atomic(ROOT / 'data/pairs.json', pairs); atomic(ROOT / 'data/diagnostics.json', diagnostics)
    atomic(ROOT / 'results/coverage.json', counts)
    atomic(ROOT / 'provenance/source-artifact-digests.json', {'monomers': digests, 'v2_feature_records': {r['uid']: h for r, (_, h) in zip(sample, old)}})
    image = PROJECT / 'images/msa-pairformer/msa-pairformer-arm64-v1.sif'; image_hash = sha(image)
    if image_hash != manifests['source_fingerprint']['artifacts'][str(image.relative_to(PROJECT))]:
        raise ValueError('Wrong SIF')
    atomic(ROOT / 'provenance/inputs.json', {'at_utc': now(), 'source_files': sources,
        'image': str(image.relative_to(PROJECT)), 'image_sha256': image_hash, 'required_pairs': sum(p['available'] for p in pairs),
        'source_monomers_verified': len(monomers), 'v2_feature_records_verified': len(old), 'counts': counts,
        'test_accessed': False, 'outcome_heads_fitted': False, 'seconds': time.monotonic() - started})
    print({'prepared': True, 'counts': counts, 'seconds': time.monotonic() - started}, flush=True)


if __name__ == '__main__':
    main()
