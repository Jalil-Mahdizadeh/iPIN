"""Snapshot verified TRAIN/DEV references and freeze auditable C token inputs."""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from study import ROOT, PROJECT, BASE, config, read, sha, token_hash, atomic, atomic_npz, now, check_budget
from inputs import monomer_reader, reconstruct, column_shuffle, seed_for
from diagnostics import describe
from heads import references


def main():
    if (ROOT / 'provenance/inputs.json').exists():
        raise RuntimeError('Inputs already prepared')
    started = time.monotonic(); cfg = config(); check_budget()
    vz = PROJECT / 'pilot-vz'; fp = sha(vz / 'provenance/freeze.json')
    if fp != cfg['vz_fingerprint']:
        raise ValueError('Wrong verified VZ provenance')
    manifest = read(vz / 'provenance/freeze.json'); complete = read(vz / 'results/COMPLETE.json')
    if not complete['verified'] or complete['fingerprint'] != fp or complete['test_accessed']:
        raise ValueError('VZ did not complete verification without TEST')
    sources = {'pilot-vz/provenance/freeze.json': fp, 'pilot-vz/results/COMPLETE.json': sha(vz / 'results/COMPLETE.json')}
    def snapshot(path, name, digest):
        if sha(path) != digest:
            raise ValueError('Changed reference: ' + str(path))
        sources[str(path.relative_to(PROJECT))] = digest
        (ROOT / 'data' / name).write_bytes(path.read_bytes())
    for name in ['sample.json', 'proteins.json', 'baseline.json', 'dev-groups.json', 'monomer-catalog.json',
                 'source.npz', 'source_metadata.json', 'source_heads.json', 'source_metrics.json', 'dev_family_witnesses.json',
                 'original_config.json', 'monomer-digests.json', 'pairs.json', 'v2_metrics.json', 'diagnostics.json',
                 'vy_dev_predictions.npz', 'vy_verification.json']:
        path = vz / 'data' / name
        snapshot(path, name, manifest['files'][str(path.relative_to(PROJECT))])
    for name in ['heads.json', 'metrics.json', 'dev_predictions.npz', 'dev_evidence.npz', 'verification.json']:
        snapshot(vz / 'results' / name, 'vz_' + name, complete['result_hashes'][name])
    for rel in ['pilot-vx/config.json', 'pilot-vx/scripts/common.py', 'pilot-vx/scripts/msa.py',
                'pilot-vx/scripts/encoder.py', 'pilot-vx/scripts/analyze.py']:
        if sha(PROJECT / rel) != manifest['files'][rel]:
            raise ValueError('Original Vx code changed')
        sources[rel] = manifest['files'][rel]
    sample = read(ROOT / 'data/sample.json'); pairs = read(ROOT / 'data/pairs.json')
    previous = read(ROOT / 'data/source_metadata.json'); proteins = read(ROOT / 'data/proteins.json')
    if len(sample) != cfg['expected']['pairs'] or len(pairs) != len(sample) or len({p['uid'] for p in pairs}) != len(pairs):
        raise ValueError('Wrong fixed cohort')
    for r, p, old in zip(sample, pairs, previous):
        if r['split'] not in ('train', 'val') or any(proteins[str(r[k])]['split'] != r['split'] for k in ('a', 'b')):
            raise ValueError('Unexpected cohort; do not open TEST')
        if p['uid'] != r['uid'] or [p['a'], p['b']] != sorted([r['a'], r['b']]) or p['available'] != (p['gate'] > 0):
            raise ValueError('Original row identity/availability differs')
        if any(old[k] != p[k] for k in ('uid', 'a', 'b', 'available', 'gate', 'reason')):
            raise ValueError('Original Vx and VZ metadata differ')
    def q_reference(p):
        path = vz / 'features' / (p['uid'] + '.json'); side = read(path.with_suffix('.sha.json')); rec = read(path)
        if side['fingerprint'] != fp or rec['fingerprint'] != fp or sha(path) != side['sha256']:
            raise ValueError('Corrupt historical Q feature')
        if any(rec.get(k) != v for k, v in p.items()):
            raise ValueError('Q reference metadata differs')
        x = np.asarray(rec['Q'], dtype=float)
        if x.shape != (128,) or not np.isfinite(x).all() or (not p['available'] and x.any()):
            raise ValueError('Invalid Q reference vector')
        return x, side['sha256']
    with ThreadPoolExecutor(max_workers=8) as pool:
        old_q = list(pool.map(q_reference, pairs))
    atomic_npz(ROOT / 'data/q_reference_features.npz', uids=np.array([p['uid'] for p in pairs]), Q=np.asarray([x for x, _ in old_q]))
    # Fail on any discrepancy in historical predictions before new C processing/fitting.
    references(ROOT, sample)
    mono = monomer_reader(); by_uid = {p['uid']: p for p in previous}
    (ROOT / 'data/column_inputs').mkdir(exist_ok=True)
    def one(p):
        check_budget(cfg['budgets']['analysis_reserve_seconds'])
        if not p['available']:
            return p, None
        t, _ = reconstruct(p, by_uid[p['uid']], mono); seed = seed_for(p)
        c, perm_sha = column_shuffle(t, p['breakpoint'], seed)
        audit = describe(t, c, p, by_uid[p['uid']], cfg['column_shuffle']['diagnostic_column_pairs_per_kind'])
        path = ROOT / 'data/column_inputs' / (p['uid'] + '.npz')
        if path.exists():
            raise RuntimeError('Refuse to overwrite prepared C inputs')
        atomic_npz(path, true=t, C=c)
        out = {**p, 'paired_depth': len(t), 'c_seed': seed, 'c_tokens_sha256': token_hash(c), 'c_permutations_sha256': perm_sha,
               'c_input_file': str(path.relative_to(ROOT)), 'c_input_sha256': sha(path)}
        return out, audit
    output = []; audits = {}; done = 0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for p, audit in pool.map(one, pairs):
            output.append(p)
            if audit is not None:
                audits[p['uid']] = audit; done += 1
                if done % 100 == 0:
                    print({'prepared_C_inputs': done, 'required': 4453, 'seconds': time.monotonic() - started}, flush=True)
    counts = {'pairs': len(sample), 'train': sum(r['split'] == 'train' for r in sample), 'dev': sum(r['split'] == 'val' for r in sample),
              'eligible_train': sum(p['available'] and r['split'] == 'train' for r, p in zip(sample, output)),
              'eligible_dev': sum(p['available'] and r['split'] == 'val' for r, p in zip(sample, output)),
              **{k: sum(r['role'] == k for r in sample) for k in ('calibration', 'assessment', 'crossing')}}
    if counts != cfg['expected'] or done != 4453:
        raise ValueError('Original fixed cohort changed')
    atomic(ROOT / 'data/pairs.json', output); atomic(ROOT / 'data/input-audits.json', audits)
    atomic(ROOT / 'results/coverage.json', counts)
    atomic(ROOT / 'provenance/source-artifact-digests.json', {'monomers': read(ROOT / 'data/monomer-digests.json'),
        'vz_Q_feature_records': {p['uid']: digest for p, (_, digest) in zip(pairs, old_q)}})
    image = PROJECT / manifest['image']; digest = sha(image)
    if digest != manifest['image_sha256']:
        raise ValueError('Wrong pinned SIF')
    atomic(ROOT / 'provenance/inputs.json', {'at_utc': now(), 'source_files': sources, 'image': str(image.relative_to(PROJECT)),
        'image_sha256': digest, 'required_pairs': done, 'verified_Q_feature_records': len(old_q), 'counts': counts,
        'prepared_column_inputs': done, 'primary_C_replicate': 0, 'R_evaluated': False, 'labels_used_for_perturbation': False,
        'test_accessed': False, 'outcome_heads_fitted': False, 'seconds': time.monotonic() - started})
    print({'prepared': True, 'counts': counts, 'seconds': time.monotonic() - started}, flush=True)


if __name__ == '__main__':
    main()
