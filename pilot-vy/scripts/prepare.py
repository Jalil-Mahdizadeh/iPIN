"""Snapshot bound TRAIN/DEV artifacts and prepare independent monomers, label-free."""
import gzip
import hashlib
import json
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from study import ROOT, PROJECT, SOURCE, BASE, read, config, sha, atomic, atomic_npz, now, check_budget, text_hash
from monomers import select, profile, pair_gate, PROFILE_NAMES


def main():
    started = time.monotonic(); cfg = config()
    if (ROOT / 'provenance/inputs.json').exists() or (ROOT / 'provenance/freeze.json').exists():
        raise RuntimeError('VY inputs already prepared; no overwrite')
    check_budget()
    for name, folder, expected in [('vx', BASE, cfg['source_vx_fingerprint']), ('v2', SOURCE, cfg['source_v2_fingerprint'])]:
        if sha(folder / 'provenance/freeze.json') != expected:
            raise ValueError('Source fingerprint changed: ' + name)
    manifest = read(SOURCE / 'provenance/freeze.json'); sources = {}
    names = ['sample.json', 'proteins.json', 'baseline.json', 'dev-groups.json', 'monomer-catalog.json',
             'source.npz', 'source_heads.json', 'source_metrics.json', 'dev_family_witnesses.json']
    for name in names:
        p = SOURCE / 'data' / name; rel = str(p.relative_to(PROJECT)); digest = sha(p)
        if manifest['files'][rel] != digest:
            raise ValueError('Changed source: ' + rel)
        sources[rel] = digest
        (ROOT / 'data' / name).write_bytes(p.read_bytes())
    for folder in (BASE, SOURCE):
        p = folder / 'provenance/freeze.json'; sources[str(p.relative_to(PROJECT))] = sha(p)
    sample = read(ROOT / 'data/sample.json'); proteins = read(ROOT / 'data/proteins.json')
    catalog = read(ROOT / 'data/monomer-catalog.json'); groups = read(ROOT / 'data/dev-groups.json')['assignments']
    assert len(sample) == 8000 and Counter(r['split'] for r in sample) == {'train': 4000, 'val': 4000}
    assert set(p['split'] for p in proteins.values()) == {'train', 'val'}
    assert len({r['uid'] for r in sample}) == len(sample)
    for r in sample:
        assert r['label'] in (0, 1)
        for key in ('a', 'b'):
            p = proteins[str(r[key])]
            assert p['split'] == r['split'] and text_hash(p['sequence']) == p['sha256']
        assert r['length'] == sum(len(proteins[str(r[k])]['sequence']) for k in ('a', 'b'))
        role = 'train' if r['split'] == 'train' else groups[str(r['a'])] if groups[str(r['a'])] == groups[str(r['b'])] else 'crossing'
        assert role == r['role']
    with np.load(ROOT / 'data/source.npz', allow_pickle=False) as f:
        assert f['uids'].tolist() == [r['uid'] for r in sample]
        assert all(np.isfinite(f[name]).all() for name in ['true', 'shuffled', 'quality', 'gate'])
        oldgate = f['gate'].copy()
    ids = sorted({str(r[k]) for r in sample for k in ('a', 'b')}, key=int)
    def one(pid):
        check_budget()
        protein = proteins[pid]; length = len(protein['sequence'])
        meta = {'pid': pid, 'split': protein['split'], 'query_sha256': protein['sha256'], 'length': length, 'available': False}
        if pid not in catalog:
            return pid, {**meta, 'reason': 'missing_exact_query'}, None
        item = catalog[pid]
        if item['query_sha256'] != protein['sha256']:
            raise ValueError('Catalog query identity mismatch')
        # Existing frozen catalog is sufficient to reject long examples without loading their MSA.
        if length > cfg['monomer']['maximum_residues']:
            return pid, {**meta, 'reason': 'long_monomer', 'quality_fraction': item['quality_fraction']}, None
        path = BASE / item['path']; digest = sha(path)
        if digest != item['sha256']:
            raise ValueError('Monomer checksum mismatch: ' + pid)
        with gzip.open(path, 'rt') as f:
            monomer = json.load(f)
        if monomer['query'] != protein['sequence']:
            raise ValueError('Monomer full query mismatch: ' + pid)
        tokens, selected = select(monomer, cfg['monomer'], cfg['seed'])
        meta.update(selected)
        if tokens is not None:
            dest = ROOT / 'data/selected' / f'{pid}.npz'
            atomic_npz(dest, tokens=tokens)
            meta['selected_path'] = str(dest.relative_to(ROOT)); meta['selected_sha256'] = sha(dest)
            meta['profile'] = profile(tokens, meta).tolist()
        return pid, meta, (str(path.relative_to(PROJECT)), digest)
    monomers = {}; monomer_digests = {}
    with ThreadPoolExecutor(max_workers=cfg['budgets']['preparation_workers']) as ex:
        for count, (pid, meta, source) in enumerate(ex.map(one, ids), 1):
            monomers[pid] = meta
            if source:
                monomer_digests[source[0]] = source[1]
            if count % 100 == 0:
                status = {'at_utc': now(), 'completed': count, 'total': len(ids), 'seconds': time.monotonic() - started}
                atomic(ROOT / 'results/preparation-progress.json', status); print(status, flush=True)
    atomic(ROOT / 'data/monomers.json', monomers)
    pairs = []
    for r, old in zip(sample, oldgate):
        a, b = monomers[str(r['a'])], monomers[str(r['b'])]; gate = pair_gate(a, b)
        reasons = sorted({x['reason'] for x in [a, b] if not x['available']})
        pairs.append({'uid': r['uid'], 'a': r['a'], 'b': r['b'], 'gate': gate, 'available': gate > 0,
                      'unavailable_reasons': reasons, 'vx_available': bool(old > 0)})
    atomic(ROOT / 'data/pairs.json', pairs)
    # Encode only proteins needed by at least one available fixed pair; no labels enter this decision.
    required = sorted({str(r[k]) for r, p in zip(sample, pairs) if p['available'] for k in ('a', 'b')}, key=int)
    atomic(ROOT / 'data/required-proteins.json', required)
    atomic(ROOT / 'data/profile-schema.json', {'dimensions': len(PROFILE_NAMES), 'names': PROFILE_NAMES})
    audit = {}
    for pop in ['train', 'calibration', 'assessment', 'crossing', 'val']:
        indices = [i for i, r in enumerate(sample) if (r['split'] == pop if pop in ['train', 'val'] else r['role'] == pop)]
        counts = Counter('both' if pairs[i]['available'] and pairs[i]['vx_available'] else 'vy_only' if pairs[i]['available'] else 'vx_only' if pairs[i]['vx_available'] else 'neither' for i in indices)
        audit[pop] = {'rows': len(indices), 'vy_available': sum(pairs[i]['available'] for i in indices),
                      'vx_available': sum(pairs[i]['vx_available'] for i in indices), 'availability_groups': dict(counts)}
    image = PROJECT / cfg['encoder']['image']; image_hash = sha(image)
    if image_hash != manifest['image_sha256']:
        raise ValueError('Container identity changed')
    atomic(ROOT / 'provenance/source-monomer-digests.json', monomer_digests)
    atomic(ROOT / 'provenance/inputs.json', {'at_utc': now(), 'source_files': sources,
        'image': cfg['encoder']['image'], 'image_sha256': image_hash, 'sample_pairs': len(sample),
        'monomers': len(monomers), 'available_monomers': sum(m['available'] for m in monomers.values()),
        'required_monomer_encodings': len(required), 'cached_monomers_verified': len(monomer_digests),
        'complete_family_annotations': False, 'test_accessed': False, 'outcome_models_fitted': False,
        'seconds': time.monotonic() - started})
    atomic(ROOT / 'results/coverage.json', {'at_utc': now(), 'populations': audit,
        'monomer_reasons': dict(Counter(m['reason'] for m in monomers.values())),
        'required_monomer_encodings': len(required), 'labels_used_for_selection': False})
    print(json.dumps({'prepared': True, 'coverage': audit, 'required_monomers': len(required)}), flush=True)


if __name__ == '__main__':
    main()
