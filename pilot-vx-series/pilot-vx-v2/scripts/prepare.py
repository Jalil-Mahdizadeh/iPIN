"""Verify and snapshot only completed TRAIN/DEV outputs; no TEST access."""
import csv
import gzip
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from study import ROOT, BASE, PROJECT, atomic, read, sha, now, config
from features import load_record


def main():
    if (ROOT / 'provenance/inputs.json').exists():
        raise RuntimeError('Inputs already prepared; do not overwrite')
    cfg = config(); fp = sha(BASE / 'provenance/freeze.json')
    assert fp == cfg['source_fingerprint']
    old_manifest = read(BASE / 'provenance/freeze.json')
    sources = {}
    for name in ['sample.json', 'proteins.json', 'baseline.json', 'dev-groups.json', 'monomer-catalog.json']:
        p = BASE / 'data' / name; digest = sha(p)
        assert old_manifest['files'][str(p.relative_to(PROJECT))] == digest
        sources[str(p.relative_to(PROJECT))] = digest
        (ROOT / 'data' / name).write_bytes(p.read_bytes())
    for name in ['heads.json', 'metrics.json']:
        p = BASE / 'results' / name
        sources[str(p.relative_to(PROJECT))] = sha(p)
        (ROOT / 'data' / ('source_' + name)).write_bytes(p.read_bytes())
    for p in [BASE / 'config.json', *sorted((BASE / 'scripts').glob('*.py'))]:
        digest = sha(p)
        assert old_manifest['files'][str(p.relative_to(PROJECT))] == digest
        sources[str(p.relative_to(PROJECT))] = digest
    sample = read(ROOT / 'data/sample.json'); proteins = read(ROOT / 'data/proteins.json')
    assert len(sample) == 8000 and set(r['split'] for r in sample) == {'train', 'val'}
    assert set(p['split'] for p in proteins.values()) == {'train', 'val'}
    def one(r):
        x = load_record(BASE / 'features', r['uid'], fp)
        assert x is not None and x['uid'] == r['uid']
        assert sorted([r['a'], r['b']]) == [x['a'], x['b']]
        for pid in [r['a'], r['b']]:
            assert proteins[str(pid)]['split'] == r['split']
        assert r['length'] == sum(len(proteins[str(p)]['sequence']) for p in [r['a'], r['b']])
        return x, read(BASE / 'features' / (r['uid'] + '.sha.json'))['sha256']
    with ThreadPoolExecutor(max_workers=8) as ex:
        outputs = list(ex.map(one, sample))
    records = [x for x, _ in outputs]
    arrays = {k: np.array([x[k] for x in records]) for k in ['true', 'shuffled', 'quality', 'gate']}
    arrays['uids'] = np.array([r['uid'] for r in sample])
    target = ROOT / 'data/source.npz'; temporary = target.with_suffix('.partial')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    temporary.replace(target)
    metadata = [{k: v for k, v in x.items() if k not in ['true', 'shuffled', 'quality']} for x in records]
    atomic(ROOT / 'data/source_metadata.json', metadata)
    atomic(ROOT / 'provenance/source-feature-digests.json', {r['uid']: digest for r, (_, digest) in zip(sample, outputs)})
    # Optional existing table contains DEV queries against TRAIN only.
    family = PROJECT / 'FADI-benchmark-v1/results/per-protein/bernett-train__bernett-validation.csv.gz'
    ann = {}; family_by_sha = {}
    if family.exists():
        sources[str(family.relative_to(PROJECT))] = sha(family)
        with gzip.open(family, 'rt') as f:
            family_by_sha = {r['sequence_sha256']: r for r in csv.DictReader(f)}
        for pid, p in proteins.items():
            if p['split'] == 'val' and p['sha256'] in family_by_sha:
                a = family_by_sha[p['sha256']]
                assert int(a['length']) == len(p['sequence'])
                ann[pid] = {k: a[k] for k in ['family', 'clan', 'shared_family', 'shared_clan', 'pfam_coverage', 'adequate', 'sequence_relationship']}
    atomic(ROOT / 'data/dev_family_witnesses.json', ann)
    image = PROJECT / 'images/msa-pairformer/msa-pairformer-arm64-v1.sif'
    image_hash = sha(image)
    assert image_hash == old_manifest['artifacts'][str(image.relative_to(PROJECT))]
    atomic(ROOT / 'provenance/inputs.json', {'at_utc': now(), 'source_fingerprint': fp, 'source_files': sources,
        'image': str(image.relative_to(PROJECT)), 'image_sha256': image_hash, 'records_verified': len(records),
        'eligible_train': sum(x['available'] and r['split'] == 'train' for x, r in zip(records, sample)),
        'eligible_dev': sum(x['available'] and r['split'] == 'val' for x, r in zip(records, sample)),
        'dev_family_witness_proteins': len(ann), 'complete_family_annotations': False, 'test_accessed': False})
    print(json.dumps({'prepared': True, 'records': len(records), 'family_witness_proteins': len(ann)}), flush=True)


if __name__ == '__main__':
    main()
