"""Complete original TRAIN/VAL and bounded monomer context; no TEST opens."""
import csv
import gzip
import heapq
import json
import os
import tarfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from shared import ROOT, PROJECT, BASE, atomic, read, config, sha, now, save_npz, mono_path
from msa import cached_monomer, records, load_monomer, profile


def profile_one(item):
    pid, entry = item
    p = mono_path(entry)
    if sha(p) != entry['sha256']:
        raise ValueError('Monomer checksum changed: ' + pid)
    m = load_monomer(p)
    return pid, profile(m)


def baseline(rows):
    import hashlib
    src = PROJECT / 'benchmark-v1'; selection = read(src / 'provenance/selection.json')
    native = selection['models']['native-bernett']; cfg = config()
    assert native['sha256'] == cfg['baseline']['sha256']
    assert sha(PROJECT / cfg['baseline']['checkpoint']) == native['sha256']
    old = read(src / 'config.json')
    assert old['scoring'] == 'mean AB/BA logits' and old['test_sequence_truncation'] is None
    assert old['precision'] == 'FP32 parameters; BF16 autocast; TF32 disabled'
    names = ['config.json', 'provenance/selection.json'] + [f'scripts/{n}' for n in ['common.py', 'infer.py', 'frozen_data.py', 'frozen_model.py']]
    fp = hashlib.sha256(json.dumps({n: sha(src / n) for n in names}, sort_keys=True).encode()).hexdigest()
    q = read(src / 'provenance/qualification.json')
    assert q['passed'] and q['prediction_fingerprint'] == fp
    assert sha(src / 'data/val.npy') == selection['data']['sha256']['val.npy']
    source_rows = np.load(src / 'data/val.npy', allow_pickle=False)
    scores = np.full((len(rows), 2), np.nan); seen = set(); sources = []
    for p in sorted((src / 'predictions/native-bernett-val').glob('rank-*-chunk-*.npz')):
        meta = read(p.with_suffix('.json'))
        assert meta['fingerprint'] == fp and sha(p) == meta['sha256']
        with np.load(p, allow_pickle=False) as z:
            values = z['predictions']
        assert values.ndim == 2 and values.shape[1] == 4 and np.isfinite(values).all()
        ids = values[:, 0].astype(int)
        np.testing.assert_array_equal(ids, values[:, 0])
        np.testing.assert_array_equal(values[:, 1], source_rows[ids, 2])
        for i, ab, ba in zip(ids, values[:, 2], values[:, 3]):
            a, b, label, original = map(int, source_rows[i])
            assert original not in seen and 0 <= original < len(rows)
            r = rows[original]
            assert (a, b, label, original) == (r['a'], r['b'], r['label'], r['row'])
            scores[original] = ab, ba; seen.add(original)
        sources.append({'path': str(p.relative_to(PROJECT)), 'sha256': meta['sha256']})
    assert len(seen) == len(source_rows)
    missing = [r for r in rows if r['row'] not in seen]
    save_npz(ROOT / 'data/baseline-partial.npz', orientations=scores)
    atomic(ROOT / 'data/baseline-missing.json', missing)
    atomic(ROOT / 'provenance/baseline-reuse.json', {'at_utc': now(), 'rows_reused': len(seen),
        'missing_rows': [r['row'] for r in missing], 'source_fingerprint': fp, 'sources': sources,
        'native_sha256': native['sha256'], 'test_predictions_read': False})
    return len(missing)


def main():
    start = time.monotonic(); cfg = config()
    if (ROOT / 'provenance/preparation.json').exists():
        raise RuntimeError('Preparation already completed; do not overwrite')
    pins = {str(p.relative_to(PROJECT)): sha(p) for p in [ROOT / 'config.json', ROOT / 'README.md',
        ROOT / 'scripts/prepare.py', ROOT / 'scripts/shared.py', BASE / 'scripts/msa.py', BASE / 'config.json']}
    atomic(ROOT / 'provenance/preparation-start.json', {'at_utc': now(), 'pid': os.getpid(), 'code': pins})
    proteins = read(BASE / 'data/proteins.json'); groups = read(BASE / 'data/dev-groups.json')
    assert len(proteins) == 7997 and len(groups['assignments']) == 3711
    old_sample = read(BASE / 'data/sample.json'); old_ids = {r['uid'] for r in old_sample}
    source = read(BASE / 'provenance/data-prepared.json')['source_hashes']
    rows = []; populations = {}; used = {'train': set(), 'val': set()}
    for split, expected in [('train', 163192), ('val', 59260)]:
        path = PROJECT / f'retrain-v1/data/prepared/{split}.raw.npy'
        csv_path = PROJECT / f'retrain-v1/data/raw/pairs_uniprot_seqs_{split}.csv'
        assert sha(path) == source[str(path.relative_to(PROJECT))]
        assert sha(csv_path) == source[str(csv_path.relative_to(PROJECT))]
        array = np.load(path, allow_pickle=False)
        assert array.shape == (expected, 4) and int(array[:, 2].sum()) == expected // 2
        count = 0
        with csv_path.open() as f:
            for i, raw in enumerate(csv.DictReader(f)):
                a, b, label, original = map(int, array[i]); assert original == i and label in [0, 1]
                pa, pb = proteins[str(a)], proteins[str(b)]
                assert pa['split'] == pb['split'] == split
                assert raw['query'].strip() == pa['sequence'] and raw['text'].strip() == pb['sequence']
                assert int(raw['label']) == label
                role = 'train'
                if split == 'val':
                    ra, rb = groups['assignments'][str(a)], groups['assignments'][str(b)]
                    role = ra if ra == rb else 'crossing'
                uid = f'{split}-{original:06d}'
                rows.append({'uid': uid, 'index': len(rows), 'split': split, 'row': original, 'a': a, 'b': b,
                    'label': label, 'length': len(pa['sequence']) + len(pb['sequence']), 'role': role,
                    'pilot_row': uid in old_ids})
                used[split].update([a, b]); count += 1
        assert count == expected
        populations[split] = {'rows': count, 'positive': expected // 2, 'proteins': len(used[split])}
    assert used['train'].isdisjoint(used['val']) and len(used['train'] | used['val']) == 7997
    assert all(groups['assignments'][str(r['a'])] == groups['assignments'][str(r['b'])] == r['role']
               for r in rows if r['role'] in ['assessment', 'calibration'])
    atomic(ROOT / 'data/pairs.json', rows); atomic(ROOT / 'data/proteins.json', proteins)
    atomic(ROOT / 'data/dev-groups.json', groups)
    for r in old_sample:
        new = rows[r['row'] + (163192 if r['split'] == 'val' else 0)]
        assert all(r[k] == new[k] for k in ['uid', 'a', 'b', 'label', 'length', 'role'])
    missing_baseline = baseline([r for r in rows if r['split'] == 'val'])
    atomic(ROOT / 'results/preparation-progress.json', {'at_utc': now(), 'stage': 'monomer_archive',
        'source_rows_verified': len(rows), 'baseline_missing_rows': missing_baseline})
    old_catalog = read(BASE / 'data/monomer-catalog.json'); catalog = {}
    for pid, entry in old_catalog.items():
        path = BASE / entry['path']; assert sha(path) == entry['sha256']
        catalog[pid] = {**entry, 'path': str(path.relative_to(PROJECT)), 'reused_original_cache': True}
    archive_audit = read(BASE / 'results/archive-coverage.json'); assert archive_audit['complete']
    known_missing = set(map(str, archive_audit['unmatched_ids']))
    wanted = set(proteins) - set(catalog) - known_missing
    by_sequence = {proteins[i]['sequence']: i for i in wanted}
    exclusion_hashes = read(BASE / 'data/test-sequence-hashes.json')
    forbidden = {'train': set(exclusion_hashes), 'val': set()}
    for p in proteins.values():
        forbidden['val' if p['split'] == 'train' else 'train'].add(p['sha256'])
    atomic(ROOT / 'data/context-exclusion-hashes.json', {k: sorted(v) for k, v in forbidden.items()})
    archive = BASE / 'data' / cfg['archive']['filename']
    assert archive.stat().st_size == cfg['archive']['bytes'] and sha(archive) == cfg['archive']['sha256']
    out = ROOT / 'data/monomers'; out.mkdir(exist_ok=True); members = 0
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            if time.time() - read(ROOT / 'provenance/authorization.json')['preflight_start_unix'] > cfg['budgets']['preflight_wall_hours'] * 3600:
                raise TimeoutError('Preparation/qualification budget exhausted')
            if not member.isfile() or not member.name.endswith(('.a3m', '.a3m.gz')):
                continue
            if member.name.endswith('.gz'):
                raise ValueError('Unqualified nested compressed archive member')
            members += 1
            with tar.extractfile(member) as handle:
                it = records(line.decode('ascii', errors='strict') for line in handle)
                first, second = next(it), next(it)
                assert first[0].split()[0] == 'mask' and second[0].split()[0] == 'query'
                quality, query = first[1], second[1]; assert len(quality) == len(query)
                pid = by_sequence.get(query)
                if pid is not None and (pid not in catalog or member.name < catalog[pid]['member']):
                    m = cached_monomer(it, query, quality, forbidden[proteins[pid]['split']], cfg['maximum_cached_homologs_per_protein'])
                    path = out / f'{pid}.json.gz'; tmp = path.with_suffix('.partial')
                    with tmp.open('wb') as f:
                        with gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as gz:
                            gz.write(json.dumps(m, separators=(',', ':')).encode())
                        f.flush(); os.fsync(f.fileno())
                    tmp.replace(path)
                    catalog[pid] = {'member': member.name, 'path': str(path.relative_to(PROJECT)),
                        'sha256': sha(path), 'query_sha256': m['query_sha256'], 'stats': m['stats'],
                        'quality_fraction': quality.count('*') / len(quality), 'reused_original_cache': False}
            if members % 500 == 0:
                atomic(ROOT / 'results/preparation-progress.json', {'at_utc': now(), 'stage': 'monomer_archive',
                    'members': members, 'monomers': len(catalog), 'new_monomers': len(catalog) - len(old_catalog),
                    'seconds': time.monotonic() - start})
                print({'members': members, 'monomers': len(catalog)}, flush=True)
    assert members == archive_audit['counts']['members']
    atomic(ROOT / 'data/monomer-catalog.json', catalog)
    atomic(ROOT / 'results/preparation-progress.json', {'at_utc': now(), 'stage': 'profiles', 'monomers': len(catalog)})
    with ProcessPoolExecutor(max_workers=cfg['prepare_workers']) as pool:
        values = list(pool.map(profile_one, sorted(catalog.items(), key=lambda p: int(p[0])), chunksize=8))
    ids, profiles = zip(*values)
    save_npz(ROOT / 'data/profiles.npz', ids=np.asarray(ids), profiles=np.asarray(profiles))
    # Greedy workload balance uses only lengths and the fixed availability prerequisites.
    shards = [[] for _ in range(cfg['world_size'])]; heap = [(0., i) for i in range(cfg['world_size'])]
    ranked = []
    for r in rows:
        entries = [catalog.get(str(r[k])) for k in ['a', 'b']]
        candidate = r['length'] <= cfg['maximum_combined_residues'] and all(
            e and e['quality_fraction'] >= cfg['minimum_quality_fraction_per_chain'] for e in entries)
        cost = (r['length'] / 1536) ** 3 + .02 if candidate else .002
        ranked.append((cost, r))
    for cost, r in sorted(ranked, key=lambda x: (-x[0], x[1]['uid'])):
        total, rank = heapq.heappop(heap); shards[rank].append(r['index']); heapq.heappush(heap, (total + cost, rank))
    assert sorted(i for shard in shards for i in shard) == list(range(len(rows)))
    atomic(ROOT / 'data/shards.json', shards)
    for path, digest in pins.items():
        assert sha(PROJECT / path) == digest
    roles = {role: sum(r['role'] == role for r in rows) for role in ['train', 'calibration', 'assessment', 'crossing']}
    result = {'at_utc': now(), 'complete': True, 'rows': len(rows), 'splits': populations, 'roles': roles,
        'primary_new_assessment_rows': sum(r['role'] == 'assessment' and not r['pilot_row'] for r in rows),
        'original_pilot_rows_reproduced': len(old_sample), 'monomers': len(catalog), 'reused_monomers': len(old_catalog),
        'new_monomers': len(catalog) - len(old_catalog), 'unmatched_proteins': sorted(set(proteins) - set(catalog), key=int),
        'archive_members': members, 'archive_sha256': cfg['archive']['sha256'], 'baseline_missing_rows': missing_baseline,
        'source_files': source, 'test_opened': False, 'shuffle_called': False, 'seconds': time.monotonic() - start}
    atomic(ROOT / 'provenance/preparation.json', result)
    print({k: v for k, v in result.items() if k not in ['source_files', 'unmatched_proteins']}, flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        atomic(ROOT / 'results/preparation-error.json', {'at_utc': now(), 'traceback': traceback.format_exc()})
        raise
