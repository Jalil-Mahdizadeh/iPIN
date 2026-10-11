"""Original TEST identity, audited native scores and label-independent MSA caches."""
import csv
import gzip
import hashlib
import heapq
import json
import os
import tarfile
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from shared import ROOT, PROJECT, TRAIN, BASE, atomic, read, config, sha, now, save_npz, source_contract
from msa import cached_monomer, records, load_monomer, profile


def budget():
    if time.time() - read(ROOT / 'provenance/authorization.json')['preflight_start_unix'] > config()['budgets']['preflight_wall_hours'] * 3600:
        raise TimeoutError('Preparation/qualification budget exhausted')


def profile_one(item):
    pid, entry = item; p = PROJECT / entry['path']
    if sha(p) != entry['sha256']:
        raise ValueError('Corrupt monomer: ' + pid)
    return pid, profile(load_monomer(p))


def baseline(rows, labels):
    src = PROJECT / 'benchmark-v1'; selection = read(src / 'provenance/selection.json')
    cfg = config(); native = selection['models']['native-bernett']
    assert native['sha256'] == cfg['baseline']['sha256']
    assert sha(PROJECT / cfg['baseline']['checkpoint']) == native['sha256']
    old = read(src / 'config.json')
    assert old['scoring'] == 'mean AB/BA logits' and old['test_sequence_truncation'] is None
    assert old['precision'] == 'FP32 parameters; BF16 autocast; TF32 disabled'
    names = ['config.json', 'provenance/selection.json'] + [f'scripts/{n}' for n in ['common.py', 'infer.py', 'frozen_data.py', 'frozen_model.py']]
    files = {str((src / n).relative_to(PROJECT)): sha(src / n) for n in names}
    fp = hashlib.sha256(json.dumps({n: sha(src / n) for n in names}, sort_keys=True).encode()).hexdigest()
    qualification = read(src / 'provenance/qualification.json')
    assert qualification['passed'] and qualification['prediction_fingerprint'] == fp
    files['benchmark-v1/provenance/qualification.json'] = sha(src / 'provenance/qualification.json')
    assert sha(src / 'data/test.npy') == selection['data']['sha256']['test.npy']
    source = np.load(src / 'data/test.npy', allow_pickle=False)
    assert source.shape == (len(rows), 4)
    orientations = np.full((len(rows), 2), np.nan); seen = set(); sources = []
    for p in sorted((src / 'predictions/native-bernett-test').glob('rank-*-chunk-*.npz')):
        meta = read(p.with_suffix('.json'))
        assert meta['fingerprint'] == fp and sha(p) == meta['sha256']
        with np.load(p, allow_pickle=False) as z:
            values = z['predictions']
        assert values.ndim == 2 and values.shape[1] == 4 and np.isfinite(values).all()
        ids = values[:, 0].astype(int)
        np.testing.assert_array_equal(ids, values[:, 0])
        assert np.all((ids >= 0) & (ids < len(rows)))
        np.testing.assert_array_equal(values[:, 1], source[ids, 2])
        for i, ab, ba in zip(ids, values[:, 2], values[:, 3]):
            a, b, label, original = map(int, source[i]); r = rows[original]
            assert original not in seen and (a, b, label, original) == (r['a'], r['b'], int(labels[original]), r['row'])
            orientations[original] = ab, ba; seen.add(original)
        sources.append({'path': str(p.relative_to(PROJECT)), 'sha256': meta['sha256'], 'rows': len(ids)})
        files[str(p.relative_to(PROJECT))] = meta['sha256']
        files[str(p.with_suffix('.json').relative_to(PROJECT))] = sha(p.with_suffix('.json'))
    assert seen == set(range(len(rows))) and np.isfinite(orientations).all()
    save_npz(ROOT / 'data/baseline.npz', uid=np.asarray([r['uid'] for r in rows]),
             baseline=orientations.mean(axis=1), orientations=orientations)
    atomic(ROOT / 'provenance/baseline-reuse.json', {'at_utc': now(), 'rows_reused': len(seen),
        'missing_rows': [], 'source_fingerprint': fp, 'sources': sources, 'source_files': files,
        'native_sha256': native['sha256'], 'baseline_sha256': sha(ROOT / 'data/baseline.npz'),
        'test_predictions_read': True, 'new_performance_computed': False})
    return files


def main():
    started = time.monotonic(); cfg = config(); authorization = read(ROOT / 'provenance/authorization.json')
    assert authorization['test_authorized'] and cfg['test_authorized']
    if (ROOT / 'provenance/preparation-start.json').exists():
        raise RuntimeError('Preparation already attempted; no automatic replacement')
    pins = {str(p.relative_to(PROJECT)): sha(p) for p in [ROOT / 'config.json', ROOT / 'README.md',
        ROOT / 'scripts/prepare.py', ROOT / 'scripts/shared.py', BASE / 'scripts/msa.py', BASE / 'config.json']}
    atomic(ROOT / 'provenance/preparation-start.json', {'at_utc': now(), 'pid': os.getpid(), 'code': pins})
    source_contract()
    # Verify the source experiment before reusing any fitted artifact or algorithm.
    for p, digest in read(TRAIN / 'provenance/freeze.json')['files'].items():
        assert sha(PROJECT / p) == digest, p
    source = PROJECT / 'retrain-v1/data/prepared'; bench = PROJECT / 'benchmark-v1'
    selection = read(bench / 'provenance/selection.json'); prior = read(bench / 'provenance/input-audit.json')['splits']['test']
    files = {str((bench / n).relative_to(PROJECT)): sha(bench / n)
             for n in ['provenance/selection.json', 'provenance/input-audit.json']}
    for n in ['tokens.npy', 'offsets.npy']:
        assert sha(source / n) == selection['data']['sha256'][n]
        files[str((source / n).relative_to(PROJECT))] = sha(source / n)
    raw = source / 'test.raw.npy'; csv_path = PROJECT / 'retrain-v1/data/raw/pairs_uniprot_seqs_test.csv'
    assert sha(raw) == selection['data']['sha256']['test.npy'] == prior['prepared_sha256']
    assert sha(csv_path) == prior['source_sha256']
    array = np.load(raw, allow_pickle=False)
    assert array.shape == (cfg['test_pairs'], 4) and int(array[:, 2].sum()) == cfg['test_pairs'] // 2
    vocab_path = PROJECT / 'retrain-v1/assets/esm2/vocab.txt'; vocab = vocab_path.read_text().splitlines()
    tokens = np.load(source / 'tokens.npy', mmap_mode='r'); offsets = np.load(source / 'offsets.npy', mmap_mode='r')
    ids = sorted(set(map(int, array[:, :2].ravel())))
    assert len(ids) == cfg['test_proteins']
    proteins = {}
    for i in ids:
        seq = ''.join(vocab[int(t)] for t in tokens[offsets[i]:offsets[i+1]])
        proteins[str(i)] = {'sequence': seq, 'sha256': hashlib.sha256(seq.encode()).hexdigest(), 'split': 'test'}
    reference = read(TRAIN / 'data/proteins.json')
    assert set(proteins).isdisjoint(reference)
    forbidden = {p['sha256'] for p in reference.values()}
    assert {p['sha256'] for p in proteins.values()}.isdisjoint(forbidden)
    assert len({p['sequence'] for p in proteins.values()}) == cfg['test_proteins']
    rows = []; labels = []
    with csv_path.open() as f:
        for i, raw_row in enumerate(csv.DictReader(f)):
            a, b, label, original = map(int, array[i]); pa, pb = proteins[str(a)], proteins[str(b)]
            assert original == i and label in [0, 1] and int(raw_row['label']) == label
            assert raw_row['query'].strip() == pa['sequence'] and raw_row['text'].strip() == pb['sequence']
            rows.append({'uid': f'test-{i:06d}', 'index': i, 'row': i, 'split': 'test', 'a': a, 'b': b,
                         'length': len(pa['sequence']) + len(pb['sequence'])})
            labels.append(label)
    assert len(rows) == cfg['test_pairs']; labels = np.asarray(labels, dtype=np.int64)
    for p in [raw, csv_path, vocab_path, TRAIN / 'data/proteins.json']:
        files[str(p.relative_to(PROJECT))] = sha(p)
    atomic(ROOT / 'data/pairs.json', rows); atomic(ROOT / 'data/proteins.json', proteins)
    save_npz(ROOT / 'data/labels.npz', y=labels, uid=np.asarray([r['uid'] for r in rows]))
    atomic(ROOT / 'data/context-exclusion-hashes.json', sorted(forbidden))
    files.update(baseline(rows, labels))
    atomic(ROOT / 'results/preparation-progress.json', {'at_utc': now(), 'stage': 'archive_checksum', 'rows_verified': len(rows)})
    archive = BASE / 'data' / cfg['archive']['filename']
    assert archive.stat().st_size == cfg['archive']['bytes'] and sha(archive) == cfg['archive']['sha256']
    atomic(ROOT / 'results/preparation-progress.json', {'at_utc': now(), 'stage': 'monomer_archive', 'rows_verified': len(rows)})
    by_sequence = {p['sequence']: pid for pid, p in proteins.items()}; catalog = {}; members = 0
    out = ROOT / 'data/monomers'; out.mkdir(exist_ok=True)
    with tarfile.open(archive, 'r|gz') as tar:
        for member in tar:
            budget()
            if not member.isfile() or not member.name.endswith(('.a3m', '.a3m.gz')):
                continue
            if member.name.endswith('.gz'):
                raise ValueError('Unqualified nested compressed archive member')
            members += 1
            with tar.extractfile(member) as handle:
                iterator = records(line.decode('ascii', errors='strict') for line in handle)
                first, second = next(iterator), next(iterator)
                assert first[0].split()[0] == 'mask' and second[0].split()[0] == 'query'
                quality, query = first[1], second[1]; assert len(quality) == len(query)
                pid = by_sequence.get(query)
                if pid is not None and (pid not in catalog or member.name < catalog[pid]['member']):
                    mono = cached_monomer(iterator, query, quality, forbidden, cfg['maximum_cached_homologs_per_protein'])
                    path = out / f'{pid}.json.gz'; temp = path.with_suffix('.partial')
                    with temp.open('wb') as f:
                        with gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as gz:
                            gz.write(json.dumps(mono, separators=(',', ':'), allow_nan=False).encode())
                        f.flush(); os.fsync(f.fileno())
                    temp.replace(path)
                    catalog[pid] = {'member': member.name, 'path': str(path.relative_to(PROJECT)), 'sha256': sha(path),
                        'query_sha256': mono['query_sha256'], 'stats': mono['stats'],
                        'quality_fraction': quality.count('*') / len(quality)}
            if members % 500 == 0:
                progress = {'at_utc': now(), 'stage': 'monomer_archive', 'members': members,
                            'monomers': len(catalog), 'seconds': time.monotonic() - started}
                atomic(ROOT / 'results/preparation-progress.json', progress); print(progress, flush=True)
    assert members == read(BASE / 'results/archive-coverage.json')['counts']['members']
    atomic(ROOT / 'data/monomer-catalog.json', catalog)
    atomic(ROOT / 'results/preparation-progress.json', {'at_utc': now(), 'stage': 'profiles', 'monomers': len(catalog)})
    with ProcessPoolExecutor(max_workers=cfg['prepare_workers']) as pool:
        values = list(pool.map(profile_one, sorted(catalog.items(), key=lambda p: int(p[0])), chunksize=8))
    profile_ids, profiles = zip(*values)
    save_npz(ROOT / 'data/profiles.npz', ids=np.asarray(profile_ids), profiles=np.asarray(profiles))
    shards = [[] for _ in range(4)]; heap = [(0., i) for i in range(4)]; ranked = []
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
    for name, digest in pins.items():
        assert sha(PROJECT / name) == digest, name
    budget()
    result = {'at_utc': now(), 'complete': True, 'rows': len(rows), 'proteins': len(proteins),
        'positive': int(labels.sum()), 'monomers': len(catalog), 'unmatched_proteins': sorted(set(proteins) - set(catalog), key=int),
        'exact_trainval_query_overlap': 0, 'context_exclusion_hashes': len(forbidden),
        'native_baseline_rows': len(rows), 'archive_members': members, 'archive_sha256': cfg['archive']['sha256'],
        'source_files': files, 'test_authorized': True, 'worker_manifest_contains_labels': False,
        'new_performance_computed': False, 'heads_fitted': False, 'shuffle_called': False,
        'source_trainval_freeze_verified': True, 'seconds': time.monotonic() - started}
    atomic(ROOT / 'provenance/preparation.json', result)
    atomic(ROOT / 'results/preparation-progress.json', {'at_utc': now(), 'stage': 'completed', 'rows': len(rows), 'monomers': len(catalog)})
    print({k: v for k, v in result.items() if k not in ['source_files', 'unmatched_proteins']}, flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        atomic(ROOT / 'results/preparation-error.json', {'at_utc': now(), 'traceback': traceback.format_exc()})
        raise
