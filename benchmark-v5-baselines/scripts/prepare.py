"""Verify and archive the v5 splits; map existing neural scores by exact sequences."""
import csv
import gzip
import hashlib
import shutil
import time
import numpy as np
from common import *

def fasta(path):
    result, key = {}, None
    with Path(path).open() as stream:
        for line in stream:
            if line.startswith('>'):
                key = line[1:].split()[0]
                assert key not in result
                result[key] = ''
            elif line.strip():
                assert key is not None
                result[key] += line.strip()
    return result

def main():
    start = time.monotonic()
    old_protocol = read(ROOT / 'provenance/original-v11-protocol-freeze.json')
    assert CONFIG == old_protocol['config'], 'The three methods must retain the complete V11 configuration.'
    marker = ROOT / 'provenance/protocol-freeze.json'
    if marker.exists():
        verify_protocol()
    else:
        atomic(marker, {'at_utc': now(), 'protocol_sha256': sha(ROOT / 'PROTOCOL.md'), 'config': CONFIG,
                        'configuration_identical_to_v11': True, 'baseline_v5_metrics_not_yet_computed': True,
                        'previous_neural_v5_results_known': True})
    completed = ROOT / 'provenance/prepared.json'
    if completed.exists():
        old = read(completed)
        for item in old['artifacts']:
            verify(item)
        print('Prepared v5 archives and mapped data verified; no changes.', flush=True)
        return
    source = read(SOURCE / 'completed.json')
    assert source['complete']
    for rel, digest in source['files'].items():
        assert sha(SOURCE / rel) == digest, rel
    training_root = PROJECT / 'retrain-v5'
    training = read(training_root / 'data/prepared/manifest.json')
    assert training['source_completion_manifest_sha256'] == sha(SOURCE / 'completed.json')
    for rel, digest in training['files'].items():
        assert sha(training_root / 'data/prepared' / rel) == digest, rel
    assert training['counts'] == {'train': 700764, 'val': 165742}
    bench = read(BENCHMARK / 'provenance/prepared.json')
    assert bench['source_completion']['sha256'] == sha(SOURCE / 'completed.json')
    for rel in ['data/sequences.json', 'data/union.npy', 'data/pair-mapping.npz', 'data/original.npy', 'data/ilp.npy',
                'data/ipin-esm2-selected-dev.npz', 'data/ipin-esmc-selected-dev.npz']:
        assert sha(BENCHMARK / rel) == bench['data_files'][rel], rel
    archive_names = {'prepared/train.csv': 'train.csv.gz', 'prepared/val.csv': 'validation.csv.gz',
                     'prepared/sequences.fasta': 'development-sequences.fasta.gz',
                     'prepared/proteins.tsv': 'development-proteins.tsv.gz',
                     'frozen-tests/original-test.csv': 'original-test.csv.gz',
                     'frozen-tests/test-ilp.csv': 'ilp-test.csv.gz',
                     'frozen-tests/original-sequences.fasta': 'test-sequences.fasta.gz'}
    sources = []
    for rel, name in archive_names.items():
        src, dst = SOURCE / rel, ROOT / 'inputs' / name
        with src.open('rb') as inp, gzip.open(dst, 'wb', compresslevel=1) as out:
            shutil.copyfileobj(inp, out, 8*1024*1024)
        digest = hashlib.sha256()
        with gzip.open(dst, 'rb') as inp:
            for chunk in iter(lambda: inp.read(8*1024*1024), b''):
                digest.update(chunk)
        assert digest.hexdigest() == source['files'][rel]
        sources.append({'source': record(src), 'archive': record(dst), 'uncompressed_sha256': digest.hexdigest()})
    development = fasta(SOURCE / 'prepared/sequences.fasta')
    test_sequences = fasta(SOURCE / 'frozen-tests/original-sequences.fasta')
    with (SOURCE / 'prepared/proteins.tsv').open() as stream:
        for r in csv.DictReader(stream, delimiter='\t'):
            sequence = development[r['protein_id']]
            assert len(sequence) == int(r['length']) and hashlib.sha256(sequence.encode()).hexdigest() == r['sequence_sha256']
    sequences, lookup = [], {}
    def index(seq):
        if seq not in lookup:
            assert seq and seq == seq.strip() and set(seq) <= set('ACDEFGHIKLMNPQRSTVWYBXZUO.-')
            lookup[seq] = len(sequences)
            sequences.append(seq)
        return lookup[seq]
    pairs, labels, ids, counts = {}, {}, {}, {}
    for dataset, rel, dictionary in [('train', 'prepared/train.csv', development),
                                      ('validation', 'prepared/val.csv', development),
                                      ('original', 'frozen-tests/original-test.csv', test_sequences),
                                      ('ilp', 'frozen-tests/test-ilp.csv', test_sequences)]:
        rows, y, names = [], [], []
        with (SOURCE / rel).open() as stream:
            for number, row in enumerate(csv.DictReader(stream)):
                if dataset == 'original':
                    a, b = row['Uniprot_a'], row['Uniprot_b']
                    assert ''.join(row['query'].split()) == dictionary[a]
                    assert ''.join(row['text'].split()) == dictionary[b]
                    pair_id = 'original:' + str(number)
                else:
                    a, b, pair_id = row['protein1'], row['protein2'], row['pair_id']
                rows.append([index(dictionary[a]), index(dictionary[b])])
                y.append(int(row['label']))
                names.append(pair_id)
        p, y = np.asarray(rows, np.int64), np.asarray(y, np.int8)
        expected = 700764 if dataset == 'train' else 165742 if dataset == 'validation' else 52048
        assert p.shape == (expected, 2) and int(y.sum()) == expected//2 and np.isin(y, [0, 1]).all()
        assert len(set(map(tuple, np.sort(p, axis=1)))) == expected, (dataset, 'duplicate sequence pairs')
        assert not (p[:, 0] == p[:, 1]).any(), (dataset, 'unexpected self pairs')
        np.save(ROOT / 'data' / (dataset + '-pairs.npy'), p)
        np.save(ROOT / 'data' / (dataset + '-labels.npy'), y)
        atomic(ROOT / 'data' / (dataset + '-pair-ids.json'), names)
        pairs[dataset], labels[dataset], ids[dataset] = p, y, names
        counts[dataset] = {'rows': expected, 'positives': int(y.sum()), 'negatives': expected-int(y.sum()),
                           'sequences': len(np.unique(p))}
    train_ids = np.unique(pairs['train'])
    assert len(train_ids) == 13110 and len(np.unique(pairs['validation'])) == 6023
    train_set, val_set = set(train_ids), set(np.unique(pairs['validation']))
    test_set = set(np.unique(np.concatenate([pairs['original'], pairs['ilp']])))
    assert len(test_set) == 3022
    assert not (train_set & val_set or train_set & test_set or val_set & test_set)
    hashes = [hashlib.sha256(s.encode()).hexdigest() for s in sequences]
    atomic(ROOT / 'data/sequences.json', {'sequence': sequences, 'sha256': hashes,
            'length': list(map(len, sequences)), 'train_indices': train_ids.tolist()})
    for name, indices in [('train', train_ids), ('queries', range(len(sequences)))]:
        (ROOT / 'data' / (name + '.fasta')).write_text(''.join(f'>p{i:07d}\n{sequences[i]}\n' for i in indices))
    # Match actual training arrays for both architectures, including their validation row order.
    for backbone in ['esm2', 'esmc']:
        base = training_root / 'data/prepared' / backbone
        order = read(base / 'protein-order.json')
        translated = np.array([lookup[development[p]] for p in order], np.int64)
        for dataset, oldname in [('train', 'train'), ('validation', 'val')]:
            old = np.load(base / (oldname + '.npy'), allow_pickle=False)
            assert np.array_equal(translated[old[:, :2]], pairs[dataset]), (backbone, dataset)
            assert np.array_equal(old[:, 2], labels[dataset])
            assert np.array_equal(old[:, 3], np.arange(len(old)))
    selected = read(BENCHMARK / 'provenance/selection.json')
    assert sha(BENCHMARK / 'provenance/selection.json') == bench['selection_sha256']
    dev_scores, reused = {}, {}
    for n in ['ipin-esm2', 'ipin-esmc']:
        path = BENCHMARK / 'data' / (n + '-selected-dev.npz')
        z = np.load(path, allow_pickle=False)['predictions']
        assert z.shape == (165742, 4)
        assert np.array_equal(z[:, 0], np.arange(165742)) and np.array_equal(z[:, 1], labels['validation'])
        dev_scores[n] = z[:, 2:4].mean(1)
        reused[n + '-validation'] = {'source': record(path), 'selected_update': selected['models'][n]['update'],
                                     'selection_validation_ap': selected['models'][n]['validation_ap']}
    npz(ROOT / 'data/validation-neural-reused.npz', **dev_scores)
    old_meta = read(BENCHMARK / 'data/sequences.json')
    old_to_new = np.array([lookup[s] for s in old_meta['sequence']], np.int64)
    mapping = np.load(BENCHMARK / 'data/pair-mapping.npz', allow_pickle=False)
    union = np.load(BENCHMARK / 'data/union.npy', allow_pickle=False)
    collection = read(BENCHMARK / 'results/collection.json')
    assert collection['complete']
    neural = {n: np.load(verify(collection['models'][n]['file']), allow_pickle=False)['scores'] for n in REFERENCES}
    for n in REFERENCES:
        reused[n + '-tests'] = collection['models'][n]['file']
    for dataset in TESTS:
        old = np.load(BENCHMARK / 'data' / (dataset + '.npy'), allow_pickle=False)
        assert np.array_equal(old_to_new[old[:, :2]], pairs[dataset])
        assert np.array_equal(old[:, 2], labels[dataset]) and np.array_equal(old[:, 3], np.arange(len(old)))
        assert np.array_equal(np.sort(old[:, :2], axis=1), np.sort(union[mapping[dataset], :2], axis=1))
        assert np.array_equal(union[mapping[dataset], 2], labels[dataset])
        npz(ROOT / 'data' / (dataset + '-neural-reused.npz'), **{n: z[mapping[dataset]] for n, z in neural.items()})
    pair_dicts = {d: {tuple(sorted(p)): int(y) for p, y in zip(pairs[d], labels[d])} for d in TESTS}
    shared = pair_dicts['original'].keys() & pair_dicts['ilp'].keys()
    assert len(shared) == 27178 and sum(pair_dicts['original'][p] for p in shared) == 26024
    assert all(pair_dicts['original'][p] == pair_dicts['ilp'][p] for p in shared)
    for path, name in [(SOURCE / 'completed.json', 'source-completion.json'),
                       (training_root / 'data/prepared/manifest.json', 'training-input-manifest.json'),
                       (BENCHMARK / 'provenance/prepared.json', 'benchmark-prepared.json'),
                       (BENCHMARK / 'results/summary.json', 'benchmark-summary.json')]:
        shutil.copyfile(path, ROOT / 'provenance' / name)
    atomic(ROOT / 'provenance/reused-neural.json', reused)
    artifacts = [record(p) for d in ['inputs', 'data'] for p in sorted((ROOT / d).iterdir()) if p.is_file()]
    atomic(completed, {'at_utc': now(), 'counts': counts, 'sources': sources, 'artifacts': artifacts,
                      'distinct_sequences': len(sequences), 'shared_test_positives': 26024, 'shared_test_negatives': 1154,
                      'checks': {'all_source_hashes_verified': True, 'both_backbone_train_val_arrays_match': True,
                                 'all_neural_pair_mappings_match': True, 'train_val_test_sequence_disjoint': True,
                                 'unique_nonself_pairs_in_each_split': True, 'same_configuration_as_v11': True},
                      'resource_use': stage_stat(start)})
    print({'prepared': counts, 'distinct_sequences': len(sequences), 'wall_seconds': time.monotonic()-start}, flush=True)

if __name__ == '__main__':
    main()
