"""Archive exact source files and verify mappings to reused neural predictions."""
import csv
import gzip
import hashlib
import shutil
import time
from pathlib import Path
import numpy as np
from common import *

def main():
    start = time.monotonic()
    for sub in ['inputs', 'data', 'work', 'models', 'results', 'provenance', 'logs', 'cache']:
        (ROOT / sub).mkdir(exist_ok=True)
    frozen_path = ROOT / 'provenance/protocol-freeze.json'
    if frozen_path.exists():
        verify_protocol()
    else:
        atomic(frozen_path, {'at_utc': now(), 'protocol_sha256': sha(ROOT / 'PROTOCOL.md'),
                             'config': CONFIG, 'previous_nonhuman_neural_results_known': True,
                             'baseline_nonhuman_metrics_not_yet_computed': True})
    complete = ROOT / 'provenance/prepared.json'
    if complete.exists():
        old = read(complete)
        for item in old['artifacts']:
            verify(item)
        print('Preparation already complete; archived and prepared hashes verified.', flush=True)
        return
    previous = read(PREVIOUS / 'provenance/inputs.json')
    nonhuman = read(NONHUMAN / 'provenance/prepared.json')
    sources = []
    for split, name in [('train', 'human-train'), ('validation', 'human-validation')]:
        item = next(r for r in previous['inputs'] if r['model_source'] == 'native' and r['split'] == split)
        src = PREVIOUS / item['archive']
        assert sha(src) == item['archive_sha256']
        dest = ROOT / 'inputs' / (name + '.csv.gz')
        shutil.copyfile(src, dest)
        sources.append({'dataset': name, 'source': record(src), 'archive': record(dest),
                        'uncompressed_sha256': item['uncompressed_sha256'], 'url': item['source_url']})
    for name in SPECIES:
        src = verify(nonhuman['source_files'][name])
        dest = ROOT / 'inputs' / (name + '.csv.gz')
        with src.open('rb') as inp, gzip.open(dest, 'wb', compresslevel=1) as out:
            shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
        sources.append({'dataset': name, 'source': record(src), 'archive': record(dest),
                        'uncompressed_sha256': sha(src),
                        'url': f"https://huggingface.co/datasets/danliu1226/cross_species_benchmarking/resolve/{previous['native_dataset_revision']}/test/{name}.ppi.qrels.seq.test.csv"})
    seqs, lookup, pairdata, labels, counts = [], {}, {}, {}, {}
    def idx(seq):
        if seq not in lookup:
            assert seq and seq == seq.strip()
            assert set(seq) <= set('ACDEFGHIKLMNPQRSTVWYBXZUO.-')
            lookup[seq] = len(seqs)
            seqs.append(seq)
        return lookup[seq]
    for item in sources:
        name = item['dataset']
        archive = Path(item['archive']['path'])
        digest = hashlib.sha256()
        with gzip.open(archive, 'rb') as inp:
            for chunk in iter(lambda: inp.read(8 * 1024 * 1024), b''):
                digest.update(chunk)
        assert digest.hexdigest() == item['uncompressed_sha256']
        arr, y = [], []
        with gzip.open(archive, 'rt', newline='') as inp:
            reader = csv.DictReader(inp)
            assert reader.fieldnames == ['query', 'text', 'label']
            for row in reader:
                arr.append([idx(row['query']), idx(row['text'])])
                y.append(int(row['label']))
        arr, y = np.array(arr, np.int64), np.array(y, np.int8)
        assert np.isin(y, [0, 1]).all()
        expected = (421792, 38344) if name == 'human-train' else ((52725, 4794) if name == 'human-validation' else ((22000, 2000) if name == 'ecoli' else (55000, 5000)))
        assert (len(y), int(y.sum())) == expected
        np.save(ROOT / 'data' / (name + '-pairs.npy'), arr)
        np.save(ROOT / 'data' / (name + '-labels.npy'), y)
        pairdata[name], labels[name] = arr, y
        counts[name] = {'rows': len(y), 'positives': int(y.sum()), 'negatives': int((1-y).sum()),
                        'sequences': len(np.unique(arr))}
    hashes = [hashlib.sha256(s.encode()).hexdigest() for s in seqs]
    train_ids = np.unique(pairdata['human-train'])
    assert len(train_ids) == 15631
    assert set(np.unique(pairdata['human-validation'])) <= set(train_ids)
    atomic(ROOT / 'data/sequences.json', {'sequence': seqs, 'sha256': hashes, 'length': list(map(len, seqs)),
                                         'train_indices': train_ids.tolist()})
    for name, indices in [('human-train', train_ids), ('queries', range(len(seqs)))]:
        (ROOT / 'data' / (name + '.fasta')).write_text(''.join(f'>p{i:07d}\n{seqs[i]}\n' for i in indices))
    # Independently reconstruct each source row and match the old union mapping.
    for rel in ['data/sequences.json', 'data/pair-mapping.npz', 'data/union.npy'] + [f'data/{s}.npy' for s in SPECIES]:
        assert sha(NONHUMAN / rel) == nonhuman['data_files'][rel], rel
    oldmeta = read(NONHUMAN / 'data/sequences.json')
    old_to_new = np.array([lookup[s] for s in oldmeta['sequence']], np.int64)
    mapping = np.load(NONHUMAN / 'data/pair-mapping.npz', allow_pickle=False)
    union = np.load(NONHUMAN / 'data/union.npy', allow_pickle=False)
    collection = read(NONHUMAN / 'results/collection.json')
    assert collection['complete']
    reused = {n: {k: v for k, v in np.load(verify(collection['models'][n]['file']), allow_pickle=False).items()}
              for n in REFERENCES}
    for name in SPECIES:
        old = np.load(NONHUMAN / 'data' / (name + '.npy'), allow_pickle=False)
        assert np.array_equal(pairdata[name], old_to_new[old[:, :2]])
        assert np.array_equal(labels[name], old[:, 2])
        assert np.array_equal(np.sort(old[:, :2], axis=1), union[mapping[name], :2])
        assert np.array_equal(old[:, 3], np.arange(len(old)))
        score = {n: reused[n]['scores'][mapping[name]] for n in REFERENCES}
        assert all(np.isfinite(v).all() for v in score.values())
        npz(ROOT / 'data' / (name + '-neural-reused.npz'), **score)
    for src, dest in [(PREVIOUS / 'provenance/inputs.json', 'prior-input-provenance.json'),
                      (PREVIOUS / 'results/summary.json', 'prior-degree-summary.json'),
                      (NONHUMAN / 'results/summary.json', 'prior-nonhuman-summary.json'),
                      (NONHUMAN / 'provenance/prepared.json', 'prior-nonhuman-prepared.json')]:
        shutil.copyfile(src, ROOT / 'provenance' / dest)
    artifacts = [record(p) for d in ['inputs', 'data'] for p in sorted((ROOT / d).iterdir()) if p.is_file()]
    atomic(complete, {'at_utc': now(), 'protocol_sha256': sha(ROOT / 'PROTOCOL.md'), 'counts': counts,
                      'distinct_sequences_total': len(seqs), 'sources': sources,
                      'reused_neural_sources': {n: collection['models'][n]['file'] for n in REFERENCES},
                      'checks': {'all_archived_bytes_match_releases': True, 'all_reused_pair_rows_match': True,
                                 'human_validation_proteins_all_in_train': True},
                      'artifacts': artifacts, 'resource_use': stage_stat(start)})
    print({'prepared': counts, 'total_sequences': len(seqs), 'wall_seconds': time.monotonic()-start}, flush=True)

if __name__ == '__main__':
    main()
