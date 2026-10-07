"""Verified source rows, TRAIN references and reused frozen neural scores."""
import csv
from nonhuman_common import *

def main():
    start = time.monotonic(); identity = freeze_extension(); verify_legacy_models()
    marker = ROOT / 'provenance/nonhuman/prepared.json'
    if marker.exists():
        previous = read(marker); assert previous['identity'] == identity
        for item in previous['artifacts'] + previous['sources']: verify(item)
        print('Nonhuman prepared inputs verified; reused.', flush=True); return
    old = read(ROOT / 'data/sequences.json')
    seqs = list(old['sequence']); lookup = {s:i for i,s in enumerate(seqs)}
    assert len(lookup) == len(seqs)
    def index(s):
        assert s and s == s.strip() and set(s) <= set('ACDEFGHIKLMNPQRSTVWYBXZUO.-')
        if s not in lookup: lookup[s] = len(seqs); seqs.append(s)
        return lookup[s]
    sources = [record(ROOT / 'data/sequences.json')]
    for rel in ['data/train-pairs.npy', 'data/train-labels.npy']:
        sources.append(record(ROOT / rel))
    train_pairs = {'v5': np.load(ROOT / 'data/train-pairs.npy')}
    train_labels = {'v5': np.load(ROOT / 'data/train-labels.npy')}
    downloads = read(PROJECT / 'retrain-v1/provenance/downloads.json')
    receipt = next(r for r in downloads if r['file'] == 'pairs_uniprot_seqs_train.csv')
    path = PROJECT / 'retrain-v1' / receipt['local_path']; assert sha(path) == receipt['sha256']
    sources.append(record(path)); pairs = []; labels = []
    with path.open() as f:
        for row in csv.DictReader(f):
            pairs.append([index(row['query']), index(row['text'])]); labels.append(int(row['label']))
    train_pairs['bernett'] = np.asarray(pairs, np.int64); train_labels['bernett'] = np.asarray(labels, np.int8)
    assert len(pairs) == 163192 and sum(labels) == 81596
    prepared = read(NONHUMAN / 'provenance/prepared.json'); sources.append(record(NONHUMAN / 'provenance/prepared.json'))
    def original(rel):
        p = NONHUMAN / rel; assert sha(p) == prepared['data_files'][rel], rel
        sources.append(record(p)); return p
    other = read(original('data/sequences.json')); trans = np.asarray([index(s) for s in other['sequence']], np.int64)
    assert other['sha256'] == [hashlib.sha256(s.encode()).hexdigest() for s in other['sequence']]
    union = np.load(original('data/union.npy')); mapping = np.load(original('data/pair-mapping.npz'))
    historical = read(HISTORICAL / 'results/summary.json'); collection = read(NONHUMAN / 'results/collection.json')
    assert historical['complete'] and collection['complete']
    sources += [record(HISTORICAL / 'results/summary.json'), record(NONHUMAN / 'results/collection.json'),
                record(NONHUMAN / 'results/combined-figure-metrics.csv')]
    neural = {}
    for name in NEURAL:
        info = historical['prediction_provenance'][name] if name.startswith('v2-') else collection['models'][name]
        source = canonical_source(info['file']); sources.append(record(source))
        z = np.load(source); neural[name] = z['scores']
        assert neural[name].shape == (len(union),) and np.isfinite(neural[name]).all()
    counts = {}; query_ids = set()
    for species in SPECIES:
        arr = np.load(original('data/' + species + '.npy'))
        p = trans[arr[:, :2]]; y = arr[:, 2].astype(np.int8)
        assert np.array_equal(arr[:, 3], np.arange(len(arr)))
        assert np.array_equal(np.sort(union[mapping[species], :2], axis=1), np.sort(arr[:, :2], axis=1))
        # The union's label column is a placeholder and is never used as ground truth.
        source = canonical_source(prepared['source_files'][species]); sources.append(record(source))
        with source.open() as handle:
            n = 0
            for n, row in enumerate(csv.DictReader(handle), 1):
                a, b = p[n-1]
                assert row['query'] == seqs[a] and row['text'] == seqs[b]
                assert int(row['label']) == int(y[n-1])
        assert n == len(p)
        assert len(p) == (22000 if species == 'ecoli' else 55000) and int(y.sum()) == len(y)//11
        npz(NDATA / (species + '-rows.npz'), pairs=p, labels=y, source_row_id=np.arange(len(p)))
        npz(NDATA / (species + '-neural.npz'), **{name:v[mapping[species]] for name,v in neural.items()})
        query_ids.update(p.ravel().tolist())
        counts[species] = {'rows': len(p), 'positives': int(y.sum()), 'negatives': int((1-y).sum()),
                           'proteins': len(np.unique(p))}
    train_ids = {ref:np.unique(p).tolist() for ref,p in train_pairs.items()}
    assert len(train_ids['v5']) == 13110 and len(train_ids['bernett']) == 4286
    for ref in TRAINING:
        npz(NDATA / (ref + '-train.npz'), pairs=train_pairs[ref], labels=train_labels[ref])
        (NDATA / (ref + '-train.fasta')).write_text(''.join(f'>p{i:07d}\n{seqs[i]}\n' for i in train_ids[ref]))
    (NDATA / 'queries.fasta').write_text(''.join(f'>p{i:07d}\n{seqs[i]}\n' for i in sorted(query_ids)))
    hashes = [hashlib.sha256(s.encode()).hexdigest() for s in seqs]; assert len(set(hashes)) == len(seqs)
    assert seqs[:len(old['sequence'])] == old['sequence']
    atomic(NDATA / 'sequences.json', {'sequence': seqs, 'sha256': hashes, 'length': list(map(len, seqs)),
                                    'train_indices': train_ids, 'query_indices': sorted(query_ids), 'legacy_v5_universe_size': len(old['sequence'])})
    artifacts = [record(p) for p in sorted(NDATA.iterdir()) if p.is_file()]
    atomic(marker, {'at_utc': now(), 'identity': identity, 'sources': sources, 'artifacts': artifacts,
                   'counts': counts, 'unique_query_sequences': len(query_ids), 'unique_all_sequences': len(seqs),
                   'train_sequences': {k:len(v) for k,v in train_ids.items()},
                   'bernett_release': receipt, 'source_rows_verified': sum(v['rows'] for v in counts.values()),
                   'resource_use': stage_stat(start)})
    print('Prepared', counts, 'TRAIN proteins', {k:len(v) for k,v in train_ids.items()}, flush=True)

if __name__ == '__main__': main()
