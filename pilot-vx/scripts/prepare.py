"""Freeze original Bernett TRAIN/DEV samples before inspecting MSA success."""
import argparse
import csv
import json
import numpy as np
from common import ROOT, PROJECT, atomic, config, fasta, now, sha, text_hash


def stratified_ids(rows, lengths, n, seed):
    rng = np.random.default_rng(seed); chosen = []
    bins = np.digitize(lengths, [512, 1024, 1536, 2048, 4096], right=True)
    for label in [0, 1]:
        groups = [np.flatnonzero((rows[:, 2] == label) & (bins == b)) for b in range(6)]
        target = np.array([len(g) for g in groups], float); target *= (n//2)/target.sum()
        counts = np.floor(target).astype(int)
        for k in np.argsort(-(target-counts), kind='stable')[:n//2-counts.sum()]:
            counts[k] += 1
        for group, count in zip(groups, counts):
            chosen.extend(rng.choice(group, int(count), replace=False).tolist())
    return np.array(sorted(chosen), dtype=np.int64)


def sample():
    cfg = config(); src = PROJECT/'retrain-v1/data/prepared'; dest = ROOT/'data'; dest.mkdir(exist_ok=True)
    vocab = (PROJECT/'retrain-v1/assets/esm2/vocab.txt').read_text().splitlines()
    tokens = np.load(src/'tokens.npy', mmap_mode='r'); offsets = np.load(src/'offsets.npy', mmap_mode='r')
    arrays = {s: np.load(src/f'{s}.raw.npy') for s in ['train', 'val']}
    assert [len(arrays[x]) for x in arrays] == [163192, 59260]
    prot_ids = sorted(set(np.concatenate([a[:, :2].ravel() for a in arrays.values()]).tolist()))
    seqs = {str(i): ''.join(vocab[int(t)] for t in tokens[offsets[i]:offsets[i+1]]) for i in prot_ids}
    records = []; audit = {}; sets = {}; inputs = {}
    for split, a in arrays.items():
        csv_path = PROJECT/f'retrain-v1/data/raw/pairs_uniprot_seqs_{split}.csv'
        count = 0
        with csv_path.open() as f:
            for i, row in enumerate(csv.DictReader(f)):
                x = a[i]; assert int(x[3]) == i
                assert row['query'].strip() == seqs[str(x[0])] and row['text'].strip() == seqs[str(x[1])]
                assert int(row['label']) == int(x[2]); count += 1
        assert count == len(a)
        inputs[str(csv_path.relative_to(PROJECT))] = sha(csv_path)
        inputs[str((src/f'{split}.raw.npy').relative_to(PROJECT))] = sha(src/f'{split}.raw.npy')
        sets[split] = set(a[:, :2].ravel().tolist())
        lens = offsets[a[:, 0]+1]-offsets[a[:, 0]]+offsets[a[:, 1]+1]-offsets[a[:, 1]]
        n = cfg['train_pairs' if split == 'train' else 'dev_pairs']
        indices = stratified_ids(a, lens, n, cfg['seed']+(split=='val'))
        for i in indices:
            x = a[i]; records.append({'uid': f'{split}-{int(x[3]):06d}', 'split': split,
                                     'row': int(x[3]), 'a': int(x[0]), 'b': int(x[1]), 'label': int(x[2]),
                                     'length': int(lens[i])})
        audit[split] = {'rows': len(a), 'positive': int(a[:, 2].sum()), 'proteins': len(sets[split]),
                        'sample_rows': len(indices), 'sample_positive': int(a[indices, 2].sum()),
                        'sample_over1536': int((lens[indices] > cfg['maximum_combined_residues']).sum())}
    assert sets['train'].isdisjoint(sets['val'])
    # A deterministic interleaved order supports unbiased stopping; it is independent of MSA availability.
    records.sort(key=lambda r: text_hash(f"{cfg['seed']}:{r['uid']}"))
    atomic(dest/'sample.json', records)
    metadata = {i: {'sequence': s, 'sha256': text_hash(s), 'split': 'train' if int(i) in sets['train'] else 'val'} for i, s in seqs.items()}
    atomic(dest/'proteins.json', metadata)
    # Coverage audit: evenly span length deciles in each split, never MSA coverage or labels.
    coverage = []
    for split in ['train', 'val']:
        ordered = sorted(sets[split], key=lambda i: (len(seqs[str(i)]), i))
        groups = np.array_split(ordered, 10)
        for j, g in enumerate(groups):
            ranked = sorted(g.tolist(), key=lambda i: text_hash(f"{cfg['seed']}:coverage:{i}"))
            coverage.extend(ranked[:cfg['coverage_proteins']//20])
    atomic(dest/'coverage-proteins.json', coverage)
    for split in ['train', 'val']:
        with (dest/f'{split}.fasta').open('w') as f:
            for i in sorted(sets[split]): f.write(f'>p{i}\n{seqs[str(i)]}\n')
    # No test labels or pair rows: use the existing sequence-only manifest for context exclusion.
    test_hashes = sorted(text_hash(s) for _, s in fasta(src/'test.fasta'))
    assert len(test_hashes) == 3022
    atomic(dest/'test-sequence-hashes.json', test_hashes)
    atomic(ROOT/'provenance/data-prepared.json', {'at_utc': now(), 'source_hashes': inputs, 'splits': audit,
           'test_labels_read': False, 'exact_train_dev_overlap': 0, 'coverage_proteins': len(coverage),
           'sample_sha256': sha(dest/'sample.json'), 'proteins_sha256': sha(dest/'proteins.json'),
           'sampling': 'balanced classes; proportional within-class combined-length strata; seed fixed before MSA'})
    print(json.dumps(audit), flush=True)


def split_dev(hits):
    proteins = json.loads((ROOT/'data/proteins.json').read_text()); ids = sorted(int(i) for i, p in proteins.items() if p['split']=='val')
    parent = {i: i for i in ids}
    def root(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    edges = 0
    with open(hits) as f:
        for line in f:
            a,b,identity,qcov,tcov,evalue = line.split(); a=int(a.removeprefix('p')); b=int(b.removeprefix('p'))
            assert a in parent and b in parent
            if float(identity)>=.4 and float(qcov)>=.8 and float(tcov)>=.8 and float(evalue)<=.001:
                x,y=root(a),root(b)
                if x!=y: parent[max(x,y)]=min(x,y); edges += 1
    components = {}
    for i in ids: components.setdefault(root(i), []).append(i)
    ranked = sorted(components.values(), key=lambda g: text_hash(f"{config()['seed']}:devgroup:{min(g)}"))
    # Greedy whole-component balancing uses sequence membership only.
    counts=[0,0]; assignments={}; group_ids={}
    for group in ranked:
        k=int(counts[1]<counts[0]); counts[k]+=len(group)
        for i in group: assignments[str(i)]=['calibration','assessment'][k]; group_ids[str(i)]=min(group)
    records=json.loads((ROOT/'data/sample.json').read_text()); tally={}
    for r in records:
        role='train'
        if r['split']=='val':
            x,y=assignments[str(r['a'])],assignments[str(r['b'])]; role=x if x==y else 'crossing'
        r['role']=role; tally.setdefault(role,[0,0])[r['label']]+=1
    atomic(ROOT/'data/sample.json',records)
    atomic(ROOT/'data/dev-groups.json', {'assignments':assignments,'component':group_ids})
    atomic(ROOT/'provenance/dev-split.json', {'at_utc':now(),'hits_sha256':sha(hits),'identity':.4,'both_coverage':.8,
        'evalue':.001,'component_count':len(components),'largest_component':max(map(len,components.values())),
        'protein_counts':counts,'sample_counts_negative_positive':tally,'label_independent_assignment':True,
        'sample_sha256':sha(ROOT/'data/sample.json')})
    print(json.dumps(tally),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['sample','split']);p.add_argument('--hits');a=p.parse_args()
    if a.stage=='sample':sample()
    else:split_dev(a.hits)
