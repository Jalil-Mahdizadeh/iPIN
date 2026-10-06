"""Freeze historical inputs and three development folds; no test labels are read."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/prepared'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def describe(rows, lengths):
    pair_lengths = lengths[rows[:, 0]] + lengths[rows[:, 1]]
    return {'rows': len(rows), 'positive': int(rows[:, 2].sum()),
            'proteins': len(np.unique(rows[:, :2])),
            'maximum_tokens': int(pair_lengths.max()) + 3,
            'over_2193': int((pair_lengths > 2193).sum()),
            'short_rows': int((pair_lengths <= 2193).sum()),
            'positive_fraction': float(rows[:, 2].mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fasta-only', action='store_true')
    args = p.parse_args()
    original = json.loads((ROOT / 'provenance/v1-prepared-manifest.json').read_text())
    subset_path = ROOT / 'provenance/token-subset.json'
    subset = json.loads(subset_path.read_text()) if subset_path.exists() else None
    for name in ['train.npy', 'val.npy', 'tokens.npy', 'offsets.npy']:
        path = DATA / 'official' / name if name in ['train.npy', 'val.npy'] else DATA / name
        expected = subset['files'][name] if subset and name in subset['files'] else original['files'][name]
        assert sha(path) == expected, name
    offsets = np.load(DATA / 'offsets.npy')
    tokens = np.load(DATA / 'tokens.npy', mmap_mode='r')
    lengths = np.diff(offsets)
    train = np.load(DATA / 'official/train.npy')
    val = np.load(DATA / 'official/val.npy')
    if subset is None:
        # The v1 shared token table also held unused test sequences. Preserve
        # existing train/validation protein IDs, but retain only their residues.
        used = set(np.concatenate((train[:, :2].ravel(), val[:, :2].ravel())).tolist())
        kept, new_offsets = [], [0]
        for index in range(max(used) + 1):
            sequence = np.array(tokens[offsets[index]:offsets[index + 1]]) if index in used else np.empty(0, dtype=np.uint8)
            kept.append(sequence); new_offsets.append(new_offsets[-1] + len(sequence))
        previous_tokens = len(tokens)
        del tokens
        for name, array in [('tokens.npy', np.concatenate(kept)), ('offsets.npy', np.asarray(new_offsets, dtype=np.int64))]:
            temporary = DATA / (name + '.partial')
            with temporary.open('wb') as f: np.save(f, array)
            temporary.replace(DATA / name)
        write_json(subset_path, {'source_files': {n: original['files'][n] for n in ['tokens.npy', 'offsets.npy']},
            'files': {n: sha(DATA / n) for n in ['tokens.npy', 'offsets.npy']},
            'active_proteins': len(used), 'previous_token_count': previous_tokens,
            'retained_token_count': new_offsets[-1], 'test_sequences_retained': False,
            'method': 'Retain tokens only for IDs appearing in the unchanged official train/validation arrays; unused IDs have zero length.'})
        offsets = np.load(DATA / 'offsets.npy')
        tokens = np.load(DATA / 'tokens.npy', mmap_mode='r')
        lengths = np.diff(offsets)
    proteins = np.unique(train[:, :2])
    assert not set(proteins) & set(np.unique(val[:, :2]))
    vocab = (ROOT / 'assets/esm2/vocab.txt').read_text().splitlines()
    fasta = DATA / 'development-proteins.fasta'
    with fasta.open('w') as out:
        for protein in proteins:
            seq = ''.join(vocab[t] for t in tokens[offsets[protein]:offsets[protein + 1]])
            assert len(seq) == lengths[protein]
            out.write(f'>{protein}\n{seq}\n')
    if args.fasta_only:
        print(json.dumps({'proteins': len(proteins), 'fasta': str(fasta), 'sha256': sha(fasta)}))
        return
    edges_path = ROOT / 'data/development-homology.tsv'
    parent = {int(x): int(x) for x in proteins}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    def union(a, b):
        a, b = find(a), find(b)
        parent[max(a, b)] = min(a, b)
    edges = []
    for line in edges_path.read_text().splitlines():
        a, b, ident, qcov, tcov, _ = line.split('\t')
        a, b = int(a), int(b)
        assert a in parent and b in parent
        if float(ident) >= .4 and float(qcov) >= .8 and float(tcov) >= .8:
            union(a, b)
            if a != b: edges.append((a, b))
    components = {}
    for protein in proteins: components.setdefault(find(int(protein)), []).append(int(protein))
    rng = np.random.default_rng(20260929)
    groups = list(components.values())
    rng.shuffle(groups)
    # Balance cluster protein counts only. Labels, model scores, official
    # validation, and test data do not participate in fold assignment.
    groups.sort(key=len, reverse=True)
    fold_proteins = [[], [], []]
    for group in groups:
        candidates = np.flatnonzero(np.array(list(map(len, fold_proteins))) == min(map(len, fold_proteins)))
        fold_proteins[int(rng.choice(candidates))].extend(group)
    assigned = {p: f for f, group in enumerate(fold_proteins) for p in group}
    assert all(assigned[a] == assigned[b] for a, b in edges)
    partitions = {'official': {'train': describe(train, lengths), 'val': describe(val, lengths)}}
    for fold, group in enumerate(fold_proteins):
        membership = np.isin(train[:, :2], group)
        training = train[~membership.any(axis=1)]
        validation = train[membership.all(axis=1)]
        crossing = train[membership.any(axis=1) & ~membership.all(axis=1)]
        assert len(training) + len(validation) + len(crossing) == len(train)
        assert not set(training[:, :2].ravel()) & set(validation[:, :2].ravel())
        assert set(np.unique(training[:, 2])) == set(np.unique(validation[:, 2])) == {0, 1}
        folder = DATA / f'fold-{fold}'
        folder.mkdir(exist_ok=True)
        np.save(folder / 'train.npy', training)
        np.save(folder / 'val.npy', validation)
        np.save(folder / 'crossing-unused.npy', crossing)
        partitions[folder.name] = {'train': describe(training, lengths), 'val': describe(validation, lengths),
                                   'crossing_excluded': len(crossing), 'heldout_protein_group_size': len(group)}
    write_json(DATA / 'fold-groups.json', {'seed': 20260929, 'assignment_uses_labels': False,
               'grouping': 'connected components of detected >=40% identity, >=80% coverage of both sequences, E<=0.001 edges',
               'mmseqs_sensitivity': 7.5, 'homology_search_sha256': sha(edges_path),
               'component_count': len(groups), 'largest_component': max(map(len, groups)),
               'nonself_directed_edges': len(edges), 'fold_proteins': fold_proteins})
    files = [p for p in DATA.rglob('*') if p.is_file() and p.name != 'manifest.json']
    manifest = {'version': 2, 'benchmark': 'Bernett human PPI, unchanged prepared v1 rows',
                'test_imported': False, 'new_evidence_curated': False, 'new_negative_sampling': False,
                'negative_meaning': 'degree-controlled sampled unreported interactions; not confirmed noninteractions',
                'partitions': partitions, 'files': {str(p.relative_to(DATA)): sha(p) for p in sorted(files)},
                'parent_manifest_sha256': sha(ROOT / 'provenance/v1-prepared-manifest.json')}
    write_json(DATA / 'manifest.json', manifest)
    print(json.dumps({'partitions': partitions, 'homology_components': len(groups),
                      'largest_component': max(map(len, groups))}, indent=2))


if __name__ == '__main__': main()
