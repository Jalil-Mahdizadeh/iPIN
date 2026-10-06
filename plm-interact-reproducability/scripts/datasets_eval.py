"""Explicit test-only tasks. All row IDs are zero-based in the released CSV."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPECIES = ['mouse','fly','worm','yeast','ecoli']

def task_specs():
    tasks=[]
    for s in SPECIES:
        path=ROOT/f'data/cross_species_benchmarking/test/{s}.ppi.qrels.seq.test.csv'
        tasks.append(dict(name=f'cross_{s}',model='human',kind='ppi',path=str(path),max_length=1603))
    tasks.append(dict(name='mutation_zero_full',model='human',kind='mutation',path=str(ROOT/'data/Mutation_effect_dataset/test_mutation_data.csv'),max_length=None))
    for limit in (1603,2196):
        tasks.append(dict(name=f'mutation_zero_{limit}',model='human',kind='mutation',path=str(ROOT/'data/Mutation_effect_dataset/test_mutation_data.csv'),max_length=limit,over_limit_only=True))
    for s in SPECIES:
        # Row subset is fixed independently of labels, scores, or sequence lengths.
        tasks.append(dict(name=f'reverse_{s}_sample',model='human',kind='ppi',path=str(ROOT/f'data/cross_species_benchmarking/test/{s}.ppi.qrels.seq.test.csv'),max_length=1603,reverse=True,sample_stride=53))
    for limit in (None,1603,2196):
        tasks.append(dict(name=f'bernett_{limit or "full"}',model='bernett',kind='ppi',path=str(ROOT/'data/Bernett_benchmarking/pairs_uniprot_seqs_test.csv'),max_length=limit,over_limit_only=limit is not None))
    for limit in (None,1603,2196):
        tasks.append(dict(name=f'mutation_ft_{limit or "full"}',model='mutation',kind='mutation',path=str(ROOT/'data/Mutation_effect_dataset/test_mutation_data.csv'),max_length=limit,over_limit_only=limit is not None))
    return tasks

def read_task(spec):
    with open(spec['path']) as f: raw=list(csv.DictReader(f))
    rows=[]
    for i,r in enumerate(raw):
        if i % spec.get('sample_stride',1):continue
        if spec['kind']=='mutation':
            assert r['Affected_species']==r['Participant_species']=='9606 - Homo sapiens'
            a,b,m=r['wild_seq'],r['participant_sequence'],r['mutant_seq']
        else:
            a,b,m=r['query'],r['text'],''
        if spec.get('reverse'):a,b=b,a
        assert a==a.strip() and b==b.strip() and m==m.strip()
        # The released mutation file contains three literal '.' gap symbols.
        # ESM's vocabulary includes '.' and '-'; preserve the supplied strings.
        assert a and b and set(a+b+m) <= set('ACDEFGHIKLMNPQRSTVWYBXZUO.-')
        length=max(len(a),len(m))+len(b)+3
        if spec.get('over_limit_only') and length <= spec['max_length']:continue
        rows.append(dict(row_id=i,label=int(r['label']),a=a,b=b,mutant=m,length=length))
    return rows
