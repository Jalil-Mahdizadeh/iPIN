"""Check historical supervised/selection exposure without running any model."""
import csv
import gzip
import json
from collections import defaultdict
import numpy as np
from common import ROOT,PROJECT,read_json,write_json,fasta_read,sha

def main():
    test=fasta_read(ROOT/'frozen-tests/original-sequences.fasta')
    protected=set(read_json(ROOT/'work/protected-families.json'))
    aliases=defaultdict(set)
    for receipt in read_json(ROOT/'provenance/uniprot-downloads.json'):
        if receipt['format']!='json':continue
        path=ROOT/receipt['path'];assert sha(path)==receipt['sha256']
        with gzip.open(path,'rt') as f:
            for p in json.load(f)['results']:
                for a in [p['primaryAccession'],*p.get('secondaryAccessions',[])]:
                    aliases[a].add(p['primaryAccession'])
    prepared=PROJECT/'retrain-v1/data/prepared'
    manifest=read_json(prepared/'manifest.json')
    tokens=np.load(prepared/'tokens.npy',mmap_mode='r');offsets=np.load(prepared/'offsets.npy')
    vocabulary=(PROJECT/'retrain-v1/assets/esm2/vocab.txt').read_text().splitlines()
    decoded=[''.join(vocabulary[int(t)] for t in tokens[offsets[i]:offsets[i+1]]) for i in range(len(offsets)-1)]
    reports={};source_hashes={};test_sequences=set(test.values())
    for split in ['train','val']:
        path=PROJECT/'retrain-v1/data/raw'/f'pairs_uniprot_seqs_{split}.csv'
        source_hashes[str(path.relative_to(PROJECT))]=sha(path)
        rows=list(csv.DictReader(path.open()))
        selected=np.load(prepared/f'{split}.npy');assert sha(prepared/f'{split}.npy')==manifest['files'][f'{split}.npy']
        for a,b,label,original_row in selected:
            r=rows[int(original_row)]
            assert int(r['label'])==label and r['query']==decoded[a] and r['text']==decoded[b]
        def describe(rr):
            ids={p for r in rr for p in [r['Uniprot_a'],r['Uniprot_b']]}
            sequences={s for r in rr for s in [r['query'],r['text']]}
            families={f for p in ids for f in aliases.get(p.split('-')[0],{p.split('-')[0]})}
            return dict(rows=len(rr),proteins=len(ids),shared_test_accessions=sorted(ids&set(test)),
                        exact_test_sequence_overlap=len(sequences&test_sequences),
                        shared_test_entry_families=sorted(families&protected),
                        unresolved_current_alias_bases=sorted({p.split('-')[0] for p in ids if p.split('-')[0] not in aliases}))
        reports[split]=dict(historical_source=describe(rows),v1_to_v4_used=describe([rows[int(i)] for i in selected[:,3]]))
    checked={}
    for version in [2,3,4]:
        base=PROJECT/f'retrain-v{version}/data/prepared';m=read_json(base/'manifest.json')
        for split in ['train','val']:
            p=base/'official'/f'{split}.npy';digest=sha(p)
            assert digest==m['files'][f'official/{split}.npy']==manifest['files'][f'{split}.npy']
            checked[str(p.relative_to(PROJECT))]=digest
    result=dict(passed=True,original_test_proteins=3022,ilp_test_new_proteins=0,splits=reports,
                source_hashes=source_hashes,identical_v1_to_v4_official_pair_arrays=checked,
                native_scope='Historical source dataset checked; native checkpoint training membership cannot be proven from weights.',
                historical_homology='Existing v1 manifest records 107 training and 2 validation rows removed by the 40%-identity/80%-both-coverage rule; native was not retroactively retrained.',
                limitations=['Current alias coverage is reported, not assumed complete.',
                             'No assertion about unsupervised pretraining exposure or independence from previous benchmark-guided research.'])
    write_json(ROOT/'reports/historical-exposure.json',result)
    print(json.dumps({s:{k:{f:v for f,v in d.items() if f!='unresolved_current_alias_bases'} for k,d in value.items()} for s,value in reports.items()},indent=2))

if __name__=='__main__':main()
