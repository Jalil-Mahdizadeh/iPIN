"""Verify retained v2 training rows and exact nonhuman exposure without using scores."""
import csv
import numpy as np
from bench_utils import ROOT, PROJECT, atomic, load_npz, now, read, record, save_npz, sha


def main():
    ext=ROOT/'v1-v4-comparison';source=PROJECT/'retrain-v2/data/prepared'
    manifest=read(source/'manifest.json');meta=read(ROOT/'data/sequences.json')
    for name,digest in manifest['files'].items():assert sha(source/name)==digest,name
    tokens=np.load(source/'tokens.npy');offsets=np.load(source/'offsets.npy')
    vocab=(PROJECT/'retrain-v1/assets/esm2/vocab.txt').read_text().splitlines()
    sequences=[''.join(vocab[int(t)] for t in tokens[offsets[i]:offsets[i+1]]) for i in range(len(offsets)-1)]
    rows={s:np.load(source/'official'/f'{s}.npy') for s in ['train','val']}
    source_records=[]
    for split,data in rows.items():
        raw=PROJECT/'retrain-v1/data/raw'/f'pairs_uniprot_seqs_{split}.csv'
        selected={int(r[3]):r for r in data};verified=0
        with raw.open() as f:
            for i,r in enumerate(csv.DictReader(f)):
                if i not in selected:continue
                a,b,label,_=map(int,selected[i])
                assert sequences[a]==r['query'].strip() and sequences[b]==r['text'].strip() and label==int(r['label'])
                verified+=1
        assert verified==len(data);source_records.append(record(raw))
    index={s:i for i,s in enumerate(meta['sequence'])};union=np.load(ROOT/'data/union.npy')
    mapping=load_npz(ROOT/'data/pair-mapping.npz');old=load_npz(ROOT/'provenance/exposure-flags.npz')
    parent=old['native-plm-public-source__exact__endpoints']>0
    audits={};flags={}
    for name,cap in [('v2-capped',2193),('v2-clean-bce',None)]:
        train=rows['train']
        if cap is not None:
            length=np.diff(offsets);train=train[length[train[:,0]]+length[train[:,1]]<=cap]
        parts={'train':train,'val':rows['val']};seen=np.zeros(len(index),dtype=bool);known=set();part_counts={}
        for split,data in parts.items():
            proteins=set(data[:,:2].ravel().tolist())
            for protein in proteins:
                if sequences[protein] in index:seen[index[sequences[protein]]]=True
            for a,b,_,_ in data:
                aa=index.get(sequences[a]);bb=index.get(sequences[b])
                if aa is not None and bb is not None:known.add(tuple(sorted([aa,bb])))
            part_counts[split]={'pairs':len(data),'positives':int(data[:,2].sum()),'proteins':len(proteins)}
        assert not (seen&~parent).any(),'Existing conservative Bernett mask must cover every retained v2 source'
        pair_seen=np.array([tuple(r[:2]) in known for r in union],dtype=bool)
        tests={}
        for test in ['mouse','fly','worm','yeast','ecoli']:
            r=np.load(ROOT/'data'/f'{test}.npy');p=pair_seen[mapping[test]];end=seen[r[:,0]]|seen[r[:,1]]
            tests[test]={'exact_pair_rows':int(p.sum()),'positive_pair_rows':int((p&(r[:,2]==1)).sum()),
                'rows_with_exposed_endpoint':int(end.sum()),'common_parent_mask_is_conservative':True}
        audits[name]={'parts':part_counts,'exact_shared_test_sequences':int(seen.sum()),'tests':tests}
        flags[name+'__endpoints']=seen;flags[name+'__union_pairs']=pair_seen
    save_npz(ext/'provenance/exposure-flags.npz',**flags)
    atomic(ext/'provenance/exposure.json',{'at_utc':now(),'audits':audits,'prepared_manifest':record(source/'manifest.json'),
        'source_files':source_records,'every_retained_source_row_and_tokenized_sequence_verified':True,
        'existing_common_human_source_mask_covers_both_models':True,'homology_exclusion':False,
        'test_predictions_used':False,'script':record(__file__)})
    print(audits,flush=True)


if __name__=='__main__':main()
