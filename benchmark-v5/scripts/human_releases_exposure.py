"""Exact public TRAIN/validation membership for both added human releases."""
import csv,hashlib
from collections import Counter
import numpy as np
from bench_utils import ROOT,PROJECT,EXTERNAL,atomic,load_npz,now,read,record,save_npz

def main():
    meta=read(ROOT/'data/sequences.json');union=np.load(ROOT/'data/union.npy');mapping=load_npz(ROOT/'data/pair-mapping.npz')
    lookup={s:i for i,s in enumerate(meta['sha256'])};pairs={tuple(sorted(map(int,r[:2]))):i for i,r in enumerate(union)}
    endpoints=np.zeros(len(lookup),np.uint8);hit=np.zeros(len(union),np.uint8)
    source_positive=hit.copy();source_negative=hit.copy();sources=[]
    for split,bit in [('train',1),('test',2)]:
        csvpath=PROJECT/'bechmark-v5-nonhuman/data/exposure'/f'human.ppi.qrels.seq.{split}.csv'
        source=Counter();memo={};n=positive=0
        def digest(s):
            if s not in memo:memo[s]=hashlib.sha256(s.encode()).hexdigest()
            return memo[s]
        with csvpath.open() as f:
            for row in csv.DictReader(f):
                hs=[digest(row['query']),digest(row['text'])];label=int(row['label']);source[(*sorted(hs),label)]+=1;n+=1;positive+=label
                ids=[lookup.get(h) for h in hs]
                for i in ids:
                    if i is not None:endpoints[i]|=bit
                if all(i is not None for i in ids):
                    j=pairs.get(tuple(sorted(ids)))
                    if j is not None:
                        hit[j]|=bit
                        (source_positive if label else source_negative)[j]|=bit
        folder=EXTERNAL/'benchmark/tuna/upstream/TUnA/data/processed/xspecies'
        dictionary=folder/f'human_{split}_dictionary.tsv';interactions=folder/f'human_{split}_interaction.tsv'
        seq={r[0]:digest(r[1]) for r in csv.reader(dictionary.open(),delimiter='\t')}
        other=Counter((*sorted((seq[a],seq[b])),int(label)) for a,b,label in csv.reader(interactions.open(),delimiter='\t'))
        assert source==other,'Native and TUnA public sources differ'
        sources.append({'split':'train' if bit==1 else 'validation','bit':bit,'native_file':record(csvpath),
            'tuna_dictionary':record(dictionary),'tuna_interactions':record(interactions),'rows':n,'positives':positive,
            'negatives':n-positive,'unordered_sequence_pair_label_multisets_identical':True})
    tests={}
    for test in ['original','ilp']:
        ids=mapping[test];rows=np.load(ROOT/'data'/f'{test}.npy');y=rows[:,2]
        tests[test]={}
        for scope,mask in [('train',1),('validation',2),('train_or_validation',3)]:
            ep=(endpoints&mask)>0;pp=(hit[ids]&mask)>0;remaining=~ep[rows[:,:2]].any(1)
            tests[test][scope]={'exact_exposed_sequences':int(ep[np.unique(rows[:,:2])].sum()),
                'pairs_exposed':int(pp.sum()),'positive_pairs_exposed':int((pp&(y==1)).sum()),'negative_pairs_exposed':int((pp&(y==0)).sum()),
                'pairs_with_exposed_endpoint':int((~remaining).sum()),'endpoint_unexposed_pairs':int(remaining.sum()),
                'endpoint_unexposed_positives':int(y[remaining].sum()),
                'test_negative_but_source_positive':int((((source_positive[ids]&mask)>0)&(y==0)).sum()),
                'test_positive_but_source_negative':int((((source_negative[ids]&mask)>0)&(y==1)).sum())}
    flags=ROOT/'provenance/human-releases-exposure-flags.npz'
    previous=load_npz(ROOT/'provenance/exposure-flags.npz')
    same_endpoints=bool(np.array_equal(endpoints>0,previous['dscript__exact__endpoints']>0))
    assert same_endpoints
    save_npz(flags,endpoints=endpoints,pairs=hit,source_positive=source_positive,source_negative=source_negative)
    atomic(ROOT/'provenance/human-releases-exposure.json',{'at_utc':now(),'models':['native-human','tuna-human'],
        'sources':sources,'source_datasets_identical':True,'endpoint_mask_identical_to_dscript_train':same_endpoints,
        'exact_union_sequences_exposed':int((endpoints>0).sum()),
        'tests':tests,'flags':record(flags),'script':record(__file__),
        'limitations':'Public supervised TRAIN/validation sources; not proof of full checkpoint history. Exact matching only; homologs, fragments, isoforms and PLM pretraining are not excluded.'})
    print({'exposed_sequences':int((endpoints>0).sum()),'tests':tests},flush=True)
if __name__=='__main__':main()
