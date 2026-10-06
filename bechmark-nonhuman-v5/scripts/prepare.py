"""Freeze existing human-trained checkpoints and released five-species tests."""
import collections,csv,hashlib,os
from pathlib import Path
import numpy as np
from bench_utils import ROOT,PROJECT,EXTERNAL,atomic,now,read,record,save_npz,sha
SPECIES=['mouse','fly','worm','yeast','ecoli']

def main():
 assert not (ROOT/'provenance/prepared.json').exists()
 src=PROJECT/'plm-interact-reproducability';catalog=read(src/'provenance/hf-tree-cross_species_benchmarking.json')
 expected={x['path']:x['lfs']['oid'] for x in catalog if 'lfs' in x};allseq=set();raw={};sources={}
 for species in SPECIES:
  rel=f'test/{species}.ppi.qrels.seq.test.csv';p=src/'data/cross_species_benchmarking'/rel;assert sha(p)==expected[rel]
  with p.open() as f:rows=list(csv.DictReader(f))
  assert len(rows)==(22000 if species=='ecoli' else 55000)
  assert sum(int(r['label']) for r in rows)==(2000 if species=='ecoli' else 5000)
  for r in rows:
   assert r['label'] in ['0','1']
   for k in ['query','text']:
    s=r[k];assert s==s.strip() and s and set(s)<=set('ACDEFGHIKLMNPQRSTVWYBXZUO.-');allseq.add(s)
  raw[species]=rows;sources[species]=record(p)
 sequences=sorted(allseq,key=lambda s:hashlib.sha256(s.encode()).hexdigest());hashes=[hashlib.sha256(s.encode()).hexdigest() for s in sequences];index={s:i for i,s in enumerate(sequences)}
 vocab=read(PROJECT/'retrain-v5/data/prepared/esm2/tokenizer-contract.json')['vocabulary'];vc=read(PROJECT/'retrain-v5/data/prepared/esmc/tokenizer-contract.json')['vocabulary']
 assert all(vocab[a]==vc[a] for a in set(''.join(sequences)))
 tokens=np.concatenate([np.array([vocab[a] for a in s],np.int16) for s in sequences]);offsets=np.r_[0,np.cumsum(list(map(len,sequences)),dtype=np.int64)]
 union=[];lookup={};mapping={};census={};anomalies=[]
 for species in SPECIES:
  arr=[];ids=[];reverse=[];seen=collections.defaultdict(list)
  for i,r in enumerate(raw[species]):
   a,b=index[r['query']],index[r['text']];label=int(r['label']);key=tuple(sorted([a,b]))
   if key not in lookup:lookup[key]=len(union);union.append([*key,0,len(union)])
   j=lookup[key];arr.append([a,b,label,i]);ids.append(j);reverse.append(a>b);seen[key].append((i,label))
  arr=np.asarray(arr,np.int64);np.save(ROOT/'data'/f'{species}.npy',arr);mapping[species]=np.array(ids,np.int64);mapping[species+'_reversed']=np.array(reverse,bool)
  conflicting=[k for k,v in seen.items() if len({x[1] for x in v})>1]
  for k in conflicting:anomalies.append({'species':species,'sequence_indices':list(k),'source_rows_and_labels':seen[k]})
  used=np.unique(arr[:,:2]);lengths=np.diff(offsets)
  census[species]={'rows':len(arr),'positives':int(arr[:,2].sum()),'negatives':int((arr[:,2]==0).sum()),'unique_sequences':len(used),
   'unique_unordered_sequence_pairs':len(seen),'duplicate_pair_rows':len(arr)-len(seen),'conflicting_label_pairs':len(conflicting),
   'rows_in_conflicting_label_pairs':sum(len(seen[k]) for k in conflicting),'self_sequence_pairs':int((arr[:,0]==arr[:,1]).sum()),
   'max_protein_residues':int(lengths[used].max()),'max_combined_residues':int((lengths[arr[:,0]]+lengths[arr[:,1]]).max())}
 union=np.array(union,np.int64)
 for name,arr in [('union',union),('tokens',tokens),('offsets',offsets)]:np.save(ROOT/'data'/f'{name}.npy',arr)
 save_npz(ROOT/'data/pair-mapping.npz',**mapping)
 atomic(ROOT/'data/sequences.json',{'sha256':hashes,'sequence':sequences,'length':list(map(len,sequences)),'identity':'Exact released sequence strings; no current-accession substitution.'})
 (ROOT/'data/test-sequences.fasta').write_text(''.join(f'>p{i:05d}\n{s}\n' for i,s in enumerate(sequences)));atomic(ROOT/'provenance/label-conflicts.json',anomalies)
 old=read(PROJECT/'benchmark-v5/provenance/selection.json');selection={k:v for k,v in old.items() if k not in ['native_reuse','models']}
 selection.update(at_utc=now(),models={},selection_source=record(PROJECT/'benchmark-v5/provenance/selection.json'),frozen_for='nonhuman inference only; no training, recalibration, or re-selection')
 for name,item in old['models'].items():
  source=Path(item['checkpoint']['path']);assert sha(source)==item['checkpoint']['sha256'];target=ROOT/'checkpoints'/source.name
  if not target.exists():os.link(source,target)
  item['checkpoint']=record(target);selection['models'][name]=item
 atomic(ROOT/'provenance/selection.json',selection);cache_audit={}
 for name,source,dir_ in [('tuna',EXTERNAL/'benchmark/tuna/data/sequences.json',None),('xpair',EXTERNAL/'experiments/x_pair_test2_v1/data/sequences.json',EXTERNAL/'experiments/x_pair_test2_v1/features')]:
  previous=read(source);lookup_={h:i for i,h in enumerate(previous['sha256'])};mapped=np.array([lookup_.get(h,-1) for h in hashes],np.int64)
  if dir_ is not None:
   for i,j in enumerate(mapped):
    if j>=0 and not ((dir_/f'{j:05d}.json').exists() and (dir_/f'{j:05d}.pt').exists()):mapped[i]=-1
  np.save(ROOT/'data'/f'{name}-cache-indices.npy',mapped);cache_audit[name]={'source':record(source),'matches':int((mapped>=0).sum()),'missing':int((mapped<0).sum())}
 report={'at_utc':now(),'species':SPECIES,'tests':census,'source_files':sources,'source_catalog':record(src/'provenance/hf-tree-cross_species_benchmarking.json'),
  'source_dataset_metadata':record(src/'provenance/hf-cross_species_benchmarking.json'),'union_pairs':len(union),'proteins':len(sequences),'residues':int(offsets[-1]),
  'source_rows':sum(c['rows'] for c in census.values()),'union_label_column':'Dummy zeros; per-species labels remain authoritative, including conflicts.',
  'cache_candidates':cache_audit,'data_files':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'data').iterdir()) if p.is_file()},'selection_sha256':sha(ROOT/'provenance/selection.json')}
 atomic(ROOT/'provenance/prepared.json',report);print({k:report[k] for k in ['union_pairs','proteins','residues','source_rows','tests','cache_candidates']},flush=True)

if __name__=='__main__':main()
