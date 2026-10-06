"""Register the user-approved variants before inference; fixed release, fixed seed."""
import csv
import numpy as np
from bench_utils import ROOT,PROJECT,atomic,read,record,sha,save_npz,now

def main():
 assert not (ROOT/'provenance/roster.json').exists()
 selected=read(ROOT/'provenance/selection.json');path=ROOT/'checkpoints/native-human.bin'
 assert sha(path)=='68c50e1dc84ee3cb6c08a7c83eefb382a29a3c1237fd577986854f746c63d665'
 selected['models']['native-human']={'checkpoint':record(path),'base':selected['models']['native-plm']['base'],
  'release':'danliu1226/PLM-interact-650M-humanV11','revision':'e86e392dec13dd0c23252c94947b04a7a9821b0e','source':'Verified existing SIF asset and prior reproduction'}
 atomic(ROOT/'provenance/selection.json',selected)
 prepared=read(ROOT/'provenance/prepared.json');mapping=np.load(ROOT/'data/pair-mapping.npz');indices=[];columns=[];logits=[];records=[]
 for species in prepared['species']:
  p=PROJECT/'plm-interact-reproducability/results'/f'cross_{species}.csv'
  with p.open() as f:rows=list(csv.DictReader(f))
  target=np.load(ROOT/'data'/f'{species}.npy');meta=read(ROOT/'data/sequences.json');lengths=np.array(meta['length'])
  assert len(rows)==len(target)
  assert all(int(r['row_id'])==i and int(r['label'])==target[i,2] for i,r in enumerate(rows))
  chosen=sorted({0,len(rows)//2,int(np.argmax(lengths[target[:,0]]+lengths[target[:,1]]))})
  for i in chosen:
   indices.append(mapping[species][i]);columns.append(int(mapping[species+'_reversed'][i]));logits.append(float(rows[i]['logit']))
  records.append(record(p))
 save_npz(ROOT/'data/native-human-qualification.npz',indices=np.array(indices),columns=np.array(columns),logits=np.array(logits))
 atomic(ROOT/'provenance/native-human-reference.json',{'files':records,'qualification_rows':len(indices),'description':'Prior independently reproduced original-order FP32 logits; every source row retained.'})
 prepared['selection_sha256']=sha(ROOT/'provenance/selection.json');prepared['data_files']['data/native-human-qualification.npz']=sha(ROOT/'data/native-human-qualification.npz')
 atomic(ROOT/'provenance/prepared.json',prepared)
 runtime=read(ROOT/'provenance/runtime-inputs.json');human=read(ROOT/'provenance/tuna-human-download.json');runtime['items']['tuna-human']=human
 atomic(ROOT/'provenance/runtime-inputs.json',runtime)
 roster=[('ipin-esm2','iPIN v5 ESM2'),('ipin-esmc','iPIN v5 ESMC'),('native-human','Native PLM-interact (humanV11)'),
  ('native-plm','Native PLM-interact (Bernett)'),('tuna-human','TUnA (human, seed 47)'),('tuna','TUnA (Bernett)'),
  ('xpair-bernett','X-PAIR (Bernett)'),('xpair-default','X-PAIR (default)'),('rapppid','RAPPPID (released mult)'),('sprint','SPRINT (v5 human TRAIN graph)'),('dscript','D-SCRIPT (human_v1)')]
 atomic(ROOT/'provenance/roster.json',{'at_utc':now(),'models':dict(roster),'user_choice':'Human cross-species releases and prior Bernett checkpoints separately labeled',
  'primary_native_reference':'native-human','secondary_native_reference':'native-plm','tuna_human_seed':47,'selection_uses_target_test_metrics':False})
 print('Frozen roster:',list(dict(roster)),flush=True)

if __name__=='__main__':main()
