"""Strict coverage and byte checks before releasing benchmark scores to analysis."""
import argparse
from pathlib import Path
import numpy as np
from scipy.special import expit
from bench_utils import ROOT,atomic,load_npz,now,read,record,save_npz,sha
NAMES=['native-plm','ipin-esm2','ipin-esmc','tuna','xpair-bernett','xpair-default','rapppid','sprint','dscript','native-human','tuna-human']
LABELS={'native-plm':'PLM-interact (Bernett)','ipin-esm2':'iPIN v5 ESM2','ipin-esmc':'iPIN v5 ESMC','tuna':'TUnA (Bernett)',
 'xpair-bernett':'X-PAIR (Bernett)','xpair-default':'X-PAIR (default)*','rapppid':'RAPPPID (released mult)','sprint':'SPRINT (v5 TRAIN graph)','dscript':'D-SCRIPT (original)*',
 'native-human':'PLM-interact (humanV11)*','tuna-human':'TUnA (human, seed 47)*'}
def verify(item):
 p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p);return p

def collect(name):
 n=len(np.load(ROOT/'data/union.npy'));directory=ROOT/'predictions'/name;sources=[]
 if name in ['native-plm','ipin-esm2','ipin-esmc','native-human']:
  if not all((directory/f'rank-{r:02d}.done.json').exists() for r in range(4)):return None
  logits=np.full((n,2),np.nan);seen=np.zeros(n,np.int8);q=read(ROOT/'qualification'/(name+'.json'));assert q['passed']
  if name=='native-human':
   from native_human_bernett import signature
   assert q['fingerprint']==signature()
  if name=='native-plm':
   gold=load_npz(ROOT/'data/native-original-reused.npz')['predictions'];assert gold.shape==(52048,4)
   assert np.array_equal(gold[:,0],np.arange(52048)) and np.array_equal(gold[:,1],np.load(ROOT/'data/union.npy')[:52048,2])
   logits[:52048]=gold[:,2:4];seen[:52048]=1;sources.append(record(ROOT/'data/native-original-reused.npz'))
  ranks=[]
  for rank in range(4):
   path=directory/f'rank-{rank:02d}.done.json';m=read(path);ranks.append(m);assert m['rank']==rank and m['world']==4 and m['fingerprint']==q['fingerprint'];count=0
   for entry in m['chunks']:
    side=read(verify(entry));assert side['fingerprint']==q['fingerprint'];arr=load_npz(verify(side['file']));ids=arr['indices'];z=arr['logits']
    assert z.shape==(len(ids),2) and len(np.unique(ids))==len(ids) and not seen[ids].any();seen[ids]+=1;logits[ids]=z;count+=len(ids)
   assert count==m['rows'];sources.append(record(path))
  assert len({r['gpu_uuid'] for r in ranks})==4 and len({r['hostname'] for r in ranks})==1
  assert (seen==1).all() and np.isfinite(logits).all();scores=logits.mean(1);data={'scores':scores,'probabilities':expit(scores),'logits':logits}
 elif name.startswith('xpair') or name=='dscript':
  if not all((directory/f'rank-{r:02d}.done.json').exists() for r in range(4)):return None
  q=read(ROOT/'qualification'/('dscript.json' if name=='dscript' else 'xpair.json'));assert q['passed'];scores=np.full(n,np.nan);seen=np.zeros(n,np.int8)
  for rank in range(4):
   path=directory/f'rank-{rank:02d}.done.json';m=read(path);assert m['rank']==rank and m['world']==4 and m['signature']==q['signature'];count=0
   for item in m['files']:
    file=verify(item);side=read(file.with_suffix('.json'));assert side['signature']==q['signature'];arr=load_npz(file);ids=arr['indices'];z=arr['scores']
    assert z.shape==(len(ids),) and len(np.unique(ids))==len(ids) and not seen[ids].any();seen[ids]+=1;scores[ids]=z;count+=len(ids)
   assert count==m['rows'];sources.append(record(path))
  assert (seen==1).all() and np.isfinite(scores).all();data={'scores':scores,'probabilities':scores.copy() if name=='dscript' else expit(scores)}
 else:
  path=directory/'done.json'
  if not path.exists():return None
  m=read(path);assert m['rows']==n;data=load_npz(verify(m['file']));sources.append(record(path))
  if name=='tuna-human':
   q=read(verify(m['qualification']));assert q['passed'] and q['fingerprint']==m['fingerprint'] and m['state_unchanged']
   manifest=read(ROOT/'provenance/human-releases.json');assert m['checkpoint']==manifest['models'][name]['checkpoint'];verify(m['checkpoint'])
   for rel,h in manifest['inference_code'].items():assert sha(ROOT/rel)==h
 for k,v in data.items():assert v.shape[0]==n and np.isfinite(v).all()
 if 'probabilities' in data:assert (data['probabilities']>=0).all() and (data['probabilities']<=1).all()
 out=ROOT/'results'/(name+'-union.npz')
 if out.exists():
  previous=load_npz(out);assert set(previous)==set(data) and all(np.array_equal(previous[k],v) for k,v in data.items()),name
 else:save_npz(out,**data)
 return {'rows':n,'file':record(out),'sources':sources,'score_kind':'raw native SPRINT score' if name=='sprint' else 'probability' if name in ['rapppid','dscript'] else 'logit','probability_available':'probabilities' in data}

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--available',action='store_true');args=parser.parse_args()
 result={};missing=[]
 for name in NAMES:
  record_=collect(name)
  if record_ is None:missing.append(name)
  else:result[name]=record_
 if missing and not args.available:raise RuntimeError(f'Incomplete predictors: {missing}')
 mapping=load_npz(ROOT/'data/pair-mapping.npz');union=np.load(ROOT/'data/union.npy')
 for test in ['original','ilp']:
  rows=np.load(ROOT/'data'/f'{test}.npy');ids=mapping[test]
  assert len(rows)==52048 and len(np.unique(ids))==52048 and np.array_equal(rows[:,2],union[ids,2])
  assert np.array_equal(np.sort(rows[:,:2],axis=1),np.sort(union[ids,:2],axis=1))
 shared=np.intersect1d(mapping['original'],mapping['ilp']);assert len(shared)==27178 and int(union[shared,2].sum())==26024
 atomic(ROOT/'results/collection.json',{'at_utc':now(),'complete':not missing,'missing':missing,'models':result,'shared_pairs':len(shared),'prepared':record(ROOT/'provenance/prepared.json')})
 print({'complete':list(result),'missing':missing},flush=True)
if __name__=='__main__':main()
