"""Strict coverage and byte checks before releasing benchmark scores to analysis."""
import argparse
from pathlib import Path
import numpy as np
from scipy.special import expit
from bench_utils import ROOT,atomic,load_npz,now,read,record,save_npz,sha
ROSTER=read(ROOT/'provenance/roster.json')
LABELS=ROSTER['models'];NAMES=list(LABELS)
TESTS=['mouse','fly','worm','yeast','ecoli']
PAIR_MODELS=['native-human','native-plm','ipin-esm2','ipin-esmc']
def verify(item):
 p=Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],str(p);return p

def collect(name):
 n=len(np.load(ROOT/'data/union.npy'));directory=ROOT/'predictions'/name;sources=[]
 if name in PAIR_MODELS:
  if not all((directory/f'rank-{r:02d}.done.json').exists() for r in range(4)):return None
  logits=np.full((n,2),np.nan);seen=np.zeros(n,np.int8);q=read(ROOT/'qualification'/(name+'.json'));assert q['passed']
  ranks=[]
  for rank in range(4):
   path=directory/f'rank-{rank:02d}.done.json';m=read(path);ranks.append(m);assert m['rank']==rank and m['world']==4 and m['fingerprint']==q['fingerprint'];count=0
   for entry in m['chunks']:
    side=read(verify(entry));assert side['fingerprint']==q['fingerprint'];arr=load_npz(verify(side['file']));ids=arr['indices'];z=arr['logits']
    assert z.shape==(len(ids),2) and len(np.unique(ids))==len(ids) and not seen[ids].any();seen[ids]+=1;logits[ids]=z;count+=len(ids)
   assert count==m['rows'];sources.append(record(path))
  assert len({r['gpu_uuid'] for r in ranks})==4
  sources.append(record(ROOT/'qualification'/(name+'.json')))
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
  if name.startswith('tuna'):
   q=read(ROOT/'qualification/tuna.json');assert q['passed'] and q['signature']==m['signature']
   for p,h in m['signature'].items():assert sha(ROOT/p)==h
  elif name=='rapppid':assert read(verify(m['qualification']))['passed'] and m['state_unchanged']
  elif name=='sprint':assert read(ROOT/'qualification/sprint-requested/qualification.json')['passed']
 for k,v in data.items():assert v.shape[0]==n and np.isfinite(v).all()
 if 'probabilities' in data:assert (data['probabilities']>=0).all() and (data['probabilities']<=1).all()
 out=ROOT/'results'/(name+'-union.npz');save_npz(out,**data)
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
 counts={}
 for test in TESTS:
  rows=np.load(ROOT/'data'/f'{test}.npy');ids=mapping[test]
  assert len(rows)==len(ids) and np.array_equal(rows[:,3],np.arange(len(rows)))
  assert np.array_equal(np.sort(rows[:,:2],axis=1),union[ids,:2])
  assert set(np.unique(rows[:,2]))=={0,1}
  counts[test]={'rows':len(rows),'positives':int(rows[:,2].sum()),'unique_pairs':len(np.unique(ids))}
 atomic(ROOT/'results/collection.json',{'at_utc':now(),'complete':not missing,'missing':missing,'models':result,'tests':counts,
  'roster':record(ROOT/'provenance/roster.json'),'prepared':record(ROOT/'provenance/prepared.json')})
 print({'complete':list(result),'missing':missing},flush=True)
if __name__=='__main__':main()
