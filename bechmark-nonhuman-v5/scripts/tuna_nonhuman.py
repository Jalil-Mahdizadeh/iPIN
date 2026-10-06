"""Two original TUnA releases, shared ESM forwards, qualified compact features."""
import argparse,fcntl,os,sys,time
import numpy as np
import torch
from scipy.special import expit
from bench_utils import ROOT,EXTERNAL,atomic,cuda,load_npz,now,read,record,save_npz,sha
SOURCE=EXTERNAL/'benchmark/tuna'
os.environ['TUNA_UPSTREAM_DIR']=str(SOURCE/'upstream/TUnA/results/bernett/TUnA')
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/tuna/matplotlib')
sys.path[:0]=['/opt/tuna/vendor',str(ROOT/'scripts/tuna')]
from adapter import create,endpoint_features,cached_scores,native_scores,enable_sdpa
from esm_cache import load_encoder,enable_fast,embed
NAMES={'tuna':(SOURCE/'weights/bernett_original.pt',64,256,'bf2dd42af75d98324798b76837ca738e4ed230b096ee538ccdbaa001dcc88500'),
 'tuna-human':(ROOT/'checkpoints/tuna-human-seed47.pt',256,1024,'f63876dfdfb8a8a2c855bbb814a74b20ccf333dc8c75be00bf281365fcac6c07')}

def signature():
 return {str(p):sha(ROOT/p) for p in ['scripts/tuna_nonhuman.py','scripts/tuna/adapter.py','scripts/tuna/esm_cache.py','scripts/container.sh','provenance/prepared.json','provenance/runtime-inputs.json']}
def models(device):
 out={}
 for name,(path,hid,ff,digest) in NAMES.items():
  assert sha(path)==digest
  out[name]=create(checkpoint=path,device=device,hid_dim=hid,ff_dim=ff).eval().requires_grad_(False)
 return out

@torch.inference_mode()
def embedding(device,rank,world):
 assert world==4
 started=time.monotonic();meta=read(ROOT/'data/sequences.json');lengths=np.array(meta['length']);seq=meta['sequence'];models_=models(device)
 initial={n:{k:v.clone() for k,v in m.state_dict().items()} for n,m in models_.items()}
 enc,alphabet,_=load_encoder(EXTERNAL/'.private/frozen_pair_models_v1/bundle/encoder',device);enc.requires_grad_(False)
 order=np.argsort(lengths,kind='stable');fixtures=[int(order[i]) for i in [0,len(order)//4,len(order)//2,-1]]
 native=embed(enc,alphabet,[seq[fixtures[1]]])[0];enable_fast(enc);fast=embed(enc,alphabet,[seq[fixtures[1]]])[0]
 encoder_error=float((native-fast).abs().max());assert encoder_error<2e-4,encoder_error
 emb={i:embed(enc,alphabet,[seq[i]])[0] for i in fixtures};pairs=[(fixtures[0],fixtures[1]),(fixtures[2],fixtures[-1]),(fixtures[-1],fixtures[1])]
 cases={}
 for name,m in models_.items():
  gold=native_scores(m,[emb[a] for a,b in pairs],[emb[b] for a,b in pairs]).cpu().numpy();enable_sdpa(m)
  matrix=torch.stack([endpoint_features(m,emb[i][None],torch.tensor([len(emb[i])],device=device))[0] for i in fixtures])
  aa=np.array([fixtures.index(a) for a,b in pairs]);bb=np.array([fixtures.index(b) for a,b in pairs])
  p=cached_scores(m,matrix,aa,bb,probabilities=True);reverse=cached_scores(m,matrix,bb,aa,probabilities=True)
  err=float(abs(gold-p).max());assert err<2e-5 and np.array_equal(p,reverse),(name,err)
  width=max(len(emb[i]) for i in fixtures);padded=torch.zeros(len(fixtures),width,640,device=device)
  for j,i in enumerate(fixtures):padded[j,:len(emb[i])]=emb[i]
  batch=endpoint_features(m,padded,torch.tensor([len(emb[i]) for i in fixtures],device=device));be=float((matrix-batch).abs().max());assert be<2e-4,(name,be)
  cases[name]={'native_probability_error':err,'feature_padding_error':be,'checkpoint':record(NAMES[name][0]),'hid_dim':NAMES[name][1]}
 # Verify any endpoint feature reuse directly against frozen Bernett evidence.
 bundle=SOURCE/'runs/scorer_bundle';freeze=read(bundle/'SCORER_FREEZE.json')
 for rel in ['endpoints.json','features/tuna_original.npy','weights/tuna_original.pt']:
  item=next(x for x in freeze['files'] if x['path']==rel);assert sha(bundle/rel)==item['sha256']
 mapping=np.load(ROOT/'data/tuna-cache-indices.npy');old=np.load(bundle/'features/tuna_original.npy');endpoints=read(bundle/'endpoints.json')
 matched=np.flatnonzero(mapping>=0);reuse_checks=[]
 if len(matched):
  for i in [int(matched[0]),int(matched[-1])]:
   assert meta['sha256'][i]==endpoints[mapping[i]]
   e=embed(enc,alphabet,[seq[i]])[0];z=endpoint_features(models_['tuna'],e[None],torch.tensor([len(e)],device=device))[0].cpu().numpy()
   error=float(abs(z-old[mapping[i]]).max());assert error<2e-4;reuse_checks.append({'id':i,'error':error})
 atomic(ROOT/'qualification'/f'tuna-rank-{rank:02d}.json',{'passed':True,'signature':signature(),'models':cases,'encoder_error':encoder_error,'fixtures':fixtures,'reuse_checks':reuse_checks,'at_utc':now()})
 assigned=np.arange(len(seq))[rank::world];assigned=assigned[np.argsort(lengths[assigned],kind='stable')];out=ROOT/'features/tuna';out.mkdir(exist_ok=True)
 lock=(out/f'rank-{rank}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);files=[]
 for start in range(0,len(assigned),512):
  ids=assigned[start:start+512];path=out/f'rank-{rank:02d}-{start:06d}.npz';side=path.with_suffix('.json')
  if side.exists():
   saved=read(side);assert saved['signature']==signature() and sha(path)==saved['file']['sha256'];assert np.array_equal(load_npz(path)['indices'],ids)
  else:
   features={name:np.empty((len(ids),spec[1]),np.float32) for name,spec in NAMES.items()}
   for begin in range(0,len(ids),8):
    b=ids[begin:begin+8];embs=embed(enc,alphabet,[seq[i] for i in b]);width=max(map(len,embs));values=torch.zeros(len(b),width,640,device=device)
    for j,e in enumerate(embs):values[j,:len(e)]=e
    sizes=torch.tensor([len(e) for e in embs],device=device)
    for name,m in models_.items():features[name][begin:begin+len(b)]=endpoint_features(m,values,sizes).cpu().numpy()
   # Reuse only the numerically qualified, identical Bernett endpoint features.
   use=mapping[ids]>=0;features['tuna'][use]=old[mapping[ids][use]]
   assert all(np.isfinite(x).all() for x in features.values());save_npz(path,indices=ids,**features);atomic(side,{'file':record(path),'signature':signature()})
  files.append(record(path));print({'stage':'tuna-features','rank':rank,'rows':start+len(ids),'total':len(assigned),'seconds':time.monotonic()-started},flush=True)
 for n,m in models_.items():assert all(torch.equal(v,initial[n][k]) for k,v in m.state_dict().items()),n
 atomic(out/f'rank-{rank:02d}.done.json',{'passed':True,'rank':rank,'world':world,'files':files,'signature':signature(),'state_unchanged':True,'at_utc':now()})

@torch.inference_mode()
def score(device):
 meta=read(ROOT/'data/sequences.json');rows=np.load(ROOT/'data/union.npy');features={n:np.empty((len(meta['sequence']),s[1]),np.float32) for n,s in NAMES.items()};seen=np.zeros(len(meta['sequence']),np.int8);qs=[]
 for rank in range(4):
  q=read(ROOT/'qualification'/f'tuna-rank-{rank:02d}.json');assert q['passed'] and q['signature']==signature();qs.append(q)
  d=read(ROOT/'features/tuna'/f'rank-{rank:02d}.done.json');assert d['passed'] and d['signature']==signature()
  for item in d['files']:
   assert sha(item['path'])==item['sha256'];a=load_npz(item['path']);ids=a['indices'];assert not seen[ids].any();seen[ids]+=1
   for name in NAMES:features[name][ids]=a[name]
 assert (seen==1).all();atomic(ROOT/'qualification/tuna.json',{'passed':True,'signature':signature(),'workers':qs})
 for name,model in models(device).items():
  enable_sdpa(model);scores=cached_scores(model,torch.tensor(features[name],device=device),rows[:,0],rows[:,1],probabilities=False).astype(np.float64)
  out=ROOT/'predictions'/name;out.mkdir(exist_ok=True);save_npz(out/'union.npz',scores=scores,probabilities=expit(scores));atomic(out/'done.json',{'rows':len(rows),'file':record(out/'union.npz'),'checkpoint':record(NAMES[name][0]),'at_utc':now(),'signature':signature(),'score_kind':'Native uncertainty-adjusted logit; no refit'})
 print('Both frozen TUnA releases complete',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['embedding','score'],required=True);a=p.parse_args();device=cuda(int(os.environ.get('LOCAL_RANK',0)))
 if a.stage=='embedding':embedding(device,int(os.environ.get('RANK',0)),int(os.environ.get('WORLD_SIZE',1)))
 else:score(device)
