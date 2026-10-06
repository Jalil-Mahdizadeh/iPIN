"""Released TUnA, exact-sequence feature reuse and native singleton oracle."""
import os,sys,time
import numpy as np
import torch
from scipy.special import expit
from bench_utils import ROOT,EXTERNAL,atomic,cuda,now,read,record,save_npz,sha
SOURCE=EXTERNAL/'benchmark/tuna'
os.environ['TUNA_UPSTREAM_DIR']=str(SOURCE/'upstream/TUnA/results/bernett/TUnA')
os.environ['MPLCONFIGDIR']=str(ROOT/'cache/tuna/matplotlib')
sys.path.insert(0,'/opt/tuna/vendor')
sys.path.insert(0,str(ROOT/'scripts/tuna'))
from adapter import load_original,endpoint_features,cached_scores,native_scores,enable_sdpa
from esm_cache import load_encoder,enable_fast,embed

@torch.inference_mode()
def main():
 started=time.monotonic();device=cuda();meta=read(ROOT/'data/sequences.json');rows=np.load(ROOT/'data/union.npy')
 bundle=SOURCE/'runs/scorer_bundle';freeze=read(bundle/'SCORER_FREEZE.json')
 for rel in ['endpoints.json','features/tuna_original.npy','weights/tuna_original.pt']:
  item=next(x for x in freeze['files'] if x['path']==rel);assert sha(bundle/rel)==item['sha256']
 assert freeze['original_checkpoint_sha256']==sha(SOURCE/'weights/bernett_original.pt')
 endpoints=read(bundle/'endpoints.json');oldseq=read(SOURCE/'data/sequences.json')
 assert endpoints==oldseq['sha256']
 mapping=np.load(ROOT/'data/tuna-cache-indices.npy');old=np.load(bundle/'features/tuna_original.npy')
 assert old.shape==(len(endpoints),64)
 for i,j in enumerate(mapping):
  if j>=0:assert meta['sha256'][i]==endpoints[j]
 model=load_original(SOURCE/'weights/bernett_original.pt',device)
 before={n:p.clone() for n,p in model.named_parameters()}
 encoder,alphabet,ignored=load_encoder(EXTERNAL/'.private/frozen_pair_models_v1/bundle/encoder',device)
 matched=np.flatnonzero(mapping>=0);order=matched[np.argsort(np.array(meta['length'])[matched],kind='stable')]
 fixtures=[int(order[j]) for j in [0,len(order)//2,int(.9*len(order)),-1]]
 # Independently recompute full FP32 features; compare native and SDPA attention.
 native_emb=embed(encoder,alphabet,[meta['sequence'][fixtures[1]]])[0]
 enable_fast(encoder)
 fast_emb=embed(encoder,alphabet,[meta['sequence'][fixtures[1]]])[0]
 encoder_error=float((native_emb-fast_emb).abs().max());assert encoder_error<2e-4,encoder_error
 emb={i:embed(encoder,alphabet,[meta['sequence'][i]])[0] for i in fixtures}
 reference_feature={i:endpoint_features(model,e[None],torch.tensor([len(e)],device=device))[0] for i,e in emb.items()}
 pairs=[(fixtures[0],fixtures[1]),(fixtures[1],fixtures[-1]),(fixtures[2],fixtures[2])]
 gold=native_scores(model,[emb[a] for a,b in pairs],[emb[b] for a,b in pairs]).cpu().numpy()
 enable_sdpa(model)
 fresh={i:endpoint_features(model,e[None],torch.tensor([len(e)],device=device))[0] for i,e in emb.items()}
 feature_error=max(float((fresh[i]-reference_feature[i]).abs().max()) for i in fixtures)
 cache_error=max(float(np.max(np.abs(fresh[i].cpu().numpy()-old[mapping[i]]))) for i in fixtures)
 z=np.zeros((len(mapping),64),np.float32);z[mapping>=0]=old[mapping[mapping>=0]]
 zt=torch.tensor(z,device=device)
 aa=np.array([a for a,b in pairs]);bb=np.array([b for a,b in pairs]);pred=cached_scores(model,zt,aa,bb,probabilities=True)
 probability_error=float(np.max(np.abs(gold-pred)))
 assert feature_error<2e-4 and cache_error<2e-4 and probability_error<2e-5,(feature_error,cache_error,probability_error)
 atomic(ROOT/'qualification/tuna.json',{'passed':True,'at_utc':now(),'fixture_ids':fixtures,'native_pairs':pairs,
  'encoder_sdpa_error':encoder_error,'classifier_sdpa_error':feature_error,'reused_feature_error':cache_error,
  'native_probability_error':probability_error,'native_oracle':'authors unchanged singleton forward; original checkpoint; FP32',
  'checkpoint':record(SOURCE/'weights/bernett_original.pt'),'reused_bundle_manifest':record(bundle/'SCORER_FREEZE.json')})
 print({'qualified':'tuna','probability_error':probability_error,'reused':int((mapping>=0).sum())},flush=True)
 dest=ROOT/'features/tuna';dest.mkdir(exist_ok=True)
 missing=np.flatnonzero(mapping<0);missing=missing[np.argsort(np.array(meta['length'])[missing],kind='stable')]
 for count,i in enumerate(missing):
  path=dest/f'{i:05d}.npz';side=path.with_suffix('.json')
  if side.exists():
   item=read(side);assert item['sequence_sha256']==meta['sha256'][i] and sha(path)==item['file']['sha256'];z[i]=np.load(path)['feature']
  else:
   e=embed(encoder,alphabet,[meta['sequence'][i]])[0];f=endpoint_features(model,e[None],torch.tensor([len(e)],device=device))[0]
   z[i]=f.cpu().numpy();save_npz(path,feature=z[i]);atomic(side,{'sequence_sha256':meta['sha256'][i],'file':record(path),'qualification_sha256':sha(ROOT/'qualification/tuna.json')})
  if count%25==0:print({'new_features':count+1,'total':len(missing),'seconds':time.monotonic()-started},flush=True)
 assert np.isfinite(z).all();save_npz(dest/'endpoints.npz',features=z)
 logits=cached_scores(model,torch.tensor(z,device=device),rows[:,0],rows[:,1],probabilities=False).astype(np.float64)
 assert all(torch.equal(p,before[n]) for n,p in model.named_parameters())
 out=ROOT/'predictions/tuna';out.mkdir(exist_ok=True);save_npz(out/'union.npz',scores=logits,probabilities=expit(logits))
 atomic(out/'done.json',{'at_utc':now(),'rows':len(rows),'file':record(out/'union.npz'),'score_kind':'native uncertainty-adjusted logit',
  'qualification':record(ROOT/'qualification/tuna.json'),'source_checkpoint':record(SOURCE/'weights/bernett_original.pt'),
  'features_reused':int((mapping>=0).sum()),'features_computed':len(missing),'trainable_parameters_unchanged':True,
  'seconds':time.monotonic()-started,'script':record(__file__)})
 print('TUnA complete',flush=True)
if __name__=='__main__':main()
