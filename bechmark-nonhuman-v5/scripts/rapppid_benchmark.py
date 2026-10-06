"""Released RAPPPID singleton inference, with its native 1500-residue limit."""
import sys,time
sys.path[:0]=['/opt/rapppid/runtime','/opt/rapppid/upstream/rapppid']
import numpy as np
import torch
from bench_utils import ROOT,atomic,cuda,now,read,record,save_npz,sha
sys.path.insert(0,str(ROOT/'scripts/rapppid'))
from native_model import load_model,tokenizer,encode,singleton_embedding,native_probability,independent_head,learned_digest,CHECKPOINT

@torch.inference_mode()
def main():
 started=time.monotonic();device=cuda();model=load_model(device);digest=learned_digest(model);spp=tokenizer()
 meta=read(ROOT/'data/sequences.json');rows=np.load(ROOT/'data/union.npy');tokens=np.stack([encode(spp,s) for s in meta['sequence']])
 order=np.argsort(meta['length'],kind='stable');fixtures=[int(order[j]) for j in [0,len(order)//2,int(.9*len(order)),-1]]
 cached={i:singleton_embedding(model,tokens[i],device) for i in fixtures}
 pairs=[(fixtures[0],fixtures[1]),(fixtures[1],fixtures[-1]),(fixtures[2],fixtures[2]),(fixtures[-1],fixtures[0])]
 gold=torch.cat([native_probability(model,tokens[a],tokens[b],device) for a,b in pairs])
 aa=torch.cat([cached[a] for a,b in pairs]);bb=torch.cat([cached[b] for a,b in pairs])
 prob=independent_head(model,aa,bb);rev=independent_head(model,bb,aa)
 error=float((gold-prob).abs().max());swap=float((prob-rev).abs().max());assert error<2e-6 and swap<2e-6
 atomic(ROOT/'qualification/rapppid.json',{'passed':True,'at_utc':now(),'fixture_ids':fixtures,'pairs':pairs,'probability_error':error,
  'swap_error':swap,'reference':'Unmodified native test_step computation at batch size one','checkpoint':record(CHECKPOINT)})
 print({'qualified':'rapppid','error':error},flush=True)
 path=ROOT/'features/rapppid-endpoints.npz';side=path.with_suffix('.json')
 if side.exists():
  item=read(side);assert item['sequence_manifest']==sha(ROOT/'data/sequences.json') and item['checkpoint_digest']==digest and sha(path)==item['file']['sha256'];z=np.load(path)['features']
 else:
  z=np.empty((len(tokens),64),np.float32)
  for i in range(len(tokens)):
   z[i]=singleton_embedding(model,tokens[i],device).cpu().numpy()[0]
   if i%250==0:print({'encoded':i+1,'total':len(tokens),'seconds':time.monotonic()-started},flush=True)
  assert np.isfinite(z).all();save_npz(path,features=z)
  atomic(side,{'sequence_manifest':sha(ROOT/'data/sequences.json'),'checkpoint_digest':digest,'file':record(path)})
 table=torch.tensor(z,device=device);probability=[]
 for start in range(0,len(rows),2048):
  subset=rows[start:start+2048];probability.append(independent_head(model,table[subset[:,0]],table[subset[:,1]]).cpu().numpy())
 probs=np.concatenate(probability).astype(np.float64);assert np.isfinite(probs).all() and (probs>=0).all() and (probs<=1).all()
 assert learned_digest(model)==digest
 out=ROOT/'predictions/rapppid';out.mkdir(exist_ok=True);save_npz(out/'union.npz',scores=probs,probabilities=probs)
 atomic(out/'done.json',{'at_utc':now(),'rows':len(rows),'file':record(out/'union.npz'),'score_kind':'native probability',
  'qualification':record(ROOT/'qualification/rapppid.json'),'source_checkpoint':record(CHECKPOINT),'state_unchanged':True,
  'proteins_over_1500_residues':sum(n>1500 for n in meta['length']),'seconds':time.monotonic()-started,'script':record(__file__)})
 print('RAPPPID complete',flush=True)
if __name__=='__main__':main()
