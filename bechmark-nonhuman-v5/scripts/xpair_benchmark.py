"""Two frozen X-PAIR releases, shared native Ankh features, resumable shards."""
import argparse,fcntl,hashlib,json,os,sys,time
import numpy as np
import torch
from bench_utils import ROOT,EXTERNAL,atomic,cuda,now,read,record,save_npz,sha
SOURCE=EXTERNAL/'experiments/x_pair_test2_v1'
sys.path.insert(0,str(ROOT/'scripts/xpair'))
from native import load_encoder,encode,projected_forward,clear_attention
from xpair.model import XPairModel
from xpair.utils.embedding_menager import pad_and_mask_emb_batch
NAMES={'xpair-bernett':'interaction_bernett','xpair-default':'multitask_xfair'}
LIMIT=8_000_000;MAX_BATCH=64;CHUNK=2048

def signature():
 return {p:sha(ROOT/p) for p in ['scripts/xpair_benchmark.py','scripts/xpair/native.py','scripts/xpair/io_utils.py','scripts/container.sh','provenance/prepared.json','provenance/runtime-inputs.json']}
def provenance():
 return {'encoder_freeze_sha256':sha(SOURCE/'sources/ANKH_FREEZE.json'),'runtime_sha256':sha(SOURCE/'runtime/requirements.freeze.txt'),
         'encoder_script_sha256':sha(SOURCE/'scripts/native.py')}
def feature_path(i):
 old=int(np.load(ROOT/'data/xpair-cache-indices.npy',mmap_mode='r')[i])
 return SOURCE/'features'/f'{old:05d}.pt' if old>=0 else ROOT/'features/xpair'/f'{i:05d}.pt'
def verified_feature(i,meta):
 path=feature_path(i);item=read(path.with_suffix('.json'))
 assert item['sequence_sha256']==meta['sha256'][i] and item['length']==meta['length'][i]
 assert item['dimension']==1536 and item['dtype']=='float32'
 assert item['file']['bytes']==path.stat().st_size and sha(path)==item['file']['sha256']
 assert all(item[k]==v for k,v in provenance().items())
 return path

def put(i,feature,meta):
 path=feature_path(i);assert ROOT in path.parents;path.parent.mkdir(exist_ok=True)
 tmp=path.with_suffix('.tmp-'+str(os.getpid()));torch.save(feature,tmp);os.replace(tmp,path)
 atomic(path.with_suffix('.json'),{'id':int(i),'sequence_sha256':meta['sha256'][i],'length':len(feature),'dimension':1536,'dtype':'float32','file':record(path),**provenance()})

def load_pair_model(name,device):
 path=SOURCE/'sources/X-PAIR/pretrained_models'/(NAMES[name]+'.ckpt')
 expected=read(ROOT/'provenance/runtime-inputs.json')['items'][name];assert sha(path)==expected['sha256']
 allowed=[argparse.Namespace,np.dtype,(np.core.multiarray.scalar,'numpy._core.multiarray.scalar'),type(np.dtype('float64')),type(np.dtype('float32'))]
 with torch.serialization.safe_globals(allowed):ckpt=torch.load(path,map_location='cpu',weights_only=True)
 model=XPairModel(ckpt['hyper_parameters']).to(device).eval().requires_grad_(False)
 model.load_state_dict(ckpt['state_dict'],strict=True)
 return model,{'checkpoint':record(path),'epoch':int(ckpt['epoch']),'global_step':int(ckpt['global_step']),'hyper_parameters':ckpt['hyper_parameters']}

@torch.inference_mode()
def embedding(rank,world,device):
 assert world==4
 meta=read(ROOT/'data/sequences.json');mapping=np.load(ROOT/'data/xpair-cache-indices.npy');lengths=np.array(meta['length'])
 for item in read(SOURCE/'sources/ANKH_FREEZE.json')['files']:
  path=SOURCE/item['path'];assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
 model,tokenizer=load_encoder(device);started=time.monotonic()
 # Before scoring, independently reproduce a reused short and long feature on each worker.
 matched=np.flatnonzero(mapping>=0);matched=matched[np.argsort(lengths[matched],kind='stable')]
 fixtures=[int(matched[len(matched)//2]),int(matched[-1])]
 cases=[]
 for i in fixtures:
  path=verified_feature(i,meta);gold=torch.load(path,map_location='cpu',weights_only=True)
  fresh=encode(model,tokenizer,meta['sequence'][i],device);error=float((gold-fresh).abs().max());assert error<=2e-4,error
  cases.append({'id':i,'length':len(fresh),'max_abs_error':error})
 assigned=np.arange(len(mapping))[rank::world];assigned=assigned[np.argsort(-lengths[assigned],kind='stable')];files=[];computed=0
 for count,i in enumerate(assigned):
  if mapping[i]>=0 or feature_path(i).with_suffix('.json').exists():path=verified_feature(i,meta)
  else:
   feature=encode(model,tokenizer,meta['sequence'][i],device);put(i,feature,meta);path=verified_feature(i,meta);computed+=1
  files.append(record(path.with_suffix('.json')))
  if count%50==0:print({'stage':'embedding','rank':rank,'finished':count+1,'total':len(assigned),'new':computed,'seconds':time.monotonic()-started},flush=True)
 atomic(ROOT/'features/xpair'/f'rank-{rank:02d}.done.json',{'passed':True,'rank':rank,'world':world,'rows':len(assigned),'files':files,'provenance':provenance(),
  'reuse_qualification':cases,'seconds':time.monotonic()-started,'at_utc':now()})

@torch.inference_mode()
def qualify(device):
 meta=read(ROOT/'data/sequences.json');order=np.argsort(meta['length'],kind='stable')
 ids=[int(order[j]) for j in [0,1,len(order)//4,len(order)//2,3*len(order)//4,-1]]
 features={i:torch.load(verified_feature(i,meta),map_location=device,weights_only=True) for i in ids}
 pairs=[(ids[0],ids[1]),(ids[0],ids[-1]),(ids[-1],ids[2]),(ids[2],ids[3]),(ids[3],ids[4]),(ids[-1],ids[-1])]
 output={}
 for name in NAMES:
  model,info=load_pair_model(name,device);before={k:v.clone() for k,v in model.state_dict().items()};cached={i:model.embedding_projection(f) for i,f in features.items()}
  gold=[];fast=[];reverse=[]
  for a,b in pairs:
   m1=torch.ones(1,len(features[a]),dtype=torch.bool,device=device);m2=torch.ones(1,len(features[b]),dtype=torch.bool,device=device)
   gold.append(model({'input1':(features[a][None],m1),'input2':(features[b][None],m2)},task='interaction')[0]);clear_attention(model)
   fast.append(projected_forward(model,cached[a][None],cached[b][None],m1,m2))
   reverse.append(projected_forward(model,cached[b][None],cached[a][None],m2,m1))
  gold=torch.cat(gold);fast=torch.cat(fast);reverse=torch.cat(reverse)
  short=[pairs[0],pairs[3],pairs[4]];pp=[short[i%3] for i in range(MAX_BATCH)]
  x1,m1=pad_and_mask_emb_batch([features[a] for a,b in pp]);x2,m2=pad_and_mask_emb_batch([features[b] for a,b in pp])
  raw=model({'input1':(x1,m1),'input2':(x2,m2)},task='interaction')[0];clear_attention(model)
  z1,_=pad_and_mask_emb_batch([cached[a] for a,b in pp]);z2,_=pad_and_mask_emb_batch([cached[b] for a,b in pp]);result=projected_forward(model,z1,z2,m1,m2)
  scalar=torch.stack([gold[pairs.index(p)] for p in pp])
  errors={'cache_logit':float((gold-fast).abs().max()),'swap_logit':float((gold-reverse).abs().max()),'padded_cache_logit':float((raw-result).abs().max()),
   'batch_singleton_logit':float((scalar-result).abs().max()),'padded_probability':float((raw.sigmoid()-result.sigmoid()).abs().max())}
  assert max(errors.values())<2e-4 and errors['padded_probability']<1e-5,errors
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
  output[name]={'errors':errors,'info':info,'state_unchanged':True}
  print({'qualified':name,'errors':errors},flush=True)
  del model,cached,before,x1,x2,z1,z2;torch.cuda.empty_cache()
 atomic(ROOT/'qualification/xpair.json',{'passed':True,'at_utc':now(),'signature':signature(),'models':output,'pairs':pairs,'test_metrics_read':False})

@torch.inference_mode()
def cache(model,meta,device):
 offsets=[];segments=[];position=0
 for i in range(len(meta['sequence'])):
  f=torch.load(verified_feature(i,meta),map_location=device,weights_only=True);assert f.shape==(meta['length'][i],1536) and f.dtype==torch.float32
  offsets.append(position);position+=len(f);segments.append(model.embedding_projection(f))
 return torch.cat(segments),torch.tensor(offsets,device=device),torch.tensor(meta['length'],device=device)
def padded(table,offsets,lengths,ids,maxlen):
 index=torch.as_tensor(ids,device=table.device);positions=torch.arange(maxlen,device=table.device)[None,:];mask=positions<lengths[index,None]
 gather=torch.where(mask,offsets[index,None]+positions,0);return table[gather]*mask[:,:,None],mask

def batches(indices,a,b,lengths):
 start=0
 while start<len(indices):
  end=start;la=lb=0
  while end<len(indices) and end-start<MAX_BATCH:
   k=indices[end];na=max(la,int(lengths[a[k]]));nb=max(lb,int(lengths[b[k]]))
   if end>start and na*nb*(end-start+1)>LIMIT:break
   la=na;lb=nb;end+=1
  yield indices[start:end],la,lb
  start=end

@torch.inference_mode()
def score(rank,world,device):
 assert world==4
 q=read(ROOT/'qualification/xpair.json');assert q['passed'] and q['signature']==signature()
 for r in range(world):
  item=read(ROOT/'features/xpair'/f'rank-{r:02d}.done.json');assert item['passed'] and item['provenance']==provenance()
 meta=read(ROOT/'data/sequences.json');lengths=np.array(meta['length']);rows=np.load(ROOT/'data/union.npy');a=rows[:,0];b=rows[:,1]
 order=np.lexsort((lengths[b]//64,lengths[a]//64))[rank::world]
 for name in NAMES:
  out=ROOT/'predictions'/name;out.mkdir(exist_ok=True);lock=(out/f'rank-{rank}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  model,info=load_pair_model(name,device);before={k:v.clone() for k,v in model.state_dict().items()};table,offsets,glengths=cache(model,meta,device)
  files=[];started=time.monotonic()
  for start in range(0,len(order),CHUNK):
   ids=order[start:start+CHUNK];path=out/f'rank-{rank:02d}-chunk-{start:06d}.npz';side=path.with_suffix('.json')
   if side.exists():
    item=read(side);assert item['signature']==signature() and sha(path)==item['file']['sha256'];assert np.array_equal(np.load(path)['indices'],ids)
   else:
    logits=[]
    for selected,la,lb in batches(ids,a,b,lengths):
     x1,m1=padded(table,offsets,glengths,a[selected],la);x2,m2=padded(table,offsets,glengths,b[selected],lb)
     logits.append(projected_forward(model,x1,x2,m1,m2).cpu().numpy())
    z=np.concatenate(logits).astype(np.float64);assert z.shape==(len(ids),) and np.isfinite(z).all()
    save_npz(path,indices=ids,scores=z);atomic(side,{'at_utc':now(),'signature':signature(),'file':record(path)})
   files.append(record(path));print({'model':name,'rank':rank,'rows':start+len(ids),'total':len(order),'seconds':time.monotonic()-started},flush=True)
  assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
  atomic(out/f'rank-{rank:02d}.done.json',{'at_utc':now(),'rank':rank,'world':world,'rows':len(order),'files':files,'signature':signature(),
   'model':info,'state_unchanged':True,'seconds':time.monotonic()-started})
  del model,table,before;lock.close();torch.cuda.empty_cache()

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['embedding','qualify','score'],required=True);args=parser.parse_args()
 rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));device=cuda(int(os.environ.get('LOCAL_RANK','0')))
 if args.stage=='qualify':qualify(device)
 elif args.stage=='embedding':embedding(rank,world,device)
 else:score(rank,world,device)
if __name__=='__main__':main()
