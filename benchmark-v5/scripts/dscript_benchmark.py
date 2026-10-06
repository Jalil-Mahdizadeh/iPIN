"""Original D-SCRIPT human_v1, full-length compatibility and verified feature reuse."""
import argparse,fcntl,os,sys,time
from pathlib import Path
import numpy as np
import torch
from bench_utils import ROOT,EXTERNAL,atomic,cuda,now,read,record,save_npz,sha
sys.path.insert(0,str(ROOT/'scripts/dscript'))
from native_adapter import create,learned_digest
from dscript.alphabets import Uniprot21
from dscript.pretrained import get_pretrained
SOURCE=EXTERNAL/'benchmark/dscript'
CACHE=SOURCE/'runs/original-v1/projection_cache'

def signature():
 return {name:sha(ROOT/name) for name in ['scripts/dscript_benchmark.py','scripts/dscript/native_adapter.py','scripts/dscript/common.py','scripts/container_dscript.sh','provenance/dscript-runtime.json','provenance/prepared.json']}

@torch.inference_mode()
def prepare(device):
 started=time.monotonic();meta=read(ROOT/'data/sequences.json');cache=read(CACHE/'CACHE.json');assert cache['no_training'] and cache['full_length']
 for item in cache['files']:
  p=CACHE/item['path'];assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
 assert cache['original_model_sha256']==read(ROOT/'provenance/dscript-runtime.json')['original_checkpoint_sha256']
 endpoints=read(CACHE/'endpoints.json');lookup={h:i for i,h in enumerate(endpoints)};mapping=np.array([lookup.get(h,-1) for h in meta['sha256']])
 offsets=np.load(CACHE/'offsets.npy');old=np.load(CACHE/'projected.npy',mmap_mode='r')
 model=create(device=device,max_length=max(meta['length']));digest=learned_digest(model);assert digest==cache['learned_state_sha256']
 lm=get_pretrained('lm_v1').to(device).eval().requires_grad_(False);alphabet=Uniprot21()
 def raw(i):
  x=torch.from_numpy(alphabet.encode(meta['sequence'][i].encode())).long()[None].to(device)
  return lm.transform(x)
 matched=np.flatnonzero(mapping>=0);matched=matched[np.argsort(np.array(meta['length'])[matched],kind='stable')]
 fixtures=[int(matched[j]) for j in [0,len(matched)//2,int(.9*len(matched)),-1]]
 representations={i:raw(i) for i in fixtures};fresh={i:model.embedding(r) for i,r in representations.items()}
 errors=[]
 for i in fixtures:
  j=mapping[i];gold=torch.as_tensor(np.array(old[offsets[j]:offsets[j+1]]),device=device)[None]
  error=float((gold-fresh[i]).abs().max());assert error<2e-4,error;errors.append(error)
 native=get_pretrained('human_v1').to(device).eval().requires_grad_(False)
 pairs=[(fixtures[0],fixtures[1]),(fixtures[1],fixtures[2]),(fixtures[2],fixtures[2])]
 native_errors=[]
 for a,b in pairs:
  assert max(meta['length'][a],meta['length'][b])<=2000
  gold=native.predict(representations[a],representations[b]);fast=model.predict(fresh[a],fresh[b]);error=float((gold-fast).abs().max());assert error<1e-5,error;native_errors.append(error)
 del native
 # Same full-map formula and nonlearned positions; compare tiling across a true long input.
 a,b=fixtures[-1],fixtures[1];model.tile_area=None;gold=model.predict(fresh[a],fresh[b]);model.tile_area=1_000_000
 tiled=model.predict(fresh[a],fresh[b]);reverse=model.predict(fresh[b],fresh[a]);tile_error=float((gold-tiled).abs().max());swap_error=float((tiled-reverse).abs().max())
 assert tile_error<1e-5 and swap_error<1e-5,(tile_error,swap_error)
 atomic(ROOT/'qualification/dscript.json',{'passed':True,'at_utc':now(),'signature':signature(),'fixture_ids':fixtures,'native_pairs':pairs,
  'native_probability_errors':native_errors,'projection_cache_errors':errors,'long_pair':[a,b],'long_pair_lengths':[meta['length'][a],meta['length'][b]],
  'tiled_fullmap_error':tile_error,'swap_error':swap_error,'learned_state_sha256':digest,
  'cache':record(CACHE/'CACHE.json'),'reference':'Unmodified human_v1 for supported lengths; same formula full-map vs tiled for long lengths',
  'checkpoint':record('/opt/dscript/weights/dscript_human_v1.pt'),'encoder':record('/opt/dscript/weights/dscript_lm_v1.pt')})
 print({'qualified':'dscript','cached_features':int((mapping>=0).sum()),'new_features':int((mapping<0).sum()),'native_errors':native_errors,'tile_error':tile_error},flush=True)
 out=ROOT/'features/dscript';out.mkdir(exist_ok=True);segments=[];new_offsets=[0]
 for i in range(len(mapping)):
  j=int(mapping[i]);path=out/f'{i:05d}.npz';side=path.with_suffix('.json')
  if j>=0:value=np.array(old[offsets[j]:offsets[j+1]])
  elif side.exists():
   item=read(side);assert item['sequence_sha256']==meta['sha256'][i] and item['state']==digest and sha(path)==item['file']['sha256'];value=np.load(path)['feature']
  else:
   value=model.embedding(raw(i))[0].cpu().numpy();save_npz(path,feature=value);atomic(side,{'sequence_sha256':meta['sha256'][i],'state':digest,'file':record(path)})
  assert value.shape==(meta['length'][i],100) and value.dtype==np.float32 and np.isfinite(value).all();segments.append(value);new_offsets.append(new_offsets[-1]+len(value))
  if i%500==0:print({'stage':'features','finished':i+1,'total':len(mapping),'seconds':time.monotonic()-started},flush=True)
 array=np.concatenate(segments);path=out/'projected.npy';temp=path.with_suffix('.tmp')
 with temp.open('wb') as f:np.save(f,array)
 os.replace(temp,path);np.save(out/'offsets.npy',np.array(new_offsets,np.int64))
 assert learned_digest(model)==digest
 atomic(out/'complete.json',{'at_utc':now(),'signature':signature(),'files':[record(path),record(out/'offsets.npy')],'learned_state_sha256':digest,
  'features_reused':int((mapping>=0).sum()),'features_computed':int((mapping<0).sum()),'seconds':time.monotonic()-started})
 print('D-SCRIPT features ready',flush=True)

@torch.inference_mode()
def score(device,rank,world):
 assert world==4;q=read(ROOT/'qualification/dscript.json');assert q['passed'] and q['signature']==signature()
 complete=read(ROOT/'features/dscript/complete.json');assert complete['signature']==signature()
 for item in complete['files']:assert sha(item['path'])==item['sha256']
 meta=read(ROOT/'data/sequences.json');model=create(device=device,max_length=max(meta['length']));digest=learned_digest(model);assert digest==complete['learned_state_sha256']
 table=torch.tensor(np.load(ROOT/'features/dscript/projected.npy'),device=device);offsets=np.load(ROOT/'features/dscript/offsets.npy');rows=np.load(ROOT/'data/union.npy')
 lengths=np.array(meta['length']);order=np.argsort(lengths[rows[:,0]]*lengths[rows[:,1]],kind='stable')[rank::world]
 out=ROOT/'predictions/dscript';out.mkdir(exist_ok=True);lock=(out/f'rank-{rank}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 records=[];started=time.monotonic()
 for start in range(0,len(order),1024):
  ids=order[start:start+1024];path=out/f'rank-{rank:02d}-chunk-{start:06d}.npz';side=path.with_suffix('.json')
  if side.exists():
   item=read(side);assert item['signature']==signature() and sha(path)==item['file']['sha256'];assert np.array_equal(np.load(path)['indices'],ids)
  else:
   pending=torch.empty(len(ids),device=device,dtype=torch.float32)
   for k,i in enumerate(ids):
    a,b=rows[i,:2];pending[k]=model.predict(table[offsets[a]:offsets[a+1]][None],table[offsets[b]:offsets[b+1]][None])
   values=pending.cpu().numpy().astype(np.float64);assert np.isfinite(values).all() and (values>=0).all() and (values<=1).all()
   save_npz(path,indices=ids,scores=values);atomic(side,{'signature':signature(),'file':record(path),'at_utc':now()})
  records.append(record(path));print({'model':'dscript','rank':rank,'finished':start+len(ids),'total':len(order),'seconds':time.monotonic()-started},flush=True)
 assert learned_digest(model)==digest
 atomic(out/f'rank-{rank:02d}.done.json',{'rank':rank,'world':world,'rows':len(order),'signature':signature(),'files':records,'at_utc':now(),'seconds':time.monotonic()-started,
  'learned_state_unchanged':True,'gpu_uuid':str(torch.cuda.get_device_properties(device).uuid)})

def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['prepare','score'],required=True);a=p.parse_args();device=cuda(int(os.environ.get('LOCAL_RANK','0')))
 if a.stage=='prepare':prepare(device)
 else:score(device,int(os.environ.get('RANK','0')),int(os.environ.get('WORLD_SIZE','1')))
if __name__=='__main__':main()
