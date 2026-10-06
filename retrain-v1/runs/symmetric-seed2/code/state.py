"""Crash-consistent checkpoint commits, hashes, retained fallback, RNG state."""
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist

def sha256(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def fsync_dir(p):
    fd=os.open(str(p),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

def atomic_json(p,value):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_name(p.name+f'.tmp-{os.getpid()}')
    with temp.open('w') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(temp,p);fsync_dir(p.parent)

def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state()}

def restore_rng(s):
    random.setstate(s['python']);np.random.set_state(s['numpy']);torch.set_rng_state(s['torch']);torch.cuda.set_rng_state(s['cuda'])

class Checkpoints:
    def __init__(self,out,rank,world,fingerprint):
        self.out=Path(out);self.path=self.out/'checkpoints';self.rank=rank;self.world=world;self.fingerprint=fingerprint
        self.path.mkdir(parents=True,exist_ok=True)
    def barrier(self):
        if self.world>1:dist.barrier()
    def save(self,model,optimizer,state,best=False):
        rng=capture_rng();all_rng=[None]*self.world
        if self.world>1:dist.all_gather_object(all_rng,rng)
        else:all_rng=[rng]
        meta=None
        if self.rank==0:
            name=f'update-{state["update"]:09d}.pt';target=self.path/name;temp=self.path/(name+'.partial')
            payload={'format_version':1,'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                     'training_state':state,'rng_by_rank':all_rng,'world_size':self.world,'fingerprint':self.fingerprint,
                     'precision':'fp32 parameters and optimizer; bf16 autocast; no GradScaler',
                     'checkpoint_boundary':'optimizer update complete, gradients cleared'}
            with temp.open('wb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
            os.replace(temp,target);fsync_dir(self.path)
            meta={'file':name,'sha256':sha256(target),'bytes':target.stat().st_size,
                  'update':state['update'],'epoch':state['epoch'],'next_chunk':state['next_chunk'],
                  'world_size':self.world,'fingerprint':self.fingerprint}
            atomic_json(self.path/(name+'.json'),meta);atomic_json(self.out/'latest.json',meta)
            if best:atomic_json(self.out/'best.json',meta)
            protected={name}
            if (self.out/'best.json').exists():protected.add(json.loads((self.out/'best.json').read_text())['file'])
            manifests=sorted(self.path.glob('update-*.pt.json'),reverse=True)
            protected.update(m.name.removesuffix('.json') for m in manifests[:2])
            for m in manifests:
                old=m.name.removesuffix('.json')
                if old not in protected:
                    (self.path/old).unlink(missing_ok=True);m.unlink()
            atomic_json(self.out/'checkpoint-status.json',{'last_committed_update':state['update'],'latest':meta,'retained':sorted(protected)})
        self.barrier()
        return meta
    def load(self,model,optimizer):
        selection=[None]
        if self.rank==0:
            candidates=[]
            if (self.out/'latest.json').exists():candidates.append(self.out/'latest.json')
            candidates+=sorted(self.path.glob('update-*.pt.json'),reverse=True)
            if not candidates and (self.out/'checkpoint-status.json').exists():
                raise RuntimeError('Previously committed run has lost its checkpoints; refusing a silent restart')
            attempted=set();failures=[]
            for m in candidates:
                try:
                    x=json.loads(m.read_text());p=self.path/Path(x['file']).name
                    candidate=(p,x['sha256'],x['bytes'])
                    if candidate in attempted:continue
                    attempted.add(candidate)
                    if x['fingerprint']!=self.fingerprint:raise RuntimeError('configuration/code/data fingerprint mismatch; refusing to resume')
                    if x['world_size']!=self.world:raise RuntimeError('world size changed; exact continuation requires the same GPU count')
                    if p.stat().st_size!=x['bytes'] or sha256(p)!=x['sha256']:raise IOError('checkpoint size/hash mismatch')
                    selection[0]={'path':str(p),'manifest':x,'fallback_failures':failures};break
                except (OSError,ValueError,KeyError) as e:failures.append({'candidate':str(m),'error':str(e)})
            if candidates and selection[0] is None:raise RuntimeError(f'No valid complete checkpoint: {failures}')
        if self.world>1:dist.broadcast_object_list(selection,src=0)
        if selection[0] is None:return None,None
        # These pickle checkpoints are generated locally by this script and hash-verified.
        saved=torch.load(selection[0]['path'],map_location='cpu',mmap=True,weights_only=False)
        assert saved['fingerprint']==self.fingerprint and saved['world_size']==self.world
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng_by_rank'][self.rank]);state=saved['training_state']
        del saved
        return state,selection[0]
