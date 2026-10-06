"""Require bitwise identity of uninterrupted versus resumed training state."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from state import atomic_json,sha256

def load(run):
    run=Path(run);m=json.loads((run/'latest.json').read_text());p=run/'checkpoints'/m['file']
    assert sha256(p)==m['sha256']
    return torch.load(p,map_location='cpu',mmap=True,weights_only=False)

def compare(a,b,path='root',stats=None):
    if stats is None:stats={'tensor_count':0,'tensor_elements':0}
    if isinstance(a,torch.Tensor):
        assert torch.equal(a,b),(path,'tensor differs',float((a-b).abs().max()))
        stats['tensor_count']+=1;stats['tensor_elements']+=a.numel()
    elif isinstance(a,np.ndarray):assert np.array_equal(a,b),path
    elif isinstance(a,dict):
        assert a.keys()==b.keys(),path
        for k in a:compare(a[k],b[k],f'{path}.{k}',stats)
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b),path
        for i,(x,y) in enumerate(zip(a,b)):compare(x,y,f'{path}.{i}',stats)
    else:assert a==b,(path,a,b)
    return stats

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('uninterrupted');p.add_argument('resumed');p.add_argument('--report',required=True);a=p.parse_args()
    left,right=load(a.uninterrupted),load(a.resumed);result={}
    for key in ['model','optimizer','training_state','rng_by_rank','fingerprint','world_size']:
        result[key]=compare(left[key],right[key],key)
    result.update(passed=True,bitwise_identical=True,uninterrupted=a.uninterrupted,resumed=a.resumed)
    atomic_json(a.report,result);print(json.dumps(result,indent=2))
