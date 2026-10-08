"""Numerical and resource qualification; synthetic cases are not biological results."""
import argparse
import json
import time
import numpy as np
import torch
from common import ROOT, atomic, config, now
from encoder import Encoder


def main():
    p=argparse.ArgumentParser();p.add_argument('--weights');args=p.parse_args()
    started=time.monotonic();model=Encoder(args.weights);rng=np.random.default_rng(config()['seed']);cases=[]
    for depth,length in [(16,128),(32,512),(64,1024),(128,1536)]:
        tokens=rng.integers(0,20,size=(depth,length),dtype=np.uint8)
        tokens[1:][rng.random((depth-1,length))<.15]=25
        tokens[:,np.arange(length)%13==0]=26
        breakpoint=length//2;torch.cuda.reset_peak_memory_stats();t=time.monotonic()
        f=model.symmetric(tokens,breakpoint);torch.cuda.synchronize()
        case={'depth':depth,'length':length,'two_orientation_seconds':time.monotonic()-t,
              'peak_gib':torch.cuda.max_memory_allocated()/2**30,'finite':bool(np.isfinite(f).all())}
        assert f.shape==(128,) and case['finite']
        if length==128:
            swapped=np.c_[tokens[:,breakpoint:],tokens[:,:breakpoint]]
            other=model.symmetric(swapped,length-breakpoint)
            case['ab_ba_max_error']=float(np.max(np.abs(f-other)))
            assert case['ab_ba_max_error']<=1e-6
        cases.append(case);print(json.dumps(case),flush=True)
        atomic(ROOT/'qualification/encoder-progress.json',{'at_utc':now(),'cases':cases})
    atomic(ROOT/'qualification/encoder.json',{'passed':True,'at_utc':now(),'seconds':time.monotonic()-started,
        'cases':cases,'synthetic_only':True,'structural_heads_evaluated':False,'parameters_frozen':True,
        'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(),
        'memory_bytes':torch.cuda.get_device_properties(0).total_memory,'backend':'upstream PyTorch triangle updates'})


if __name__=='__main__':main()
