"""Full 650M memory/gradient qualification on actual longest train/validation pairs."""
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from data import PairData
from model import PairModel
from state import atomic_json, sha256

root=Path(__file__).resolve().parents[1]
torch.set_num_threads(8);torch.use_deterministic_algorithms(True)
torch.backends.cuda.matmul.allow_tf32=False
train=PairData(root/'data/prepared','fold-1','train')
val=PairData(root/'data/prepared','fold-0','val')
cfg=json.loads((root/'configs/clean-cls-linear-lr5e-6-fold0-seed2.json').read_text())
results=[]
for name,extra in [('clean-cls',{}),('clean-residue',{'readout':'residue_mean'})]:
    torch.manual_seed(2)
    model=PairModel(root/'assets/esm2',{**cfg,**extra}).cuda()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-6,foreach=False)
    for stage,data,training in [('longest-training',train,True),('longest-validation',val,False)]:
        ids=np.array([np.argmax(data.lengths)])
        batch=data.batch(ids,np.zeros(1,dtype=int),2,False,'cuda')
        model.train(training);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        if training:
            with torch.autocast('cuda',dtype=torch.bfloat16): loss,*_=model(**batch)
            assert torch.isfinite(loss)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            value=float(loss.detach());del loss
        else:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16): logits=model(**batch,compute_loss=False)
            assert torch.isfinite(logits).all();value=logits.float().cpu().tolist();del logits
        torch.cuda.synchronize()
        item={'model':name,'stage':stage,'tokens':int(data.lengths[ids[0]]),
            'orientations':2,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),'seconds':time.monotonic()-start,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
            'peak_reserved_gib':torch.cuda.max_memory_reserved()/2**30,
            'finite_output':value,'source_row':int(data.rows[ids[0],3])}
        results.append(item);print(json.dumps(item),flush=True)
        del batch
    del optimizer,model;gc.collect();torch.cuda.empty_cache()
report={'passed':True,'profiles':results,'precision':'FP32 parameters/AdamW, BF16 autocast',
    'device':torch.cuda.get_device_name(),'total_device_gib':torch.cuda.get_device_properties(0).total_memory/2**30,
    'no_test_data':True,'source_sha256':{'scripts/qualify_profile.py':sha256(Path(__file__))},'disposable_optimizer_steps':2,
    'code':{n:sha256(root/'scripts'/n) for n in ['train.py','model.py','data.py','state.py','release.py']}}
atomic_json(root/'qualification/full-model-profile.json',report)
