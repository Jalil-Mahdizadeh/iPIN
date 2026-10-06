"""Numerical attention qualification and worst-length forward/backward checks."""
import argparse
import gc
import json
import sys
import time
import types
from pathlib import Path
import torch
from transformers.models.esm.modeling_esm import EsmSelfAttention
from model import PairModel,sdpa_forward
from data import PairData
from state import atomic_json

ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(8);torch.manual_seed(2)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
torch.use_deterministic_algorithms(True)
model=PairModel(ROOT/'assets/esm2',backend='eager').cuda()
model.esm_mask.gradient_checkpointing_disable()
features={'input_ids':torch.randint(4,24,(4,97),device='cuda'),'attention_mask':torch.ones(4,97,device='cuda',dtype=torch.long),
          'labels':torch.tensor([0.,1.],device='cuda'),'mlm_labels':torch.full((4,97),-100,device='cuda',dtype=torch.long)}
features['input_ids'][:,0]=0;features['input_ids'][:,-1]=2
features['input_ids'][2:,73:]=1;features['attention_mask'][2:,73:]=0
features['mlm_labels'][:,8:12]=features['input_ids'][:,8:12];features['input_ids'][:,8:12]=32

def switch(backend):
    for layer in model.esm_mask.esm.encoder.layer:
        a=layer.attention.self;a.profile_backend=backend
        a.forward=types.MethodType(EsmSelfAttention.forward if backend=='eager' else sdpa_forward,a)

def run(backend,bf16):
    switch(backend);model.zero_grad(set_to_none=True)
    with torch.autocast('cuda',dtype=torch.bfloat16,enabled=bf16):loss,*_,z=model(**features)
    loss.backward()
    grads={n:p.grad.detach().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}
    assert all(torch.isfinite(v).all() for v in grads.values())
    return loss.item(),z.float().cpu(),grads

results={'checks':[]}
for bf16,candidate in [(False,'math'),(True,'efficient')]:
    l,z,g=run('eager',bf16);l2,z2,g2=run(candidate,bf16)
    sq=sum(float((g[n]-g2[n]).double().square().sum()) for n in g)
    denom=sum(float(g[n].double().square().sum()) for n in g)
    row={'precision':'bf16' if bf16 else 'fp32','backend':candidate,'loss_abs_diff':abs(l-l2),
         'logit_max_abs_diff':float((z-z2).abs().max()),'gradient_relative_l2_diff':(sq/denom)**.5,'gradient_tensors':len(g)}
    assert row['logit_max_abs_diff']<(.03 if bf16 else .0001),row
    assert row['gradient_relative_l2_diff']<(.1 if bf16 else .0001),row
    results['checks'].append(row);print(json.dumps(row),flush=True)
    del g,g2
model.zero_grad(set_to_none=True);del features;gc.collect();torch.cuda.empty_cache()
switch('efficient');model.esm_mask.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
for split,backward in [('train',True),('val',False)]:
    data=PairData(ROOT/'data/prepared',split);idx=int(data.lengths.argmax());f=data.batch([idx],0,2,backward,'cuda')
    model.train(backward);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
    if backward:
        with torch.autocast('cuda',dtype=torch.bfloat16):loss,*_=model(**f)
        loss.backward();assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
    else:
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            z=model(f['input_ids'],f['attention_mask']);assert torch.isfinite(z).all()
            swapped=model(f['input_ids'].flip(0),f['attention_mask'].flip(0))
            assert torch.equal(z.mean(1),swapped.mean(1))
    torch.cuda.synchronize()
    row={'split':split,'paired_tokens':int(data.lengths[idx]),'orientation_batch':2,'backward':backward,
         'seconds':time.monotonic()-start,'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
    results['checks'].append(row);print(json.dumps(row),flush=True)
    model.zero_grad(set_to_none=True);del f;gc.collect();torch.cuda.empty_cache()
results['passed']=True;atomic_json(ROOT/'qualification/attention.json',results)
