"""Sharded, frozen FP32 evaluation; no training or test-based selection."""
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import socket
import time
import torch
from torch.nn import functional as F
from native_model import ROOT, configure, load_model, tokenize, sha256
from datasets_eval import task_specs, read_task

parser=argparse.ArgumentParser()
parser.add_argument('--rank',type=int,default=0)
parser.add_argument('--world-size',type=int,default=1)
parser.add_argument('--job-id',default='interactive')
parser.add_argument('--tasks',nargs='*')
args=parser.parse_args()
configure()
assert torch.cuda.is_available() and torch.cuda.device_count()==1, (os.getenv('CUDA_VISIBLE_DEVICES'),torch.cuda.device_count())
assert 0 <= args.rank < args.world_size
outdir=ROOT/'results/predictions';outdir.mkdir(exist_ok=True)
specs=[s for s in task_specs() if not args.tasks or s['name'] in args.tasks]
codehash=hashlib.sha256(b''.join((ROOT/'scripts'/f).read_bytes() for f in ['infer.py','native_model.py','datasets_eval.py'])).hexdigest()
ATTENTION_BUDGET=16*1603**2
MAX_BATCH=32
model=None; current=None

def evaluate(batch, spec):
    pairs=[(r['a'],r['b']) for r in batch]
    f=tokenize(tok,pairs,spec['max_length']).to('cuda')
    expected=[min(len(r['a'])+len(r['b'])+3,spec['max_length'] or 10**9) for r in batch]
    assert f.attention_mask.sum(1).tolist()==expected, ('token_count',spec['name'])
    wild=model(f)
    if spec['kind']=='mutation':
        m=tokenize(tok,[(r['mutant'],r['b']) for r in batch],spec['max_length']).to('cuda')
        assert m.attention_mask.sum(1).tolist()==[min(len(r['mutant'])+len(r['b'])+3,spec['max_length'] or 10**9) for r in batch]
        mutant=model(m)
        difference=mutant-wild
        probability_log_ratio=F.logsigmoid(mutant.double())-F.logsigmoid(wild.double())
        vals=torch.stack([wild,mutant,difference,torch.sigmoid(difference),probability_log_ratio],dim=1).cpu().tolist()
        return [dict(row_id=r['row_id'],label=r['label'],original_tokens=r['length'],tokens=n,
                     wild_logit=v[0],mutant_logit=v[1],logit_difference=v[2],score=v[3],probability_log_ratio=v[4]) for r,n,v in zip(batch,expected,vals)]
    vals=torch.stack([wild,torch.sigmoid(wild)],dim=1).cpu().tolist()
    return [dict(row_id=r['row_id'],label=r['label'],original_tokens=r['length'],tokens=n,logit=v[0],score=v[1]) for r,n,v in zip(batch,expected,vals)]

def safe_evaluate(batch,spec):
    try:
        return evaluate(batch,spec)
    except torch.cuda.OutOfMemoryError:
        if len(batch)==1:raise
        torch.cuda.empty_cache()
        print(json.dumps(dict(event='split_oom_batch',task=spec['name'],n=len(batch))),flush=True)
        k=len(batch)//2
        return safe_evaluate(batch[:k],spec)+safe_evaluate(batch[k:],spec)

for spec in specs:
    rows=read_task(spec)
    rows.sort(key=lambda r:(-min(r['length'],spec['max_length'] or 10**9),r['row_id']))
    shard=rows[args.rank::args.world_size]
    config=dict(task=spec,rank=args.rank,world_size=args.world_size,code_sha256=codehash,input_sha256=sha256(spec['path']),attention_budget=ATTENTION_BUDGET,max_batch=MAX_BATCH)
    config_digest=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    dest=outdir/f'{spec["name"]}.rank{args.rank:02d}.csv'
    done=dest.with_suffix('.json')
    if done.exists():
        previous=json.loads(done.read_text())
        assert previous['config_digest']==config_digest, 'Existing output has different settings; do not overwrite silently.'
        assert previous['csv_sha256']==sha256(dest)
        print('Verified existing',dest.name,flush=True)
        continue
    if current != spec['model']:
        if model is not None:
            del model;gc.collect();torch.cuda.empty_cache()
        model,tok,modelmeta=load_model(spec['model'])
        current=spec['model']
        print(json.dumps(dict(event='model_loaded',rank=args.rank,model=modelmeta)),flush=True)
    print(json.dumps(dict(event='task_start',rank=args.rank,name=spec['name'],rows=len(shard),total_rows=len(rows))),flush=True)
    start=time.monotonic();last_log=start;count=0
    torch.cuda.reset_peak_memory_stats()
    tmp=dest.with_suffix('.csv.partial')
    fields=['row_id','label','original_tokens','tokens']+(['wild_logit','mutant_logit','logit_difference','score','probability_log_ratio'] if spec['kind']=='mutation' else ['logit','score'])
    with open(tmp,'w') as f, torch.inference_mode():
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
        index=0
        while index<len(shard):
            length=min(shard[index]['length'],spec['max_length'] or 10**9)
            size=min(MAX_BATCH,max(1,ATTENTION_BUDGET//length**2),len(shard)-index)
            batch=shard[index:index+size]
            predictions=safe_evaluate(batch,spec)
            writer.writerows(predictions);f.flush()
            count+=len(batch);index+=len(batch)
            now=time.monotonic()
            if now-last_log>=45:
                print(json.dumps(dict(event='progress',task=spec['name'],rank=args.rank,done=count,total=len(shard),elapsed_s=now-start)),flush=True)
                last_log=now
    assert count==len(shard)
    tmp.replace(dest)
    meta=dict(config=config,config_digest=config_digest,model=modelmeta,n=count,total_rows=len(rows),elapsed_seconds=time.monotonic()-start,
              gpu_peak_bytes=torch.cuda.max_memory_allocated(),hostname=socket.gethostname(),gpu_uuid=str(torch.cuda.get_device_properties(0).uuid),
              cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),job_id=args.job_id,csv_sha256=sha256(dest))
    done.write_text(json.dumps(meta,indent=2)+'\n')
    print(json.dumps(dict(event='task_done',task=spec['name'],rank=args.rank,n=count,seconds=meta['elapsed_seconds'])),flush=True)
print('ALL_TASKS_COMPLETE',flush=True)
