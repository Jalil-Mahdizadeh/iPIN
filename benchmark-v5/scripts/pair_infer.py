"""Qualified, resumable inference for the two iPIN models and native PLM-interact."""
import argparse
import fcntl
import importlib.util
import json
import os
import socket
import sys
import time
import numpy as np
import torch
from bench_utils import ROOT,PROJECT,atomic,cuda,load_npz,now,read,record,save_npz,sha

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);result=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result);return result

Data=module('benchmark_v5_data',ROOT/'scripts/v5_model/data.py').PairData

def data_arrays(rows,tokens,offsets):
    result=Data.__new__(Data);result.rows=rows;result.tokens=tokens;result.offsets=offsets
    result.lengths=np.diff(offsets)[rows[:,0]]+np.diff(offsets)[rows[:,1]]+3
    result._plans={};return result

def test_data():
    return data_arrays(np.load(ROOT/'data/union.npy'),np.load(ROOT/'data/tokens.npy'),np.load(ROOT/'data/offsets.npy'))

def signature(name):
    entry=read(ROOT/'provenance/selection.json')['models'][name]
    code=[ROOT/'scripts/pair_infer.py',ROOT/'scripts/bench_utils.py',ROOT/'scripts/container.sh']
    code+=list((ROOT/'scripts'/('native_model' if name=='native-plm' else 'v5_model')).glob('*.py'))
    body={'entry':entry,'prepared_sha256':sha(ROOT/'provenance/prepared.json'),
          'code':{str(p.relative_to(ROOT)):sha(p) for p in code},'torch':torch.__version__,
          'precision':'FP32 parameters, BF16 autocast, TF32 disabled','token_budget':16384,'max_pairs':8,'shards':4,'commit_rows':512}
    import hashlib
    return hashlib.sha256(json.dumps(body,sort_keys=True).encode()).hexdigest()

def load_model(name,device):
    entry=read(ROOT/'provenance/selection.json')['models'][name]
    path=entry['checkpoint']['path'];assert sha(path)==entry['checkpoint']['sha256']
    checkpoint=torch.load(path,map_location='cpu',mmap=True,weights_only=name=='native-plm')
    if name=='native-plm':
        cls=module('benchmark_native_model',ROOT/'scripts/native_model/model.py').PairModel
        model=cls(entry['base'],mode='reference',backend='efficient');state=checkpoint
    else:
        sys.path.insert(0,str(ROOT/'scripts/v5_model'))
        from model import PairModel
        assert checkpoint['training_state']['update']==entry['update']
        assert checkpoint['fingerprint']==entry['source_manifest']['fingerprint']
        model=PairModel(entry['base'],entry['configuration']);state=checkpoint['model']
    model.load_state_dict(state,strict=True)
    assert all(torch.equal(v,state[k]) for k,v in model.state_dict().items())
    if hasattr(model,'esm_mask'):model.esm_mask.gradient_checkpointing_disable()
    if hasattr(model,'gradient_checkpointing'):model.gradient_checkpointing=False
    model.eval().requires_grad_(False).to(device)
    del checkpoint,state
    return model

@torch.inference_mode()
def predict(model,name,data,ids,device):
    features=data.batch(ids,np.zeros(len(ids),dtype=np.int64),2,False,device)
    assert features['attention_mask'].sum(1).cpu().tolist()==[int(data.lengths[i]) for i in ids for _ in range(2)]
    with torch.autocast('cuda',dtype=torch.bfloat16):
        if name=='native-plm':z=model(features['clean_ids'],features['attention_mask'])
        else:z=model(**features,compute_loss=False)
    z=z.float().cpu().numpy().astype(np.float64)
    assert z.shape==(len(ids),2) and np.isfinite(z).all()
    return z

def qualify(name,model,device):
    from scipy.special import expit
    if name=='native-plm':
        path=PROJECT/'benchmark-v4/data'
        data=data_arrays(np.load(path/'val.npy'),np.load(path/'tokens.npy'),np.load(path/'offsets.npy'))
        gold=load_npz(PROJECT/'benchmark-v4/results/native-bernett-val.npz')['predictions']
    else:
        backbone=name.removeprefix('ipin-');path=PROJECT/'retrain-v5/data/prepared'/backbone
        data=data_arrays(np.load(path/'val.npy'),np.load(path/'tokens.npy'),np.load(path/'offsets.npy'))
        gold=load_npz(ROOT/'data'/(name+'-selected-dev.npz'))['predictions']
    indices=np.arange(0,len(data),4);indices=indices[np.argsort(data.lengths[indices],kind='stable')]
    batches=list(data.microbatches(indices,16384,8));cases=[]
    for j in sorted({0,len(batches)//2,len(batches)-1}):
        ids=indices[batches[j]];z=predict(model,name,data,ids,device);expected=gold[ids,2:4]
        logit_error=float(np.max(np.abs(z-expected)));probability_error=float(np.max(np.abs(expit(z)-expit(expected))))
        assert logit_error<=0.05 and probability_error<=0.005,(name,logit_error,probability_error)
        cases.append({'rows':ids.tolist(),'max_tokens':int(data.lengths[ids].max()),'max_logit_error':logit_error,'max_probability_error':probability_error})
    atomic(ROOT/'qualification'/(name+'.json'),{'passed':True,'at_utc':now(),'fingerprint':signature(name),
        'source':'Fresh full-size model forwards against the frozen selected DEV predictions; production batching reproduced',
        'cases':cases,'test_metrics_read':False,'torch':torch.__version__,'device':torch.cuda.get_device_name(device)})
    print(json.dumps({'event':'qualification_passed','model':name,'cases':cases}),flush=True)

def infer(name,model,device,rank,world):
    assert world==4
    contract=signature(name);q=read(ROOT/'qualification'/(name+'.json'));assert q['passed'] and q['fingerprint']==contract
    data=test_data();all_ids=np.arange(52048 if name=='native-plm' else 0,len(data))
    ids=all_ids[rank::world];ids=ids[np.argsort(data.lengths[ids],kind='stable')]
    out=ROOT/'predictions'/name;out.mkdir(exist_ok=True)
    lock=(out/f'rank-{rank}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    started=time.monotonic();records=[]
    for chunk,start in enumerate(range(0,len(ids),512)):
        subset=ids[start:start+512];path=out/f'rank-{rank:02d}-chunk-{chunk:04d}.npz';meta=path.with_suffix('.json')
        if meta.exists():
            old=read(meta);assert old['fingerprint']==contract and sha(path)==old['file']['sha256']
            saved=load_npz(path);assert np.array_equal(saved['indices'],subset)
        else:
            if (ROOT/'REQUEST_STOP').exists():raise SystemExit(75)
            logits=np.empty((len(subset),2),np.float64)
            for positions in data.microbatches(subset,16384,8):logits[positions]=predict(model,name,data,subset[positions],device)
            save_npz(path,indices=subset,logits=logits)
            atomic(meta,{'fingerprint':contract,'file':record(path),'rows':len(subset),'at_utc':now()})
        records.append(record(meta))
        print(json.dumps({'event':'chunk','model':name,'rank':rank,'rows':min(start+512,len(ids)),'total':len(ids),'seconds':time.monotonic()-started}),flush=True)
    atomic(out/f'rank-{rank:02d}.done.json',{'model':name,'rank':rank,'world':world,'rows':len(ids),'fingerprint':contract,
        'chunks':records,'at_utc':now(),'hostname':socket.gethostname(),'gpu_uuid':str(torch.cuda.get_device_properties(device).uuid),
        'seconds':time.monotonic()-started,'gpu_peak_gib':torch.cuda.max_memory_allocated()/2**30})

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['ipin-esm2','ipin-esmc','native-plm'],required=True);p.add_argument('--qualify',action='store_true')
    a=p.parse_args();device=cuda(int(os.environ.get('LOCAL_RANK','0')));model=load_model(a.model,device)
    if a.qualify:qualify(a.model,model,device)
    else:infer(a.model,model,device,int(os.environ.get('RANK','0')),int(os.environ.get('WORLD_SIZE','1')))

if __name__=='__main__':main()
