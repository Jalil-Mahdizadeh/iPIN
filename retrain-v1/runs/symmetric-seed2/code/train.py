"""Fully resumable all-layer human PPI fine-tuning. Test data are never evaluated here."""
import argparse
import contextlib
import datetime
import hashlib
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from sklearn.metrics import average_precision_score,roc_auc_score
from torch.nn.parallel import DistributedDataParallel as DDP
from data import PairData
from model import PairModel
from state import Checkpoints,atomic_json,sha256

def rank_ordered_average(group,bucket):
    """Make FP32 summation independent of NCCL ring topology and bucket rebuilds.

    All-gather transmits each rank's gradient without arithmetic. Addition then
    follows rank order on every device. This costs more communication than an
    all-reduce, but also makes a restart on another identical GPU node exact.
    """
    buf=bucket.buffer();world=dist.get_world_size(group)
    gathered=torch.empty(world*buf.numel(),device=buf.device,dtype=buf.dtype)
    work=dist.all_gather_into_tensor(gathered,buf,group=group,async_op=True)
    def reduce_in_order(future):
        shards=gathered.view(world,-1)
        buf.copy_(shards[0])
        for rank in range(1,world):buf.add_(shards[rank])
        return buf.div_(world)
    return work.get_future().then(reduce_in_order)

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output');p.add_argument('--tiny',action='store_true')
    p.add_argument('--max-updates',type=int);p.add_argument('--stop-after-updates',type=int)
    p.add_argument('--train-limit',type=int);p.add_argument('--val-limit',type=int);p.add_argument('--sleep-after-update',type=float,default=0.)
    args=p.parse_args();cfg=json.loads(Path(args.config).read_text());root=Path(cfg['root']);out=Path(args.output or root/'runs'/cfg['name'])
    out.mkdir(parents=True,exist_ok=True)
    rank=int(os.environ.get('RANK','0'));world=int(os.environ.get('WORLD_SIZE','1'));local=int(os.environ.get('LOCAL_RANK','0'))
    torch.cuda.set_device(local);device=torch.device('cuda',local)
    torch.set_num_threads(8);torch.manual_seed(cfg['seed']);random.seed(cfg['seed']);np.random.seed(cfg['seed'])
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    if world>1:dist.init_process_group('nccl',timeout=datetime.timedelta(minutes=20),device_id=device)
    attempt=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S')+f'-{os.environ.get("SLURM_JOB_ID","local")}'
    started=time.monotonic();stop_requested=False;stopfile=out/'REQUEST_STOP';eventfile=out/'events.jsonl'
    def event(kind,**details):
        if rank==0:
            e={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'attempt':attempt,'event':kind,**details}
            with eventfile.open('a') as f:f.write(json.dumps(e,allow_nan=False)+'\n');f.flush()
            print(json.dumps(e,allow_nan=False),flush=True)
    def handle(signum,frame):
        nonlocal stop_requested
        stop_requested=True
    signal.signal(signal.SIGTERM,handle);signal.signal(signal.SIGUSR1,handle);signal.signal(signal.SIGINT,handle)
    def stopping():
        flag=torch.tensor(int(stop_requested or stopfile.exists() or (out/'REQUEST_REQUEUE').exists()),device=device)
        if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MAX)
        return bool(flag.item())
    manifest=json.loads((root/'data/prepared/manifest.json').read_text())
    if rank==0:
        for name,h in manifest['files'].items():assert sha256(root/'data/prepared'/name)==h,(name,'data hash changed')
        for r in json.loads((root/'provenance/downloads.json').read_text()):
            if r['repository'].startswith('facebook/'):assert sha256(root/r['local_path'])==r['sha256'],r['file']
    if world>1:dist.barrier()
    train=PairData(root/'data/prepared','train');val=PairData(root/'data/prepared','val')
    def qualification_subset(data,limit):
        if limit is None:return
        assert 2<=limit<=len(data)
        rng=np.random.default_rng(1729)
        selected=np.concatenate([rng.choice(np.flatnonzero(data.rows[:,2]==y),limit//2+(limit%2 if y else 0),replace=False) for y in [0,1]])
        selected.sort();data.rows=data.rows[selected];data.lengths=data.lengths[selected]
    qualification_subset(train,args.train_limit);qualification_subset(val,args.val_limit)
    steps_per_epoch=math.ceil(len(train)/cfg['global_pairs_per_update'])
    total_steps=args.max_updates or cfg['epochs']*steps_per_epoch
    contract={'configuration':cfg,'tiny':args.tiny,'train_limit':args.train_limit,'val_limit':args.val_limit,
              'total_steps':total_steps,'data_manifest_sha256':sha256(root/'data/prepared/manifest.json'),
              'initialization_manifest_sha256':sha256(root/'provenance/downloads.json'),
              'code':{f:sha256(Path(__file__).with_name(f)) for f in ['train.py','model.py','data.py','state.py']},
              'torch':torch.__version__,'cuda':torch.version.cuda}
    fingerprint=hashlib.sha256(json.dumps(contract,sort_keys=True).encode()).hexdigest()
    if rank==0:
        lock=out/'RUNNING.lock'
        # Per-run OS lock prevents two jobs updating the same output concurrently.
        import fcntl
        lock_handle=lock.open('w');fcntl.flock(lock_handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lock_handle.write(attempt);lock_handle.flush()
        if (out/'contract.json').exists():
            assert json.loads((out/'contract.json').read_text())['fingerprint']==fingerprint,'Refusing to overwrite a different run contract'
        atomic_json(out/'contract.json',{'fingerprint':fingerprint,**contract})
        (out/'stopped.json').unlink(missing_ok=True)
    model=PairModel(root/'assets/esm2',mode=cfg['mode'],backend=cfg['attention_backend'],tiny=args.tiny).to(device)
    decay=[p for p in model.parameters() if p.requires_grad and p.ndim>=2]
    no_decay=[p for p in model.parameters() if p.requires_grad and p.ndim<2]
    optimizer=torch.optim.AdamW([{'params':decay,'weight_decay':cfg['weight_decay']},{'params':no_decay,'weight_decay':0.}],lr=cfg['learning_rate'],foreach=False)
    checkpoints=Checkpoints(out,rank,world,fingerprint)
    state,selection=checkpoints.load(model,optimizer)
    if state is None:state={'update':0,'epoch':0,'next_chunk':0,'examples_seen':0,'best_ap':None,'best_update':None,'total_steps':total_steps,
                           'pending_validation':False,'last_validation_update':None}
    assert state['total_steps']==total_steps
    runtime={'rank':rank,'uuid':str(torch.cuda.get_device_properties(device).uuid),'device':torch.cuda.get_device_name(device)}
    devices=[None]*world
    if world>1:dist.all_gather_object(devices,runtime)
    else:devices=[runtime]
    assert len({d['uuid'] for d in devices})==world,devices
    event('resumed' if selection else 'initialized',state=state,world_size=world,devices=devices,checkpoint=selection,parameters=sum(x.numel() for x in model.parameters()))
    wrapper=DDP(model,device_ids=[local],broadcast_buffers=False,find_unused_parameters=False,gradient_as_bucket_view=True) if world>1 else model
    if world>1:wrapper.register_comm_hook(state=None,hook=rank_ordered_average)
    optimizer.zero_grad(set_to_none=True)
    if selection is None:checkpoints.save(model,optimizer,state)
    last_checkpoint=time.monotonic();last_committed=state['update']

    def finish_stop(reason):
        if rank==0:atomic_json(out/'stopped.json',{'state':state,'attempt':attempt,'reason':reason,'checkpoint_committed':True})
        event('stopped',state=state,reason=reason)
        if world>1:dist.destroy_process_group()
        return 0

    def validate():
        model.eval();pred=[];interrupted=False
        indices=np.arange(rank,len(val),world,dtype=np.int64)
        # Sorting only changes evaluation batching, not scored rows or labels.
        indices=indices[np.argsort(val.lengths[indices],kind='stable')]
        with torch.inference_mode():
            for micro in val.microbatches(indices,cfg['eval_token_budget'],cfg['eval_max_pairs']):
                if stop_requested or stopfile.exists() or (out/'REQUEST_REQUEUE').exists():interrupted=True;break
                features=val.batch(micro,0,cfg['seed'],False,device)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    logits=model(features['input_ids'],features['attention_mask'])
                assert torch.isfinite(logits).all()
                z=logits.float().cpu().numpy()
                pred.extend((int(i),int(val.rows[i,2]),float(a),float(b)) for i,(a,b) in zip(micro,z))
        gathered=[None]*world
        if world>1:dist.all_gather_object(gathered,{'rows':pred,'interrupted':interrupted})
        else:gathered=[{'rows':pred,'interrupted':interrupted}]
        metrics=None;best=False
        if not any(x['interrupted'] for x in gathered):
            a=np.asarray([r for shard in gathered for r in shard['rows']],dtype=np.float64);a=a[np.argsort(a[:,0])]
            assert len(a)==len(val) and np.array_equal(a[:,0],np.arange(len(val)))
            pooled=a[:,2:4].mean(1);score=pooled if cfg['mode']=='symmetric' else a[:,2]
            prob=1/(1+np.exp(-np.clip(score,-700,700)))
            metrics={'rows':len(a),'ap':float(average_precision_score(a[:,1],score)),
                     'auroc':float(roc_auc_score(a[:,1],score)) if len(np.unique(a[:,1]))==2 else None,
                     'pooled_ap':float(average_precision_score(a[:,1],pooled)),
                     'brier':float(np.mean((prob-a[:,1])**2)),
                     'order_logit_gap_mean':float(np.abs(a[:,2]-a[:,3]).mean())}
            best=state['best_ap'] is None or metrics['ap']>state['best_ap']
            if best:state['best_ap']=metrics['ap'];state['best_update']=state['update']
            state['pending_validation']=False;state['last_validation_update']=state['update']
            if rank==0:
                (out/'validation').mkdir(exist_ok=True)
                np.savez_compressed(out/'validation'/f'update-{state["update"]:09d}.npz',predictions=a)
                event('validation',update=state['update'],metrics=metrics,best=best)
        model.train();return metrics,best

    # A signal during (or just before) validation must not silently lose that
    # model-selection opportunity, including validation at the final update.
    if state['pending_validation']:
        metrics,best=validate() if not stopping() else (None,False)
        checkpoints.save(model,optimizer,state,best=best)
        last_checkpoint=time.monotonic();last_committed=state['update']
        event('checkpoint_committed',update=state['update'],best=best,resumed_validation=True)
        if state['pending_validation'] or stopping():return finish_stop('signal_or_REQUEST_STOP')

    for epoch in range(state['epoch'],cfg['epochs']):
        plan=train.plan(epoch,cfg['seed'],cfg['global_pairs_per_update'])
        first_chunk=state['next_chunk'] if state['epoch']==epoch else 0
        for chunk_id in range(first_chunk,len(plan)):
            if state['update']>=total_steps:break
            if stopping():
                checkpoints.save(model,optimizer,state)
                return finish_stop('signal_or_REQUEST_STOP')
            batch_indices=plan[chunk_id];local_indices=batch_indices[rank::world]
            dummy=not len(local_indices)
            if dummy:local_indices=batch_indices[:1]
            micros=list(train.microbatches(local_indices,cfg['train_token_budget'],cfg['train_max_pairs']))
            model.train();t0=time.monotonic();optimizer.zero_grad(set_to_none=True)
            statistics=torch.zeros(3,device=device,dtype=torch.float64)
            for m,micro in enumerate(micros):
                sync=contextlib.nullcontext() if world==1 or m==len(micros)-1 else wrapper.no_sync()
                with sync:
                    features=train.batch(micro,epoch,cfg['seed'],cfg['mlm_weight']>0,device)
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        loss,cls,mlm,z=wrapper(**features,class_weight=cfg['classification_weight'],mlm_weight=cfg['mlm_weight'])
                        scaled=loss*(0. if dummy else world/len(batch_indices))
                    scaled.backward()
                if not dummy:
                    statistics+=torch.stack([loss.detach().double(),cls.double(),mlm.double()])
                del features,loss,scaled,cls,mlm,z
            grad_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),cfg['clip_grad_norm'],error_if_nonfinite=True)
            step=state['update'];warmup=min(cfg['warmup_updates'],max(1,total_steps-1))
            factor=(step+1)/warmup if step<warmup else max(0.,(total_steps-step)/(total_steps-warmup))
            lr=cfg['learning_rate']*factor
            for group in optimizer.param_groups:group['lr']=lr
            optimizer.step();optimizer.zero_grad(set_to_none=True)
            if world>1:dist.all_reduce(statistics)
            state['update']+=1;state['examples_seen']+=len(batch_indices)
            state['epoch']=epoch;state['next_chunk']=chunk_id+1
            if state['next_chunk']==len(plan):state['epoch']=epoch+1;state['next_chunk']=0
            elapsed=time.monotonic()-t0
            status={'state':state.copy(),'last_committed_update':last_committed,'attempt':attempt,
                    'seconds_since_launch':time.monotonic()-started,'world_size':world,'phase':'training'}
            if rank==0 and (state['update']<=3 or state['update']%cfg['log_every_updates']==0):
                atomic_json(out/'status.json',status)
                event('update',update=state['update'],epoch=epoch,chunk=chunk_id,lr=lr,
                      mean_loss=float(statistics[0]/len(batch_indices)),mean_bce=float(statistics[1]/len(batch_indices)),
                      mean_mlm=float(statistics[2]/len(batch_indices)),grad_norm=float(grad_norm),seconds=elapsed,
                      global_pairs=len(batch_indices),max_tokens=int(train.lengths[batch_indices].max()),
                      gpu_peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30)
            forced=args.stop_after_updates is not None and state['update']>=args.stop_after_updates
            stopped=stopping() or forced
            final=state['update']>=total_steps
            due_val=state['update']%cfg['validate_every_updates']==0 or final or state['next_chunk']==0
            if due_val:state['pending_validation']=True
            metrics,best=validate() if due_val and not stopped else (None,False)
            stopped=stopped or stopping()
            due_checkpoint=stopped or final or due_val or state['update']%cfg['checkpoint_every_updates']==0 or time.monotonic()-last_checkpoint>=cfg['checkpoint_minutes']*60
            if due_checkpoint:
                checkpoints.save(model,optimizer,state,best=best);last_checkpoint=time.monotonic();last_committed=state['update']
                event('checkpoint_committed',update=state['update'],best=best)
            if stopped:
                return finish_stop('qualification_stop' if forced else 'signal_or_REQUEST_STOP')
            if args.sleep_after_update:time.sleep(args.sleep_after_update)
        if state['update']>=total_steps:break
    # Verify all ranks hold the same final trained model, including optimizer-updated layers.
    h=hashlib.sha256()
    for name,value in model.state_dict().items():h.update(name.encode());h.update(value.detach().cpu().numpy().tobytes())
    hashes=[None]*world
    if world>1:dist.all_gather_object(hashes,h.hexdigest())
    else:hashes=[h.hexdigest()]
    assert len(set(hashes))==1,hashes
    if rank==0:
        atomic_json(out/'completed.json',{'state':state,'fingerprint':fingerprint,'test_evaluated':False,'model_hashes_by_rank':hashes})
        atomic_json(out/'status.json',{'state':state,'last_committed_update':state['update'],'phase':'complete','attempt':attempt})
    event('complete',state=state)
    if world>1:dist.destroy_process_group()
    return 0

if __name__=='__main__':sys.exit(main())
