"""Full pretrained longest-pair execution and representative throughput checks."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import torch
from contracts import core_hashes,sha256
from data import PairData
from model import PairModel
from state import atomic_json

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backbone',choices=['esm2','esmc'],required=True)
    args=parser.parse_args();backbone=args.backbone
    cfg=json.loads((ROOT/'configs'/f'{backbone}-native-ilp-seed2.json').read_text())
    torch.set_num_threads(8);torch.manual_seed(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    train,val=[PairData(ROOT/'data/prepared',backbone,s) for s in ['train','val']]
    model=PairModel(ROOT/'assets'/backbone,cfg).cuda()
    groups=[dict(params=[p for p in model.parameters() if p.requires_grad and p.ndim>=2],weight_decay=.01),
            dict(params=[p for p in model.parameters() if p.requires_grad and p.ndim<2],weight_decay=0.)]
    opt=torch.optim.AdamW(groups,lr=cfg['learning_rate'],foreach=False)
    records=[]
    for stage,data,training in [('longest-training',train,True),('longest-validation',val,False)]:
        index=np.array([int(np.argmax(data.lengths))]);features=data.batch(index,np.zeros(1,dtype=int),2,training,'cuda')
        assert features['clean_ids'].shape==(2,int(data.lengths[index[0]]))
        model.train(training);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
        if training:
            with torch.autocast('cuda',dtype=torch.bfloat16):loss,cls,mlm,z=model(**features)
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
            decoder=model.esm_mask.lm_head if backbone=='esm2' else model.esmc.sequence_head
            assert sum(float(p.grad.abs().sum()) for p in decoder.parameters())>0
            opt.step();opt.zero_grad(set_to_none=True)
            value=float(loss.detach());del loss,cls,mlm,z
        else:
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):z=model(**features,compute_loss=False)
            assert torch.isfinite(z).all();value=z.float().cpu().tolist();del z
        torch.cuda.synchronize()
        record=dict(stage=stage,tokens=int(data.lengths[index[0]]),orientations=2,
                    source_row=int(data.rows[index[0],3]),seconds=time.monotonic()-start,
                    peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                    peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,finite_output=value)
        records.append(record);print(json.dumps(record),flush=True);del features
    # Twelve deterministic length quantiles cover the body and long-pair tail.
    order=np.argsort(train.lengths,kind='stable')
    qs=np.array([.05,.15,.25,.35,.45,.55,.65,.75,.85,.90,.95,.99])
    ids=order[np.floor(qs*(len(order)-1)).astype(int)]
    model.train();opt.zero_grad(set_to_none=True)
    total_masks=train.masked_token_count(ids,np.zeros(len(ids),dtype=int),2)
    torch.cuda.synchronize();started=time.monotonic()
    timings=[]
    for idx in ids:
        t=time.monotonic();batch=train.batch(np.array([idx]),np.zeros(1,dtype=int),2,True,'cuda')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            loss,*_=model(**batch,pair_normalizer=len(ids),mlm_normalizer=total_masks)
        loss.backward();torch.cuda.synchronize()
        timings.append(dict(tokens=int(train.lengths[idx]),seconds=time.monotonic()-t))
        del batch,loss
    norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);opt.step();opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize();elapsed=time.monotonic()-started
    report=dict(passed=True,backbone=backbone,profiles=records,representative_training_pair_timings=timings,
                representative_pairs=len(ids),representative_training_seconds=elapsed,
                measured_training_includes_masked_classification_mlm_backward_and_optimizer=True,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                no_sequence_truncation=True,decoder_gradients_verified=True,
                test_data_used=False,candidate_checkpoints_saved=False,disposable_optimizer_steps=2,
                code=core_hashes(ROOT/'scripts'),source_sha256={'scripts/qualify_profile.py':sha256(Path(__file__))},
                device=torch.cuda.get_device_name(),device_memory_gib=torch.cuda.get_device_properties(0).total_memory/2**30)
    atomic_json(ROOT/'qualification'/f'full-length-{backbone}.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
