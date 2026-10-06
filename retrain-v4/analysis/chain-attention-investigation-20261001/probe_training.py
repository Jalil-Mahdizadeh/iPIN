"""Frozen-weight masked-residue and full-model gradient/recomputation diagnostics."""
import argparse
import gc
import json
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from diagnose import (ROOT,CODE,OUT,PairModel,PairData,attention_layers,save,sha256,core_hashes)
from model_esm2 import checkpoint_contexts
import numpy as np
import torch
import torch.nn.functional as F


def checkpointing(model,backbone,on):
    if backbone=='esmc':model.gradient_checkpointing=on
    elif on:model.esm_mask.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False,'context_fn':checkpoint_contexts})
    else:model.esm_mask.gradient_checkpointing_disable()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backbone',choices=['esm2','esmc'],required=True)
    args=parser.parse_args(); b=args.backbone
    torch.set_num_threads(8);torch.manual_seed(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    cfg=json.loads((ROOT/'configs'/(b+'-chain-aware-official-seed2.json')).read_text())
    data=PairData(ROOT/'data/prepared','official','train')
    selected=json.loads((OUT/f'{b}-pretrained-diagnosis.json').read_text())['training_pair_indices']
    model=PairModel(ROOT/'assets'/b,cfg).cuda()
    layers=attention_layers(model,b)
    result={'backbone':b,'training_pair_indices':selected,'no_optimizer_steps':True,'test_data_used':False,
            'masked_residue_protocol':'15% of actual residues independently masked in both orientations with fixed seed; diagnostics, not a held-out benchmark',
            'stages':{},'code':core_hashes(CODE),'script_sha256':sha256(Path(__file__))}
    model.eval()
    control_batch=data.batch(np.array(selected[:1]),np.zeros(1,dtype=int),2,False,'cuda')
    with torch.no_grad():
        for layer in layers:layer.pair_attention='standard'
        standard=model.encode(control_batch['clean_ids'],control_batch['attention_mask'],control_batch['chain_ids'])
        for layer in layers:layer.pair_attention='chain_aware'
        all_one=model.encode(control_batch['clean_ids'],control_batch['attention_mask'],torch.zeros_like(control_batch['chain_ids']))
    result['fp32_all_one_chain_control']={'relative_l2':float((standard-all_one).norm()/standard.norm()),
        'max_abs':float((standard-all_one).abs().max())}
    del standard,all_one,control_batch
    for stage in ['pretrained','trained_chain_best1000']:
        if stage!='pretrained':
            path=ROOT/'runs'/(b+'-chain-aware-official-seed2')/'checkpoints/update-000001000.pt'
            meta=json.loads(path.with_suffix('.pt.json').read_text()); assert sha256(path)==meta['sha256']
            payload=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
            assert payload['training_state']['update']==1000
            # Measure real encoder updates before replacing the pretrained weights.
            delta2=0.;base2=0.;changed=0;count=0
            for name,p in model.named_parameters():
                if not name.startswith(('classifier.','readout_')) and p.requires_grad:
                    initial=p.detach().cpu();trained=payload['model'][name]
                    delta2+=float((trained-initial).double().square().sum())
                    base2+=float(initial.double().square().sum())
                    changed+=int((trained!=initial).sum());count+=p.numel()
            names=[n for ndim in [True,False] for n,p in model.named_parameters() if p.requires_grad and (p.ndim>=2)==ndim]
            opt=payload['optimizer'];mapping=[i for group in opt['param_groups'] for i in group['params']]
            assert len(names)==len(mapping)
            opt_steps={n:float(opt['state'][i]['step']) for n,i in zip(names,mapping)}
            assert set(opt_steps.values())=={1000.}
            nonzero_moments=sum(int(torch.count_nonzero(opt['state'][i]['exp_avg'])) for i in mapping)
            model.load_state_dict(payload['model'],strict=True)
            result['checkpoint']={'path':str(path),'sha256':meta['sha256'],'update':1000,
                'encoder_relative_weight_l2_change':(delta2/base2)**.5,'encoder_changed_parameter_elements':changed,
                'encoder_trainable_parameter_elements':count,'optimizer_parameter_tensors':len(names),
                'optimizer_steps_all_1000':True,'optimizer_exp_avg_nonzero_elements':nonzero_moments}
            del payload;gc.collect()
        stage_result={'masked_residue':{},'ppi_gradient':{}}
        model.eval()
        for mode in ['standard','chain_aware']:
            for layer in layers:layer.pair_attention=mode
            losses=[];tokens=0;loss_sum=0.;correct=0
            for pair in selected:
                batch=data.batch(np.array([pair]),np.zeros(1,dtype=int),2,False,'cuda')
                rng=np.random.default_rng(np.random.SeedSequence([20261001,pair,57]))
                chosen=torch.tensor(rng.random(tuple(batch['clean_ids'].shape))<.15,device='cuda')&batch['residue_mask']
                ids=batch['clean_ids'].clone();targets=ids[chosen].clone();ids[chosen]=32
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):
                    hidden=model.encode(ids,batch['attention_mask'],batch['chain_ids'])
                    head=model.esm_mask.lm_head if b=='esm2' else model.esmc.sequence_head
                    logits=head(hidden[chosen])
                values=F.cross_entropy(logits.float(),targets,reduction='none')
                losses.append({'pair':pair,'tokens':len(targets),'mean_ce':float(values.mean())})
                tokens+=len(targets);loss_sum+=float(values.sum());correct+=int((logits.argmax(-1)==targets).sum())
                del hidden,logits
            stage_result['masked_residue'][mode]={'mean_ce':loss_sum/tokens,'tokens':tokens,'accuracy':correct/tokens,'pairs':losses}
            print(json.dumps({'backbone':b,'stage':stage,'mode':mode,'masked_residue':stage_result['masked_residue'][mode]}),flush=True)
        # The trained checkpoint tests whether gradients pass through the actual
        # full encoder/head and whether recomputation changes them. No steps.
        if stage!='pretrained':
            for mode in ['standard','chain_aware']:
                for layer in layers:layer.pair_attention=mode
                reference=None;details=[]
                for enabled in [False,True]:
                    checkpointing(model,b,enabled);model.train();model.zero_grad(set_to_none=True)
                    loss_sum=0.
                    for pair in selected[:2]:
                        batch=data.batch(np.array([pair]),np.zeros(1,dtype=int),2,False,'cuda')
                        with torch.autocast('cuda',dtype=torch.bfloat16):
                            loss,*_=model(**batch)
                            loss=loss/2
                        loss.backward();loss_sum+=float(loss.detach())
                    groups={};grads={};missing=[]
                    for name,p in model.named_parameters():
                        if not p.requires_grad:continue
                        if p.grad is None:missing.append(name);continue
                        g=p.grad.detach();assert torch.isfinite(g).all(),name
                        family='classifier' if name.startswith('classifier.') else 'residual_head' if name.startswith('readout_') else 'encoder'
                        groups[family]=groups.get(family,0.)+float(g.double().square().sum())
                        grads[name]=g.cpu().clone()
                    item={'checkpointing':enabled,'mean_classification_loss_times_10':loss_sum,
                          'gradient_norms':{k:v**.5 for k,v in groups.items()},'missing_trainable_gradients':missing}
                    if reference is not None:
                        assert grads.keys()==reference.keys()
                        error2=0.;base2=0.;worst=0.;exact=True
                        for name,g in grads.items():
                            d=g-reference[name]
                            exact=exact and torch.equal(g,reference[name])
                            error2+=float(d.double().square().sum());base2+=float(reference[name].double().square().sum())
                            worst=max(worst,float(d.abs().max()))
                        item.update(all_gradients_bitwise_identical=exact,gradient_relative_l2=(error2/base2)**.5,max_abs_gradient_difference=worst)
                    else:reference=grads
                    details.append(item);print(json.dumps({'backbone':b,'stage':stage,'mode':mode,'gradient':item}),flush=True)
                stage_result['ppi_gradient'][mode]=details
                del reference,grads;gc.collect();model.zero_grad(set_to_none=True)
        result['stages'][stage]=stage_result
        save(OUT/f'{b}-training-probes.json',result)
    print(json.dumps({'finished':True,'backbone':b,'checkpoint':result['checkpoint']}),flush=True)


if __name__=='__main__':main()
