"""Read-only model diagnostics on training examples; no optimizer steps or test data."""
import argparse
import contextlib
import gc
import hashlib
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path('/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v4')
CODE = ROOT / 'releases/20261001-production/code'
OUT = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
import numpy as np
import torch
import torch.nn.functional as F
from data import PairData
from model import PairModel
from model_esm2 import chain_context, expand_chain_qk
from contracts import core_hashes, sha256


def save(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def attention_layers(model, backbone):
    if backbone == 'esm2':
        return [layer.attention.self for layer in model.esm_mask.esm.encoder.layer]
    return [block.attn for block in model.esmc.transformer.blocks]


def qkv(layer, x, backbone):
    if backbone == 'esm2':
        q = layer.transpose_for_scores(layer.query(x)) * layer.attention_head_size ** -.5
        k = layer.transpose_for_scores(layer.key(x))
        v = layer.transpose_for_scores(layer.value(x))
        qc, kc = q.clone(), k.clone()
        qr, kr = layer.rotary_embeddings(q, k)
        scale = 1.
    else:
        q, k, v = layer.layernorm_qkv(x).chunk(3, -1)
        q, k = layer.q_ln(q).to(q.dtype), layer.k_ln(k).to(k.dtype)
        qc, kc = q.clone(), k.clone()
        qr, kr = layer._apply_rotary(q, k)
        assert torch.equal(q, qc) and torch.equal(k, kc), 'RoPE mutated unrotated input'
        heads = lambda z: z.unflatten(-1, (layer.n_heads, layer.d_head)).transpose(1, 2)
        q, k, v, qr, kr = map(heads, (q, k, v, qr, kr))
        scale = layer.d_head ** -.5
    if backbone == 'esm2':
        assert torch.equal(q, qc) and torch.equal(k, kc), 'RoPE mutated unrotated input'
    return [z.to(v.dtype) for z in (q, k, v, qr, kr)], scale


def dense(layer, x, valid, chains, backbone):
    (q, k, v, qr, kr), scale = qkv(layer, x, backbone)
    # Independent explicit score construction, FP32 accumulation and softmax.
    with torch.autocast('cuda', enabled=False):
        rot = qr.float() @ kr.float().transpose(-1, -2)
        raw = q.float() @ k.float().transpose(-1, -2)
        same = chains[:, None, :, None] == chains[:, None, None, :]
        scores = torch.where(same, rot, raw) if layer.pair_attention == 'chain_aware' else rot
        probs = (scores * scale).masked_fill(~valid[:, None, None, :], -torch.inf).softmax(-1)
        context = (probs @ v.float()).to(v.dtype)
    context = context.transpose(1, 2).contiguous().flatten(-2)
    return layer.out_proj(context) if backbone == 'esmc' else context


def kernel_checks(layers, captures, batch, backbone):
    records = []
    for idx, x in captures.items():
        layer = layers[idx]
        # Small, real activation slices with both chains and padding retained.
        n = min(x.shape[1], 192)
        chosen = torch.linspace(0, x.shape[1]-1, n, device='cuda').round().long()
        x = x[:, chosen].detach().float()
        valid = batch['attention_mask'][:, chosen].bool()
        chains = batch['chain_ids'][:, chosen]
        for mode in ['standard', 'chain_aware']:
            layer.pair_attention, layer.pair_backend = mode, 'efficient'
            for bf16 in [False, True]:
                a = x.clone().requires_grad_(True)
                b = x.clone().requires_grad_(True)
                with chain_context(chains), torch.autocast('cuda', dtype=torch.bfloat16, enabled=bf16):
                    if backbone == 'esmc':
                        actual = layer(a, valid)
                    else:
                        mask = (~valid[:, None, None, :]).float() * torch.finfo(torch.float32).min
                        actual = layer(a, attention_mask=mask)[0]
                    reference = dense(layer, b, valid, chains, backbone)
                torch.manual_seed(814 + idx)
                probe = torch.randn_like(actual.float()) * valid[..., None]
                parameters = [p for p in layer.parameters() if p.requires_grad]
                ga = torch.autograd.grad((actual.float()*probe).sum(), [a] + parameters)
                gb = torch.autograd.grad((reference.float()*probe).sum(), [b] + parameters)
                def rel(u, v):
                    return float((u.float()-v.float()).norm() / v.float().norm().clamp_min(1e-12))
                errors = [rel(u, v) for u, v in zip(ga, gb)]
                record = {'layer': idx, 'mode': mode, 'precision': 'bf16' if bf16 else 'fp32',
                          'forward_relative_l2': rel(actual[valid], reference[valid]),
                          'input_gradient_relative_l2': errors[0],
                          'parameter_gradient_relative_l2_max': max(errors[1:]),
                          'all_gradient_relative_l2': errors,
                          'forward_max_abs': float((actual[valid].float()-reference[valid].float()).abs().max())}
                records.append(record)
                print(json.dumps({'kernel': record}), flush=True)
                del actual, reference, a, b, ga, gb
    return records


def probe_attention(layer, x, batch, backbone, layer_index, records, context):
    (q, k, v, qr, kr), scale = qkv(layer, x, backbone)
    with torch.autocast('cuda', enabled=False):
        for i in range(len(x)):
            valid = batch['attention_mask'][i].bool()
            residues = batch['residue_mask'][i]
            chains = batch['chain_ids'][i]
            positions = [0]
            for c in [0, 1]:
                ids = torch.nonzero(residues & chains.eq(c)).flatten()
                positions += ids[torch.linspace(0, len(ids)-1, min(6, len(ids)), device=x.device).round().long()].tolist()
            positions = torch.tensor(positions, device=x.device)
            same = chains[positions, None].eq(chains[None, :])
            cross = ~same & valid[None, :]
            rot = (qr[i, :, positions].float() @ kr[i].float().transpose(-1,-2)) * scale
            raw = (q[i, :, positions].float() @ k[i].float().transpose(-1,-2)) * scale
            for score_mode in ['standard', 'chain_aware']:
                scores = rot if score_mode == 'standard' else torch.where(same[None], rot, raw)
                p = scores.masked_fill(~valid[None, None, :], -torch.inf).softmax(-1)
                cross_mass = (p * cross[None]).sum(-1)
                special_mass = (p * (valid & ~residues)[None, None, :]).sum(-1)
                entropy = -(p * p.clamp_min(1e-30).log()).sum(-1)
                # Residue queries only; CLS is reported separately.
                rec = {'pair': context['pair'], 'orientation': i, 'layer': layer_index,
                       'activation_mode': context['mode'], 'score_mode': score_mode,
                       'residue_cross_mass_mean': float(cross_mass[:, 1:].mean()),
                       'residue_cross_mass_gt_90pct_fraction': float((cross_mass[:, 1:] > .9).float().mean()),
                       'cls_cross_mass_mean': float(cross_mass[:, 0].mean()),
                       'special_key_mass_mean': float(special_mass[:, 1:].mean()),
                       'normalized_entropy_mean': float(entropy[:, 1:].mean()/math.log(int(valid.sum()))),
                       'top1_mass_mean': float(p[:, 1:].max(-1).values.mean()),
                       'cross_score_mean': float(scores[:, 1:][cross[None, 1:].expand(scores.shape[0], -1, -1)].mean()),
                       'within_score_mean': float(scores[:, 1:][(same & valid[None]) [None, 1:].expand(scores.shape[0], -1, -1)].mean())}
                records.append(rec)


def representation(hidden, batch):
    pooled = []
    for chain in [0, 1]:
        mask = (batch['chain_ids'].eq(chain) & batch['residue_mask']).unsqueeze(-1)
        pooled.append((hidden.float()*mask).sum(1)/mask.sum(1))
    a, b = pooled
    return torch.cat([hidden[:,0].float(), a+b, (a-b).abs(), a*b], -1)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--backbone', choices=['esm2','esmc'], required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--tag', default='pretrained'); parser.add_argument('--skip-kernels', action='store_true')
    args=parser.parse_args()
    torch.set_num_threads(8); torch.manual_seed(2)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    data=PairData(ROOT/'data/prepared','official','train')
    rng=np.random.default_rng(20261001)
    selected=[]
    for low, high in [(128,384),(385,768),(769,1536),(1537,2048)]:
        for label in [0,1]:
            candidates=np.flatnonzero((data.lengths>=low)&(data.lengths<=high)&(data.rows[:,2]==label))
            selected.append(int(rng.choice(candidates)))
    cfg=json.loads((ROOT/'configs'/(args.backbone+'-chain-aware-official-seed2.json')).read_text())
    model=PairModel(ROOT/'assets'/args.backbone,cfg).cuda().eval()
    checkpoint=None
    if args.checkpoint:
        meta=json.loads(args.checkpoint.with_suffix('.pt.json').read_text())
        assert sha256(args.checkpoint)==meta['sha256']
        payload=torch.load(args.checkpoint,map_location='cpu',mmap=True,weights_only=False)
        model.load_state_dict(payload['model'],strict=True)
        checkpoint={'path':str(args.checkpoint),'sha256':meta['sha256'],'training_state':payload['training_state']}
        del payload
    layers=attention_layers(model,args.backbone)
    context={}; traces=[]; captures={}; handles=[]
    def hook(index):
        def observe(layer, inputs):
            if context['mode'] == 'all_one_chain':
                return
            if context.get('capture') and index in [0,len(layers)//2,len(layers)-1]:
                captures[index]=inputs[0].detach().float().clone()
            probe_attention(layer,inputs[0],context['batch'],args.backbone,index,traces,context)
        return observe
    for i, layer in enumerate(layers): handles.append(layer.register_forward_pre_hook(hook(i)))
    features={}; logits={}; comparisons=[]; last_batch=None
    for number,index in enumerate(selected):
        batch=data.batch(np.array([index]),np.zeros(1,dtype=int),2,False,'cuda')
        modes=['standard','chain_aware'] + (['all_one_chain'] if number==0 else [])
        hiddens={}
        for mode in modes:
            for layer in layers:layer.pair_attention='standard' if mode=='standard' else 'chain_aware'
            context.update(pair=index,mode=mode,batch=batch,capture=number==0 and mode=='standard')
            actual_chains=batch['chain_ids'] if mode!='all_one_chain' else torch.where(batch['attention_mask'].bool(),0,-1)
            if mode=='all_one_chain':
                # Probe hook metadata is unused by the tested forward; keep traces labeled.
                context['batch']={**batch,'chain_ids':actual_chains}
            with torch.no_grad(), torch.autocast('cuda',dtype=torch.bfloat16):
                hidden=model.encode(batch['clean_ids'],batch['attention_mask'],actual_chains)
                z=model.classify(hidden,batch['chain_ids'],batch['residue_mask'])
            features.setdefault(mode,[]).append(representation(hidden,batch).cpu())
            logits.setdefault(mode,[]).append(z.float().cpu())
            hiddens[mode]=hidden.float()
            print(json.dumps({'backbone':args.backbone,'tag':args.tag,'pair':index,'tokens':int(data.lengths[index]),'mode':mode,'logits':z.float().cpu().tolist()}),flush=True)
        base=hiddens['standard']; mask=batch['attention_mask'].bool()
        comparisons.append({'pair':index,'tokens':int(data.lengths[index]),
            'chain_vs_standard_hidden_relative_l2':float((hiddens['chain_aware'][mask]-base[mask]).norm()/base[mask].norm()),
            'chain_vs_standard_hidden_cosine_mean':float(F.cosine_similarity(hiddens['chain_aware'][mask],base[mask],dim=-1).mean()),
            **({'all_one_chain_vs_standard_relative_l2':float((hiddens['all_one_chain'][mask]-base[mask]).norm()/base[mask].norm())} if 'all_one_chain' in hiddens else {})})
        if number==0:last_batch=batch
        del hiddens, hidden, z
    for handle in handles:handle.remove()
    feature_summary={}
    for mode, values in features.items():
        f=torch.cat(values).float(); z=torch.cat(logits[mode]).float()
        norm=F.normalize(f,dim=-1); sim=norm@norm.T
        off=~torch.eye(len(sim),dtype=torch.bool)
        feature_summary[mode]={'feature_rms':float(f.square().mean().sqrt()),
            'between_input_centered_rms':float((f-f.mean(0)).square().mean().sqrt()),
            'between_input_centered_relative_rms':float((f-f.mean(0)).square().mean().sqrt()/f.square().mean().sqrt()),
            'mean_pairwise_cosine':float(sim[off].mean()),'logit_std':float(z.std()),'logits':z.tolist()}
    save(OUT/f'{args.backbone}-{args.tag}-attention-traces.json',traces)
    kernels=[] if args.skip_kernels else kernel_checks(layers,captures,last_batch,args.backbone)
    result={'backbone':args.backbone,'tag':args.tag,'checkpoint':checkpoint,
        'training_pair_indices':selected,'training_rows':data.rows[selected].tolist(),'tokens':data.lengths[selected].tolist(),
        'no_optimizer_steps':True,'test_data_used':False,'representations':comparisons,'feature_summary':feature_summary,
        'kernel_checks':kernels,'code':core_hashes(CODE),'script_sha256':sha256(Path(__file__)),
        'trace_file':f'{args.backbone}-{args.tag}-attention-traces.json','torch':torch.__version__,
        'completed_at_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()}
    save(OUT/f'{args.backbone}-{args.tag}-diagnosis.json',result)
    print(json.dumps({'finished':True,'backbone':args.backbone,'tag':args.tag,'feature_summary':feature_summary,'representations':comparisons}),flush=True)


if __name__=='__main__':main()
