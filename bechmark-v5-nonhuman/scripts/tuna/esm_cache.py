#!/usr/bin/env python3
"""Frozen ESM-2 last-layer residue cache; no PPI labels or test pairs used.

Load the existing pinned safetensors into fair-esm, qualifying the conversion
against Transformers and the optional SDPA path against native fair-esm.
No truncation, windowing, reduced precision, or contact-head inference.
"""
from __future__ import annotations
import argparse
import re
import time
import types
from pathlib import Path
import h5py
import numpy as np
import torch
from torch.nn import functional as F
import esm
from safetensors.torch import load_file
from common import cuda, now, read, record, sha, write

ENCODER_SHA='c3f1da8aea53bddd32c246c86168c23b9fd72341fb9db9a94436f855f5053566'

def load_encoder(directory,device):
    directory=Path(directory)
    path=directory/'model.safetensors'
    if sha(path)!=ENCODER_SHA:
        raise RuntimeError('Frozen ESM checkpoint checksum mismatch')
    alphabet=esm.Alphabet.from_architecture('ESM-1b')
    if alphabet.all_toks!=(directory/'vocab.txt').read_text().splitlines():
        raise RuntimeError('ESM alphabet mismatch')
    model=esm.ESM2(num_layers=30,embed_dim=640,attention_heads=20,alphabet=alphabet,token_dropout=True)
    state=load_file(path)
    converted={}; ignored=[]
    replacements={
        'attention.self.query.':'self_attn.q_proj.',
        'attention.self.key.':'self_attn.k_proj.',
        'attention.self.value.':'self_attn.v_proj.',
        'attention.self.rotary_embeddings.':'self_attn.rot_emb.',
        'attention.output.dense.':'self_attn.out_proj.',
        'attention.LayerNorm.':'self_attn_layer_norm.',
        'intermediate.dense.':'fc1.',
        'output.dense.':'fc2.',
        'LayerNorm.':'final_layer_norm.',
    }
    for name,value in state.items():
        if name in ('esm.embeddings.position_embeddings.weight','esm.embeddings.position_ids'):
            ignored.append(name)
            continue
        target=name
        if name=='esm.embeddings.word_embeddings.weight':
            target='embed_tokens.weight'
        elif name.startswith('esm.encoder.emb_layer_norm_after.'):
            target=name.replace('esm.encoder.','',1)
        elif name.startswith('esm.contact_head.'):
            target=name[4:]
        elif name=='lm_head.decoder.weight':
            target='lm_head.weight'
        elif name.startswith('esm.encoder.layer.'):
            match=re.fullmatch(r'esm\.encoder\.layer\.(\d+)\.(.+)',name)
            layer,suffix=match.groups()
            for old,new in replacements.items():
                if suffix.startswith(old):
                    target=f'layers.{layer}.'+suffix.replace(old,new,1)
                    break
        if target in converted:
            raise RuntimeError('Duplicate converted weight')
        converted[target]=value
    if 'lm_head.weight' not in converted:
        converted['lm_head.weight']=converted['embed_tokens.weight']
    model.load_state_dict(converted,strict=True)
    return model.to(device).eval(),alphabet,ignored

def fast_attention(self,query,key,value,key_padding_mask=None,incremental_state=None,
                   need_weights=True,static_kv=False,attn_mask=None,before_softmax=False,
                   need_head_weights=False):
    if incremental_state is not None or static_kv or attn_mask is not None or before_softmax or need_head_weights:
        return self._tuna_native_forward(query,key,value,key_padding_mask,incremental_state,
            need_weights,static_kv,attn_mask,before_softmax,need_head_weights)
    length,batch,dim=query.shape
    q=(self.q_proj(query)*self.scaling).reshape(length,batch*self.num_heads,self.head_dim).transpose(0,1)
    k=self.k_proj(key).reshape(length,batch*self.num_heads,self.head_dim).transpose(0,1)
    v=self.v_proj(value).reshape(length,batch*self.num_heads,self.head_dim).transpose(0,1)
    if self.rot_emb is not None:
        q,k=self.rot_emb(q,k)
    q=q.reshape(batch,self.num_heads,length,self.head_dim)
    k=k.reshape_as(q); v=v.reshape_as(q)
    mask=None if key_padding_mask is None else (~key_padding_mask)[:,None,None,:]
    out=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,dropout_p=0.,scale=1.)
    out=out.permute(2,0,1,3).reshape(length,batch,dim)
    return self.out_proj(out),None

def enable_fast(model):
    for layer in model.layers:
        module=layer.self_attn
        module._tuna_native_forward=module.forward
        module.forward=types.MethodType(fast_attention,module)

def embed(model,alphabet,seqs):
    _,_,tokens=alphabet.get_batch_converter()([(str(i),s) for i,s in enumerate(seqs)])
    tokens=tokens.to(next(model.parameters()).device)
    with torch.inference_mode():
        rep=model(tokens,repr_layers=[30],return_contacts=False)['representations'][30]
    return [rep[i,1:len(s)+1].float() for i,s in enumerate(seqs)]

def qualify(model,alphabet,directory,sequences,output):
    from transformers import AutoTokenizer, EsmModel
    # Fixed TRAIN-only fixtures spanning padding and lengths; no outcome tuning.
    fixtures=[sequences[i] for i in range(min(3,len(sequences)))]
    original=embed(model,alphabet,fixtures)
    hf=EsmModel.from_pretrained(directory,local_files_only=True,add_pooling_layer=False,
        attn_implementation='eager').to(next(model.parameters()).device).eval()
    tokenizer=AutoTokenizer.from_pretrained(directory,local_files_only=True)
    inputs=tokenizer(fixtures,return_tensors='pt',padding=True).to(next(model.parameters()).device)
    with torch.inference_mode():
        href=hf(**inputs).last_hidden_state
    hf_error=max(float((x-href[i,1:len(s)+1]).abs().max()) for i,(x,s) in enumerate(zip(original,fixtures)))
    del hf,href
    enable_fast(model)
    accelerated=embed(model,alphabet,fixtures)
    fast_error=max(float((a-b).abs().max()) for a,b in zip(original,accelerated))
    # A length not used in the conversion fixture checks rotary/padding behavior.
    separate=embed(model,alphabet,[fixtures[0]])[0]
    padding_error=float((separate-accelerated[0]).abs().max())
    passed=hf_error<=2e-4 and fast_error<=2e-4 and padding_error<=2e-4
    result={'at_utc':now(),'passed':passed,'encoder_sha256':ENCODER_SHA,
        'transformers_vs_native_max_absolute_error':hf_error,
        'sdpa_vs_native_max_absolute_error':fast_error,'padding_max_absolute_error':padding_error,
        'absolute_tolerance':2e-4,'fixture_lengths':[len(s) for s in fixtures],
        'precision':'FP32; TF32 disabled; no autocast','test_pairs_or_truth_read':False}
    write(output,result)
    if not passed:
        raise RuntimeError(f'ESM qualification failed: {result}')
    return result

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--encoder',type=Path,required=True)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--qualify-only',action='store_true')
    args=p.parse_args()
    device=cuda(); started=time.monotonic()
    seq=read(args.data/'sequences.json')
    model,alphabet,ignored=load_encoder(args.encoder,device)
    args.output.mkdir(parents=True,exist_ok=True)
    train=[s for s,part in zip(seq['sequence'],seq['partition']) if part=='train']
    fixtures=sorted(train,key=len)
    fixture=[fixtures[len(fixtures)//4],fixtures[len(fixtures)//2],fixtures[int(.9*len(fixtures))]]
    qualification=qualify(model,alphabet,args.encoder,fixture,args.output/'ESM_QUALIFICATION.json')
    print(qualification,flush=True)
    if args.qualify_only:
        return
    target=args.output/'residues.h5'
    with h5py.File(target,'a') as cache:
        identity=sha(args.data/'sequences.json')
        if 'sequence_manifest_sha256' in cache.attrs and cache.attrs['sequence_manifest_sha256']!=identity:
            raise RuntimeError('Cache sequence-manifest mismatch')
        cache.attrs['sequence_manifest_sha256']=identity
        cache.attrs['encoder_sha256']=ENCODER_SHA
        cache.attrs['extraction']='full-context layer30 FP32 residues, no BOS/EOS, no truncation'
        order=sorted(range(len(seq['sequence'])),key=lambda i:len(seq['sequence'][i]))
        done=0; residue_count=0
        for start in range(0,len(order),4):
            indices=[i for i in order[start:start+4] if str(i) not in cache]
            if not indices:
                continue
            values=embed(model,alphabet,[seq['sequence'][i] for i in indices])
            for i,value in zip(indices,values,strict=True):
                array=value.cpu().numpy()
                if array.shape!=(seq['length'][i],640) or not np.isfinite(array).all():
                    raise RuntimeError('Invalid residue representation')
                cache.create_dataset(str(i),data=array,dtype='float32')
                done+=1; residue_count+=len(array)
            if done%100<4:
                cache.flush()
                print({'new_proteins':done,'new_residues':residue_count,
                    'elapsed_seconds':time.monotonic()-started,'last_length':seq['length'][indices[-1]]},flush=True)
        if len(cache)!=len(seq['sequence']):
            raise RuntimeError('Incomplete residue cache')
        cache.attrs['complete']=True
        cache.flush()
    summary={'completed_at_utc':now(),'elapsed_seconds':time.monotonic()-started,
        'proteins':len(seq['sequence']),'residues':sum(seq['length']),'maximum_length':max(seq['length']),
        'cache':record(target),'encoder':record(args.encoder/'model.safetensors'),
        'ignored_unused_absolute_position_weights':ignored,'qualification':qualification,
        'test_pairs_read':False,'test_truth_read':False}
    write(args.output/'RESIDUE_CACHE_MANIFEST.json',summary)
    print(summary,flush=True)

if __name__=='__main__':
    main()
