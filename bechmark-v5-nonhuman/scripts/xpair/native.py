"""Load unchanged released models and reproduce their documented inference."""
import argparse
import re
import numpy as np
import torch
from io_utils import ROOT, read, verify

def load_model(name, device):
    from xpair.model import XPairModel
    item=read(ROOT/'sources/XPAIR_FREEZE.json')['models'][name]
    path=ROOT/item['path'];verify(path,item)
    # Released NumPy-2 scalar metadata, decoded safely with NumPy 1.26.
    allowed=[argparse.Namespace,np.dtype,
             (np.core.multiarray.scalar,'numpy._core.multiarray.scalar'),
             type(np.dtype('float64')),type(np.dtype('float32'))]
    with torch.serialization.safe_globals(allowed):
        checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    params=checkpoint['hyper_parameters']
    model=XPairModel(params).to(device).eval()
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    for p in model.parameters():p.requires_grad_(False)
    return model, {'checkpoint':item,'hyper_parameters':params,
                   'epoch':int(checkpoint['epoch']),'global_step':int(checkpoint['global_step']),
                   'parameters':sum(p.numel() for p in model.parameters())}

def clear_attention(model):
    for layer in model.cross_transformer_block.cross_transformer_layers:
        layer.attn1=layer.attn2=None
        layer.cross_attention.attn=None

def load_encoder(device):
    from transformers import T5EncoderModel, AutoTokenizer
    source=ROOT/'sources/ankh-large'
    model=T5EncoderModel.from_pretrained(source,local_files_only=True).to(device).eval()
    for p in model.parameters():p.requires_grad_(False)
    tokenizer=AutoTokenizer.from_pretrained(source,local_files_only=True)
    assert next(model.parameters()).dtype==torch.float32
    return model,tokenizer

@torch.inference_mode()
def encode(model,tokenizer,sequence,device):
    sequence=re.sub(r'[UZOBJ]','X',sequence.upper())
    inputs=tokenizer(sequence,return_tensors='pt',add_special_tokens=True,
                     return_special_tokens_mask=True).to(device)
    out=model(input_ids=inputs['input_ids'],attention_mask=inputs['attention_mask'])
    feature=out.last_hidden_state[~inputs['special_tokens_mask'].bool()].cpu()
    assert feature.shape==(len(sequence),1536) and feature.dtype==torch.float32
    assert torch.isfinite(feature).all()
    return feature

@torch.inference_mode()
def projected_forward(model,x1,x2,mask1,mask2):
    # These are the unchanged upstream modules; only the independent initial
    # linear projection is cached per protein. No attention approximation.
    h1,h2=model.cross_transformer_block(x1,x2,(mask1,mask2))
    m1=mask1.float().unsqueeze(-1);m2=mask2.float().unsqueeze(-1)
    p1=(h1*m1).sum(1)/m1.sum(1).clamp(min=1.)
    p2=(h2*m2).sum(1)/m2.sum(1).clamp(min=1.)
    out=model.interaction_head(p1*p2).squeeze(-1)
    clear_attention(model)
    return out
