"""Adapter to the unmodified, pinned Bernett TUnA implementation.

In eval mode the upstream block-diagonal masks and first-column pooling make
its AB/BA max feature factorizable into independent endpoint representations.
This is a property of this exact implementation, not a claim about every TUnA.
Qualification must establish score agreement before cached inference is used.
"""
from __future__ import annotations
import os
from pathlib import Path
import sys
import types
import numpy as np
import torch
from torch.nn import functional as F
from common import ORIGINAL_SHA, sha

UPSTREAM=Path(os.environ.get('TUNA_UPSTREAM_DIR',str(Path(__file__).resolve().parents[1]/'upstream/TUnA/results/bernett/TUnA')))
sys.path.insert(0,str(UPSTREAM))
import model as native
from uncertaintyAwareDeepLearn import VanillaRFFLayer

def create(seed=47,checkpoint=None,device='cuda',fresh_initialize=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    device=torch.device(device)
    intra=native.IntraEncoder(640,64,1,8,256,.2,'swish',device)
    inter=native.InterEncoder(640,64,1,8,256,.2,'swish',device)
    gp=VanillaRFFLayer(64,4096,1,gp_cov_momentum=-1,gp_ridge_penalty=1,likelihood='binary_logistic',random_seed=seed)
    model=native.ProteinInteractionNet(intra,inter,gp,device).to(device)
    if checkpoint is not None:
        model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True),strict=True)
    if fresh_initialize:
        for parameter in model.parameters():
            if parameter.ndim>1:
                torch.nn.init.xavier_uniform_(parameter)
    return model

def load_original(path,device='cuda'):
    if sha(path)!=ORIGINAL_SHA:
        raise RuntimeError('Authors checkpoint checksum mismatch')
    model=create(checkpoint=path,device=device)
    model.eval()
    return model

def endpoint_features(model,x,lengths):
    """Full-length native eval representation, mathematically factorizable."""
    if model.training:
        raise RuntimeError('Per-protein inference caching requires frozen eval mode')
    mask=model.make_masks(lengths,x.shape[1])
    z=model.intra_encoder(x,mask)
    for layer in model.inter_encoder.layer:
        z=layer(z,mask)
    keep=mask[:,0,:,0]
    return (z*keep[:,:,None]).sum(1)/keep.sum(1,keepdim=True)

def rff(model,pair_features):
    gp=model.gp_layer
    v=gp.amplitude*pair_features@gp.weight_mat
    return gp.feature_scale*torch.cat([torch.cos(v),torch.sin(v)],dim=1)

def cached_scores(model,z,a,b,probabilities=False,batch_size=2048):
    result=np.empty(len(a),np.float32)
    with torch.inference_mode():
        for begin in range(0,len(a),batch_size):
            end=min(begin+batch_size,len(a))
            f=torch.maximum(z[torch.as_tensor(a[begin:end],device=z.device)],z[torch.as_tensor(b[begin:end],device=z.device)])
            logit,var=model.gp_layer(f,update_precision=False,get_var=True)
            adjusted=logit.reshape(-1)/torch.sqrt(1+(np.pi/8)*var)
            if probabilities:
                adjusted=torch.sigmoid(adjusted)
            result[begin:end]=adjusted.cpu().numpy()
    if not np.isfinite(result).all():
        raise RuntimeError('Nonfinite TUnA score')
    return result

def native_scores(model,prot_a,prot_b):
    # The released test loader uses batch size ONE. Its mean_field_average
    # broadcasts [B,1] logits against [B] variances for B>1; retain its actual
    # released batch-one predictor as the oracle, not that unintended matrix.
    result=[]
    with torch.inference_mode():
        for a,b in zip(prot_a,prot_b,strict=True):
            packed=native.test_pack([a],[b],[0],512,640,model.device)
            logits,var=model.forward(packed[0],packed[1],packed[3],packed[4],packed[5],packed[6],True,False)
            result.append(model.mean_field_average(logits,var).reshape(()))
        return torch.stack(result)

def sdpa_attention(self,query,key,value,mask=None):
    batch=query.shape[0]
    q=self.w_q(query).view(batch,-1,self.n_heads,self.hid_dim//self.n_heads).transpose(1,2)
    k=self.w_k(key).view(batch,-1,self.n_heads,self.hid_dim//self.n_heads).transpose(1,2)
    v=self.w_v(value).view(batch,-1,self.n_heads,self.hid_dim//self.n_heads).transpose(1,2)
    bias=None if mask is None else torch.where(mask!=0,0.,-1e10).to(q.dtype)
    out=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=self.do.p if self.training else 0.)
    return self.fc(out.transpose(1,2).contiguous().view(batch,-1,self.hid_dim))

def enable_sdpa(model):
    for module in model.modules():
        if isinstance(module,native.SelfAttention):
            module.forward=types.MethodType(sdpa_attention,module)
