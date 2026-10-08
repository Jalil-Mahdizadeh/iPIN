"""Frozen trunk-only features; released structural heads are never evaluated."""
import json
import math
from pathlib import Path
import numpy as np
import torch
from common import config


class Encoder:
    def __init__(self, weights=None):
        from MSA_Pairformer.model import MSAPairformer
        torch.set_num_threads(8);torch.manual_seed(config()['seed'])
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        self.device=torch.device('cuda')
        self.model=MSAPairformer()
        dest=Path(weights or '/opt/vx/weights');state={}
        for name in ['model_cuex.bin','contact.bin','confind_contact.bin']:
            state.update(torch.load(dest/name,map_location='cpu',weights_only=True))
        self.model.load_state_dict(state,strict=True)
        self.model.eval().requires_grad_(False).to(self.device)
        assert not any(p.requires_grad for p in self.model.parameters())
        rng=torch.Generator(device='cpu');rng.manual_seed(config()['seed'])
        self.projection=(torch.randn(256,32,generator=rng)/math.sqrt(256)).to(self.device)

    @torch.inference_mode()
    def one(self,tokens,breakpoint):
        from MSA_Pairformer.dataset import prepare_msa_masks
        t=torch.as_tensor(tokens.astype(np.int64),device=self.device).unsqueeze(0)
        masks=prepare_msa_masks(t,device=self.device)
        good=(t[0,0]!=26);a=good[:breakpoint];b=good[breakpoint:]
        if not a.any() or not b.any():raise ValueError('No valid inter-chain positions')
        with torch.autocast('cuda',dtype=torch.bfloat16):
            msa,pair=self.model.init_representations(torch.nn.functional.one_hot(t,28).to(torch.bfloat16),[[breakpoint]])
            result=self.model.core_stack(msa=msa,pairwise_repr=pair,mask=masks[0],msa_mask=masks[1],
                full_mask=masks[2],pairwise_mask=masks[3],return_repr_after_layer_idx=15,
                return_pairwise_repr_only=True,store_pairwise_repr_cpu=False)
            z=result['final_pairwise_repr'][0]
            # Symmetrize the two directed inter-chain blocks within each forward.
            cross=(z[:breakpoint,breakpoint:]+z[breakpoint:,:breakpoint].transpose(0,1))/2
            projected=(cross @ self.projection)[a][:,b].reshape(-1,32).float()
        # Fixed random projection plus four distribution summaries: 128 scalars.
        feat=torch.cat([projected.mean(0),projected.std(0,unbiased=False),
                        projected.amax(0),torch.quantile(projected,.95,dim=0)])
        if not torch.isfinite(feat).all():raise FloatingPointError('Nonfinite trunk features')
        return feat.cpu().numpy().astype(np.float64)

    def symmetric(self,tokens,breakpoint):
        a=self.one(tokens,breakpoint)
        reverse=np.concatenate([tokens[:,breakpoint:],tokens[:,:breakpoint]],axis=1)
        b=self.one(reverse,tokens.shape[1]-breakpoint)
        return (a+b)/2
