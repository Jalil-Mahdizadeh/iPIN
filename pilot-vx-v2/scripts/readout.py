"""The original frozen encoder, with both pooling views from each forward."""
import numpy as np
import torch
from study import config
from encoder import Encoder
from pooling import pool_views, spatial_indices


class LocalEncoder(Encoder):
    @torch.inference_mode()
    def one_views(self, tokens, breakpoint, permutation_a, permutation_b):
        from MSA_Pairformer.dataset import prepare_msa_masks
        t = torch.as_tensor(tokens.astype(np.int64), device=self.device).unsqueeze(0)
        masks = prepare_msa_masks(t, device=self.device)
        good = t[0, 0] != 26
        with torch.autocast('cuda', dtype=torch.bfloat16):
            msa, pair = self.model.init_representations(torch.nn.functional.one_hot(t, 28).to(torch.bfloat16), [[breakpoint]])
            result = self.model.core_stack(msa=msa, pairwise_repr=pair, mask=masks[0], msa_mask=masks[1],
                full_mask=masks[2], pairwise_mask=masks[3], return_repr_after_layer_idx=15,
                return_pairwise_repr_only=True, store_pairwise_repr_cpu=False)
            z = result['final_pairwise_repr'][0]
            cross = (z[:breakpoint, breakpoint:] + z[breakpoint:, :breakpoint].transpose(0, 1)) / 2
            projected = cross @ self.projection
        views, meta = pool_views(projected, good[:breakpoint], good[breakpoint:], permutation_a, permutation_b)
        return {k: v.cpu().numpy().astype(np.float64) for k, v in views.items()}, meta

    def symmetric_views(self, tokens, breakpoint, uid, a, b):
        pa = spatial_indices(tokens[0, :breakpoint] != 26, uid, a)
        pb = spatial_indices(tokens[0, breakpoint:] != 26, uid, b)
        first, ma = self.one_views(tokens, breakpoint, pa, pb)
        reverse = np.concatenate([tokens[:, breakpoint:], tokens[:, :breakpoint]], axis=1)
        second, mb = self.one_views(reverse, tokens.shape[1] - breakpoint, pb, pa)
        if ma != mb:
            raise ValueError('Orientation changed block availability')
        return {k: (first[k] + second[k]) / 2 for k in first}, ma
