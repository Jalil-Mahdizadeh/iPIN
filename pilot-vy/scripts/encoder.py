"""Frozen monomer trunk; structural and language-model output heads are forbidden."""
import time
from pathlib import Path
import numpy as np
import torch
from study import config
from monomers import PAD, blocks


def pool_query(hidden, good):
    if hidden.ndim != 2 or good.shape != hidden.shape[:1] or not good.any():
        raise ValueError('Invalid query-state coordinates')
    h = hidden.float(); valid = h[good]
    if not torch.isfinite(valid).all():
        raise FloatingPointError('Nonfinite query-residue representation')
    values = [valid.mean(0), valid.std(0, unbiased=False)]
    for indices in blocks(len(h)):
        idx = torch.as_tensor(indices, dtype=torch.long, device=h.device)
        selected = h[idx][good[idx]]
        values.append(selected.mean(0) if len(selected) else torch.zeros(h.shape[1], device=h.device))
    result = torch.cat(values)
    if not torch.isfinite(result).all():
        raise FloatingPointError('Nonfinite pooled representation')
    return result.cpu().numpy().astype(np.float32)


class Encoder:
    def __init__(self):
        from MSA_Pairformer.model import MSAPairformer
        cfg = config()
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('VY requires exactly one allocated visible GPU')
        torch.set_num_threads(cfg['budgets']['torch_threads']); torch.manual_seed(cfg['seed'])
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        self.model = MSAPairformer(); state = {}; folder = Path('/opt/vx/weights')
        for name in ('model_cuex.bin', 'contact.bin', 'confind_contact.bin'):
            state.update(torch.load(folder / name, map_location='cpu', weights_only=True))
        self.model.load_state_dict(state, strict=True)
        if self.model.dim_msa != cfg['encoder']['channels'] or len(self.model.core_stack.layers) != 22:
            raise ValueError('Unexpected frozen encoder architecture')
        self.model.eval().requires_grad_(False).cuda()
        self.final_update_calls = 0; self.core_layer_calls = 0
        def forbidden(*args):
            raise RuntimeError('A forbidden pretrained output head was evaluated')
        for module in [self.model.contact_head, self.model.confind_contact_head, self.model.lm_head]:
            module.register_forward_pre_hook(forbidden)
        def final_hook(*args):
            self.final_update_calls += 1
        self.model.core_stack.final_msa_transition.register_forward_hook(final_hook)
        def layer_hook(*args):
            self.core_layer_calls += 1
        for layer in self.model.core_stack.layers:
            layer[-1].register_forward_hook(layer_hook)

    @torch.inference_mode()
    def one(self, tokens):
        from MSA_Pairformer.dataset import prepare_msa_masks
        if tokens.ndim != 2 or not np.isin(tokens, np.arange(27)).all():
            raise ValueError('Invalid encoded MSA')
        before_final, before_layers = self.final_update_calls, self.core_layer_calls
        torch.cuda.synchronize(); started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
        t = torch.as_tensor(tokens.astype(np.int64), device='cuda').unsqueeze(0)
        masks = prepare_msa_masks(t, device=torch.device('cuda'))
        good = t[0, 0] != PAD
        with torch.autocast('cuda', dtype=torch.bfloat16):
            msa, pair = self.model.init_representations(torch.nn.functional.one_hot(t, 28).to(torch.bfloat16), None)
            result = self.model.core_stack(msa=msa, pairwise_repr=pair, mask=masks[0], msa_mask=masks[1],
                full_mask=masks[2], pairwise_mask=masks[3], query_only=True, return_seq_weights=False,
                return_msa_repr_layer_idx=None, return_pairwise_repr_layer_idx=None,
                return_repr_after_layer_idx=None, store_msa_repr_cpu=False, store_pairwise_repr_cpu=False,
                return_pairwise_repr_only=False)
            hidden = result['final_msa_repr']
            if tuple(hidden.shape) != (1, 1, tokens.shape[1], 464):
                raise ValueError('Unexpected query representation shape')
        feature = pool_query(hidden[0, 0], good)
        torch.cuda.synchronize()
        timing = {'seconds': time.monotonic() - started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                  'rows': len(tokens), 'length': tokens.shape[1], 'final_msa_updates': self.final_update_calls - before_final,
                  'core_layers': self.core_layer_calls - before_layers}
        if timing['final_msa_updates'] != 1 or timing['core_layers'] != 22:
            raise ValueError('Trunk terminated before the declared representation')
        return feature, timing

    def both(self, tokens):
        M, tm = self.one(tokens)
        S, ts = self.one(tokens[:1])
        return M, S, {'M': tm, 'S': ts}
