"""Original encoder with observational guards; only true MSA is encoded."""
import time
import numpy as np
import torch
from shared import config
from encoder import Encoder


class GuardedEncoder(Encoder):
    def __init__(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Exactly one visible GPU is required per worker')
        super().__init__(); self.calls = []; self.layers = []
        def forbidden(*args):
            raise RuntimeError('Pretrained output heads are forbidden')
        for head in [self.model.contact_head, self.model.confind_contact_head, self.model.lm_head]:
            head.register_forward_pre_hook(forbidden)
        for i, layer in enumerate(self.model.core_stack.layers):
            layer[-1].register_forward_hook(lambda m, args, out, i=i: self.layers.append(i))
        self.model.core_stack.register_forward_pre_hook(self.check_call, with_kwargs=True)

    def check_call(self, module, args, kwargs):
        if kwargs.get('return_repr_after_layer_idx') != 15 or kwargs.get('return_pairwise_repr_only') is not True:
            raise ValueError('Frozen layer/readout changed')
        self.calls.append(int(kwargs['msa'].shape[1]))

    def encode(self, tokens, breakpoint):
        cfg = config(); depth = len(tokens)
        if tokens.ndim != 2 or not cfg['minimum_paired_homologs'] + 1 <= depth <= 128:
            raise ValueError('Invalid original MSA depth')
        if not 0 < breakpoint < tokens.shape[1] <= 1536:
            raise ValueError('Invalid chain boundary/length')
        self.calls.clear(); self.layers.clear()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
        value = super().symmetric(tokens, breakpoint)
        torch.cuda.synchronize()
        if self.calls != [depth, depth] or self.layers != list(range(16)) * 2:
            raise ValueError('Actual MSA depth/layer execution changed')
        if value.shape != (128,) or not np.isfinite(value).all():
            raise FloatingPointError('Invalid true feature')
        return value, {'seconds': time.monotonic() - started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
            'orientation_depths': list(self.calls), 'layer_indices': list(self.layers)}
