"""Unchanged Vx encoder plus observational validation hooks; no new readout."""
import time
import numpy as np
import torch
from study import config
from encoder import Encoder


class QueryEncoder(Encoder):
    def __init__(self):
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('VZ requires exactly one visible GPU')
        super().__init__()
        self.calls = []
        self.layers = []
        def forbidden(*args):
            raise RuntimeError('Forbidden pretrained output head executed')
        for head in (self.model.contact_head, self.model.confind_contact_head, self.model.lm_head):
            head.register_forward_pre_hook(forbidden)
        for i, layer in enumerate(self.model.core_stack.layers):
            layer[-1].register_forward_hook(lambda module, args, output, i=i: self.layers.append(i))
        self.model.core_stack.register_forward_pre_hook(self.check_call, with_kwargs=True)

    def check_call(self, module, args, kwargs):
        if kwargs.get('return_repr_after_layer_idx') != 15 or kwargs.get('return_pairwise_repr_only') is not True:
            raise ValueError('Frozen layer/readout call changed')
        self.calls.append(int(kwargs['msa'].shape[1]))

    def query(self, tokens, breakpoint):
        if tokens.ndim != 2 or tokens.shape[0] != 1:
            raise ValueError('Q encoder must receive exactly one row')
        if not 0 < breakpoint < tokens.shape[1] <= config()['encoder']['maximum_combined_length']:
            raise ValueError('Invalid Q chain boundaries/length')
        self.calls.clear(); self.layers.clear()
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
        out = super().symmetric(tokens, breakpoint)
        torch.cuda.synchronize()
        # Vx returns after block 15; no final full-stack MSA update is requested.
        expected_layers = list(range(16)) * 2
        if self.calls != [1, 1] or self.layers != expected_layers:
            raise ValueError(f'Wrong depth or layer execution: {self.calls}, {self.layers}')
        if out.shape != (128,) or not np.isfinite(out).all():
            raise ValueError('Invalid Q summary')
        timing = {'seconds': time.monotonic() - started, 'peak_cuda_bytes': torch.cuda.max_memory_allocated(),
                  'orientation_depths': self.calls.copy(), 'orientation_layers': self.layers.copy(),
                  'length': int(tokens.shape[1]), 'breakpoint': int(breakpoint)}
        return out, timing
