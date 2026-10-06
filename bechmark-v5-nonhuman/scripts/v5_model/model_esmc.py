"""ESMC standard attention with the native CLS classifier and active MLM decoder."""
import types
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.utils.checkpoint import checkpoint
from esm.models.esmc import ESMC
from esm.tokenization import EsmSequenceTokenizer
from native_head import NativePairModel


def attention_forward(self, x, valid):
    assert valid is not None and valid.dtype == torch.bool and valid.shape == x.shape[:2]
    q, k, v = self.layernorm_qkv(x).chunk(3, dim=-1)
    q, k = self.q_ln(q).to(q.dtype), self.k_ln(k).to(k.dtype)
    q, k = self._apply_rotary(q, k)
    def heads(z): return z.unflatten(-1, (self.n_heads, self.d_head)).transpose(1, 2)
    q, k, v = map(heads, (q, k, v))
    backend = SDPBackend.MATH if self.pair_backend == 'math' else SDPBackend.EFFICIENT_ATTENTION
    with sdpa_kernel(backend):
        x = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v,
                    attn_mask=valid[:, None, None, :], dropout_p=0., is_causal=False,
                    scale=self.d_head**-.5)
    return self.out_proj(x.transpose(1, 2).contiguous().flatten(-2))


class ESMCPairModel(NativePairModel):
    def __init__(self, base, cfg, tiny=False):
        super().__init__(); self.cfg = cfg
        assert cfg['readout'] == 'cls_linear' and cfg['attention_mode'] == 'standard'
        assert cfg['mlm_weight'] == 1. and cfg['classification_corruption'] and cfg['positive_weight'] == 1.
        dim, heads, layers = (64, 4, 2) if tiny else (1152, 18, 36)
        self.esmc = ESMC(dim, heads, layers, EsmSequenceTokenizer(), use_flash_attn=False)
        if not tiny:
            state = torch.load(Path(base)/'esmc_600m_2024_12_v0.pth', map_location='cpu',
                               weights_only=True, mmap=True)
            self.esmc.load_state_dict(state, strict=True); del state
        self.classifier = torch.nn.Linear(dim, 1)
        self.gradient_checkpointing = True
        assert all(p.requires_grad for p in self.esmc.sequence_head.parameters())
        for block in self.esmc.transformer.blocks:
            assert not block.use_geom_attn and block.use_plain_attn
            block.attn.pair_backend = cfg['attention_backend']
            block.attn.forward = types.MethodType(attention_forward, block.attn)

    def encode(self, ids, mask):
        valid = mask.bool(); x = self.esmc.embed(ids)
        for block in self.esmc.transformer.blocks:
            if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                x = checkpoint(block, x, valid, None, None, None, use_reentrant=False)
            else:
                x = block(x, valid, None, None, None)
        return self.esmc.transformer.norm(x)

    def decode(self, selected_hidden):
        return self.esmc.sequence_head(selected_hidden)
