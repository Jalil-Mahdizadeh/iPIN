"""Original ESM C blocks, final-only embeddings and joint positional attention.

The metadata in CHAIN_CONTEXT never enters the SDK sequence_id mask. There is
one joint attention normalization over both proteins, with key-only padding.
"""
import types
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.utils.checkpoint import checkpoint
from esm.models.esmc import ESMC
from esm.tokenization import EsmSequenceTokenizer
from model_esm2 import (PairModel as ESM2PairModel, CHAIN_CONTEXT, chain_context,
                        checkpoint_contexts, expand_chain_qk)


def esmc_attention_forward(self, x, valid_tokens):
    assert valid_tokens is not None and valid_tokens.dtype == torch.bool
    assert valid_tokens.shape == x.shape[:2]
    q, k, v = self.layernorm_qkv(x).chunk(3, dim=-1)
    q, k = self.q_ln(q).to(q.dtype), self.k_ln(k).to(k.dtype)
    qr, kr = self._apply_rotary(q, k)

    def heads(value):
        return value.unflatten(-1, (self.n_heads, self.d_head)).transpose(1, 2)

    if self.pair_attention == 'chain_aware':
        chains = CHAIN_CONTEXT.get()
        assert chains is not None and chains.shape == x.shape[:2]
        q, k = expand_chain_qk(heads(q), heads(k), heads(qr), heads(kr), chains)
    else:
        assert self.pair_attention == 'standard'
        q, k = heads(qr), heads(kr)
    v = heads(v)
    backend = SDPBackend.MATH if self.pair_backend == 'math' else SDPBackend.EFFICIENT_ATTENTION
    with sdpa_kernel(backend):
        context = F.scaled_dot_product_attention(
            q.to(v.dtype), k.to(v.dtype), v,
            attn_mask=valid_tokens[:, None, None, :],
            dropout_p=0., is_causal=False, scale=self.d_head ** -.5)
    return self.out_proj(context.transpose(1, 2).contiguous().flatten(-2))


class ESMCPairModel(ESM2PairModel):
    def __init__(self, base, cfg, tiny=False):
        # Reuse the v3 classify/loss methods without constructing an ESM2 encoder.
        torch.nn.Module.__init__(self)
        assert cfg['readout'] == 'residue_mean'
        assert not cfg['classification_corruption'] and cfg['mlm_weight'] == 0
        self.cfg = cfg
        self.readout = cfg['readout']
        dim, heads, layers = (64, 4, 2) if tiny else (1152, 18, 36)
        self.esmc = ESMC(dim, heads, layers, EsmSequenceTokenizer(), use_flash_attn=False)
        if not tiny:
            weights = Path(base) / 'esmc_600m_2024_12_v0.pth'
            state = torch.load(weights, map_location='cpu', weights_only=True, mmap=True)
            self.esmc.load_state_dict(state, strict=True)
            del state
        self.classifier = torch.nn.Linear(dim, 1)
        self.readout_projection = torch.nn.Linear(4 * dim, cfg['readout_width'], bias=False)
        self.readout_output = torch.nn.Linear(cfg['readout_width'], 1)
        torch.nn.init.zeros_(self.readout_output.weight)
        torch.nn.init.zeros_(self.readout_output.bias)
        self.esmc.sequence_head.requires_grad_(False)
        self.gradient_checkpointing = True
        for block in self.esmc.transformer.blocks:
            assert not block.use_geom_attn and block.use_plain_attn
            attention = block.attn
            attention.pair_attention = cfg['attention_mode']
            attention.pair_backend = cfg['attention_backend']
            attention.forward = types.MethodType(esmc_attention_forward, attention)

    def encode(self, ids, mask, chains):
        valid = mask.bool()
        with chain_context(chains):
            x = self.esmc.embed(ids)
            for block in self.esmc.transformer.blocks:
                if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
                    # Explicit block/mask arguments and captured chain context remain
                    # valid if other pairs are encoded before this graph's backward.
                    x = checkpoint(block, x, valid, None, None, None,
                                   use_reentrant=False, context_fn=checkpoint_contexts)
                else:
                    x = block(x, valid, None, None, None)
            return self.esmc.transformer.norm(x)
