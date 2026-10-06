"""Original ESM2 architecture, joint standard attention, native PPI/MLM heads."""
import types
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForMaskedLM, EsmConfig
from native_head import NativePairModel


def sdpa_forward(self, hidden_states, attention_mask=None, head_mask=None,
                 encoder_hidden_states=None, encoder_attention_mask=None,
                 past_key_value=None, output_attentions=False):
    assert head_mask is None and encoder_hidden_states is None and past_key_value is None
    assert not self.is_decoder and not output_attentions and self.position_embedding_type == 'rotary'
    q = self.transpose_for_scores(self.query(hidden_states))*self.attention_head_size**-.5
    k = self.transpose_for_scores(self.key(hidden_states))
    v = self.transpose_for_scores(self.value(hidden_states))
    q, k = self.rotary_embeddings(q, k)
    mask = None if attention_mask is None else attention_mask.eq(0)
    backend = SDPBackend.MATH if self.pair_backend == 'math' else SDPBackend.EFFICIENT_ATTENTION
    with sdpa_kernel(backend):
        x = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v,
                    attn_mask=mask, dropout_p=0., is_causal=False, scale=1.)
    x = x.permute(0, 2, 1, 3).contiguous()
    return (x.view(*x.shape[:-2], self.all_head_size),)


class PairModel(NativePairModel):
    def __init__(self, base, cfg, tiny=False):
        super().__init__(); self.cfg = cfg
        assert cfg['readout'] == 'cls_linear' and cfg['attention_mode'] == 'standard'
        assert cfg['mlm_weight'] == 1. and cfg['classification_corruption'] and cfg['positive_weight'] == 1.
        if tiny:
            c = EsmConfig(vocab_size=33, hidden_size=64, num_hidden_layers=2,
                num_attention_heads=4, intermediate_size=128, pad_token_id=1, mask_token_id=32,
                position_embedding_type='rotary', hidden_dropout_prob=0.,
                attention_probs_dropout_prob=0., token_dropout=True)
            self.esm_mask = AutoModelForMaskedLM.from_config(c)
        else:
            self.esm_mask, info = AutoModelForMaskedLM.from_pretrained(base, local_files_only=True,
                torch_dtype=torch.float32, output_loading_info=True)
            assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys'], info
        self.classifier = torch.nn.Linear(self.esm_mask.config.hidden_size, 1)
        self.esm_mask.esm.contact_head.requires_grad_(False)
        self.esm_mask.esm.embeddings.position_embeddings.requires_grad_(False)
        # All decoder parameters, including tied word embeddings, stay trainable.
        assert all(p.requires_grad for p in self.esm_mask.lm_head.parameters())
        for layer in self.esm_mask.esm.encoder.layer:
            layer.attention.self.pair_backend = cfg['attention_backend']
            layer.attention.self.forward = types.MethodType(sdpa_forward, layer.attention.self)
        self.esm_mask.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})

    def encode(self, ids, mask):
        return self.esm_mask.esm(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state

    def decode(self, selected_hidden):
        return self.esm_mask.lm_head(selected_hidden)
