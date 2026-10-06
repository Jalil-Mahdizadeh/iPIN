"""Controlled objectives/readouts and exact hybrid attention without L-by-L masks."""
import contextlib
import contextvars
import types
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForMaskedLM, EsmConfig

CHAIN_CONTEXT = contextvars.ContextVar('plmi_chain_metadata', default=None)


@contextlib.contextmanager
def chain_context(ids):
    token = CHAIN_CONTEXT.set(ids)
    try: yield
    finally: CHAIN_CONTEXT.reset(token)


def checkpoint_contexts():
    # Capture the tensor for this forward; later forwards may have other chains.
    ids = CHAIN_CONTEXT.get()
    return chain_context(ids), chain_context(ids)


def expand_chain_qk(q, k, qr, kr, chains):
    """Exact hybrid scores: RoPE within chains, unrotated across chains.

    q_A=[qr,q,0,0], k_A=[kr,0,k,0]
    q_B=[0,0,q,qr], k_B=[0,k,0,kr].
    One SDPA call keeps one joint softmax. Q/K have 4*D features; V is unchanged.
    No quadratic chain mask or dense attention matrix is allocated.
    """
    a = chains.eq(0)[:, None, :, None]
    b = chains.eq(1)[:, None, :, None]
    return (torch.cat((qr * a, q * a, q * b, qr * b), dim=-1),
            torch.cat((kr * a, k * b, k * a, kr * b), dim=-1))


def sdpa_forward(self, hidden_states, attention_mask=None, head_mask=None,
                 encoder_hidden_states=None, encoder_attention_mask=None,
                 past_key_value=None, output_attentions=False):
    assert head_mask is None and encoder_hidden_states is None and past_key_value is None
    assert not self.is_decoder and not output_attentions and self.position_embedding_type == 'rotary'
    q = self.transpose_for_scores(self.query(hidden_states)) * self.attention_head_size ** -.5
    k = self.transpose_for_scores(self.key(hidden_states))
    v = self.transpose_for_scores(self.value(hidden_states))
    qr, kr = self.rotary_embeddings(q, k)
    if self.pair_attention == 'chain_aware':
        chains = CHAIN_CONTEXT.get()
        assert chains is not None and chains.shape == hidden_states.shape[:2]
        q, k = expand_chain_qk(q, k, qr, kr, chains)
    else: q, k = qr, kr
    mask = None if attention_mask is None else attention_mask.eq(0)
    backend = SDPBackend.MATH if self.pair_backend == 'math' else SDPBackend.EFFICIENT_ATTENTION
    with sdpa_kernel(backend):
        result = F.scaled_dot_product_attention(q.to(v.dtype), k.to(v.dtype), v,
                    attn_mask=mask, dropout_p=0., is_causal=False, scale=1.)
    result = result.permute(0, 2, 1, 3).contiguous()
    return (result.view(*result.shape[:-2], self.all_head_size),)


class PairModel(torch.nn.Module):
    def __init__(self, base, cfg, tiny=False):
        super().__init__()
        self.cfg = cfg
        if tiny:
            config = EsmConfig(vocab_size=33, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                intermediate_size=128, pad_token_id=1, mask_token_id=32, position_embedding_type='rotary',
                hidden_dropout_prob=0., attention_probs_dropout_prob=0., token_dropout=True)
            self.esm_mask = AutoModelForMaskedLM.from_config(config)
        else:
            self.esm_mask, info = AutoModelForMaskedLM.from_pretrained(base, local_files_only=True,
                torch_dtype=torch.float32, output_loading_info=True)
            assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys'], info
        dim = self.esm_mask.config.hidden_size
        self.classifier = torch.nn.Linear(dim, 1)
        self.readout = cfg['readout']
        if self.readout != 'cls_linear':
            width = cfg.get('readout_width', 128)
            # Exactly equal added parameter counts. CLS uses 4*width nonlinear
            # units, averaged in four blocks; residue head uses four D-vectors.
            inputs, outputs = (dim, 4 * width) if self.readout == 'cls_mlp' else (4 * dim, width)
            self.readout_projection = torch.nn.Linear(inputs, outputs, bias=False)
            self.readout_output = torch.nn.Linear(width, 1)
            torch.nn.init.zeros_(self.readout_output.weight)
            torch.nn.init.zeros_(self.readout_output.bias)
        self.esm_mask.esm.contact_head.requires_grad_(False)
        self.esm_mask.esm.embeddings.position_embeddings.requires_grad_(False)
        if not cfg['mlm_weight']:
            encoder_ids = {id(p) for p in self.esm_mask.esm.parameters()}
            for p in self.esm_mask.lm_head.parameters():
                if id(p) not in encoder_ids: p.requires_grad_(False)
            assert self.esm_mask.esm.embeddings.word_embeddings.weight.requires_grad
        for layer in self.esm_mask.esm.encoder.layer:
            attention = layer.attention.self
            attention.pair_attention = cfg['attention_mode']
            attention.pair_backend = cfg['attention_backend']
            attention.forward = types.MethodType(sdpa_forward, attention)
        self.esm_mask.gradient_checkpointing_enable(gradient_checkpointing_kwargs={
            'use_reentrant': False, 'context_fn': checkpoint_contexts})

    def encode(self, ids, mask, chains):
        with chain_context(chains):
            return self.esm_mask.esm(input_ids=ids, attention_mask=mask, return_dict=True).last_hidden_state

    def classify(self, hidden, chain_ids, residue_mask):
        cls = hidden[:, 0]
        logits = self.classifier(F.relu(cls))
        if self.readout != 'cls_linear':
            if self.readout == 'cls_mlp':
                features = F.gelu(self.readout_projection(cls)).reshape(len(hidden), 4, -1).mean(1)
            else:
                pooled = []
                for chain in [0, 1]:
                    mask = (chain_ids.eq(chain) & residue_mask).unsqueeze(-1)
                    pooled.append((hidden.float() * mask).sum(1) / mask.sum(1).clamp_min(1))
                a, b = pooled
                features = torch.cat((cls.float(), a + b, (a - b).abs(), a * b), dim=-1)
                features = F.gelu(self.readout_projection(features))
            logits = logits + self.readout_output(features)
        return logits.view(-1, 2)

    def forward(self, clean_ids, masked_ids, attention_mask, chain_ids, residue_mask,
                labels=None, mlm_labels=None, compute_loss=True):
        corrupted = self.cfg['classification_corruption'] and compute_loss
        hidden = self.encode(masked_ids if corrupted else clean_ids, attention_mask, chain_ids)
        logits = self.classify(hidden, chain_ids, residue_mask)
        if not compute_loss: return logits
        positive_weight = logits.new_tensor(self.cfg['positive_weight'], dtype=torch.float32)
        cls = F.binary_cross_entropy_with_logits(logits.float(), labels[:, None].expand_as(logits).float(),
                                                  pos_weight=positive_weight, reduction='none').mean(1)
        mlm = cls * 0
        if self.cfg['mlm_weight']:
            masked_hidden = hidden if corrupted else self.encode(masked_ids, attention_mask, chain_ids)
            selected = mlm_labels.ne(-100)
            values = F.cross_entropy(self.esm_mask.lm_head(masked_hidden[selected]).float(),
                                     mlm_labels[selected], reduction='none')
            orientation = selected.nonzero()[:, 0]
            sums = torch.zeros(len(hidden), device=hidden.device, dtype=torch.float32).scatter_add(0, orientation, values)
            mlm = (sums / selected.sum(1).clamp_min(1)).view(-1, 2).mean(1)
        loss = self.cfg['classification_weight'] * cls + self.cfg['mlm_weight'] * mlm
        return loss.sum(), cls.detach().sum(), mlm.detach().sum(), logits.detach()
