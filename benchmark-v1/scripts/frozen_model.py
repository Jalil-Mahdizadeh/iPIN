"""ESM-2 pair model; shared MLM/classification computation, optional symmetry."""
import math
import types
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoModelForMaskedLM, EsmConfig

def sdpa_forward(self, hidden_states, attention_mask=None, head_mask=None,
                 encoder_hidden_states=None, encoder_attention_mask=None,
                 past_key_value=None, output_attentions=False):
    # Preserve HF ESM's query scaling BEFORE rotary, then disable SDPA's rescaling.
    assert head_mask is None and encoder_hidden_states is None and past_key_value is None
    assert not self.is_decoder and not output_attentions and self.position_embedding_type=='rotary'
    q=self.transpose_for_scores(self.query(hidden_states)) * self.attention_head_size**-.5
    k=self.transpose_for_scores(self.key(hidden_states));v=self.transpose_for_scores(self.value(hidden_states))
    q,k=self.rotary_embeddings(q,k)
    mask=None if attention_mask is None else attention_mask.eq(0)
    backend=SDPBackend.MATH if self.profile_backend=='math' else SDPBackend.EFFICIENT_ATTENTION
    with sdpa_kernel(backend):
        out=F.scaled_dot_product_attention(q.to(v.dtype),k.to(v.dtype),v,attn_mask=mask,
                                          dropout_p=0.,is_causal=False,scale=1.)
    out=out.permute(0,2,1,3).contiguous()
    return (out.view(*out.shape[:-2],self.all_head_size),)

class PairModel(torch.nn.Module):
    def __init__(self, base, mode='symmetric', backend='efficient', tiny=False):
        super().__init__();self.mode=mode
        if tiny:
            cfg=EsmConfig(vocab_size=33,hidden_size=64,num_hidden_layers=2,num_attention_heads=4,
                          intermediate_size=128,pad_token_id=1,mask_token_id=32,
                          position_embedding_type='rotary',hidden_dropout_prob=0.,attention_probs_dropout_prob=0.,token_dropout=True)
            self.esm_mask=AutoModelForMaskedLM.from_config(cfg)
        else:
            self.esm_mask,info=AutoModelForMaskedLM.from_pretrained(base,local_files_only=True,torch_dtype=torch.float32,output_loading_info=True)
            assert not info['missing_keys'] and not info['unexpected_keys'] and not info['mismatched_keys'],info
        self.classifier=torch.nn.Linear(self.esm_mask.config.hidden_size,1)
        self.esm_mask.esm.contact_head.requires_grad_(False) # unused structure head, not a PPI encoder layer
        # HF also instantiates an absolute-position table, unused by rotary ESM-2.
        self.esm_mask.esm.embeddings.position_embeddings.requires_grad_(False)
        self.backend=backend
        if backend!='eager':
            for layer in self.esm_mask.esm.encoder.layer:
                attn=layer.attention.self;attn.profile_backend=backend
                attn.forward=types.MethodType(sdpa_forward,attn)
        self.esm_mask.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})

    def forward(self, input_ids, attention_mask, labels=None, mlm_labels=None, class_weight=10., mlm_weight=1.):
        hidden=self.esm_mask.esm(input_ids=input_ids,attention_mask=attention_mask,return_dict=True).last_hidden_state
        logits=self.classifier(F.relu(hidden[:,0])).view(-1,2)
        if labels is None:return logits
        if self.mode=='symmetric':
            cls=F.binary_cross_entropy_with_logits(logits.float().mean(1),labels.float(),reduction='none')
        else:
            cls=F.binary_cross_entropy_with_logits(logits.float(),labels[:,None].float().expand_as(logits),reduction='none').mean(1)
        if mlm_weight:
            # Apply the MLM head only at supervised positions; equivalent masked-token loss.
            selected=mlm_labels.ne(-100)
            mlm_logits=self.esm_mask.lm_head(hidden[selected]).float()
            values=F.cross_entropy(mlm_logits,mlm_labels[selected],reduction='none')
            orientation_ids=selected.nonzero()[:,0]
            sums=torch.zeros(hidden.shape[0],device=hidden.device,dtype=torch.float32).scatter_add(0,orientation_ids,values)
            counts=selected.sum(1).clamp_min(1)
            mlm=(sums/counts).view(-1,2).mean(1)
        else:mlm=cls*0
        return (class_weight*cls+mlm_weight*mlm).sum(),cls.detach().sum(),mlm.detach().sum(),logits.detach()
