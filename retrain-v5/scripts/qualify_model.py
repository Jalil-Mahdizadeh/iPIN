"""Bounded native-head, decoder-gradient, masking and accumulation qualification."""
import argparse
import copy
import json
import types
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from data import PairData
from model import PairModel
from contracts import core_hashes,sha256
from state import atomic_json

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--backbone',choices=['esm2','esmc'],required=True)
    args=parser.parse_args();backbone=args.backbone
    cfg=json.loads((ROOT/'configs'/f'{backbone}-native-ilp-seed2.json').read_text())
    cfg['attention_backend']='math'
    torch.set_num_threads(8);torch.manual_seed(2);torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    train=PairData(ROOT/'data/prepared',backbone,'train')
    ids=np.array([np.flatnonzero((train.rows[:,2]==label)&(train.lengths<384))[-1] for label in [0,1]])
    cycles=np.array([0,1]);batch=train.batch(ids,cycles,2,True,'cuda')
    again=train.batch(ids,cycles,2,True,'cuda')
    assert all(torch.equal(batch[k],again[k]) for k in batch)
    selected=batch['mlm_labels'].ne(-100)
    assert selected.sum().item()==train.masked_token_count(ids,cycles,2)
    assert not selected[~batch['residue_mask']].any()
    assert torch.equal(batch['mlm_labels'][selected],batch['clean_ids'][selected])
    assert torch.equal(batch['clean_ids'][~batch['residue_mask']],batch['masked_ids'][~batch['residue_mask']])
    assert (batch['attention_mask'].sum(1).cpu().numpy()==np.repeat(train.lengths[ids],2)).all()
    assert (batch['clean_ids'][:,0]==0).all()
    # Exchanging the orientation permutes the exact same residue corruption.
    for pair_index,idx in enumerate(ids):
        a,b=map(int,train.rows[idx,:2]);la=len(train.sequence(a));lb=len(train.sequence(b));ab=2*pair_index;ba=ab+1
        for key in ['clean_ids','masked_ids','mlm_labels']:
            assert torch.equal(batch[key][ab,1:1+la],batch[key][ba,lb+2:lb+2+la])
            assert torch.equal(batch[key][ab,la+2:la+2+lb],batch[key][ba,1:1+lb])
    model=PairModel(None,cfg,tiny=True).cuda().train()
    assert list(model.classifier.weight.shape)==[1,64]
    assert not any('readout_' in name for name,_ in model.named_parameters())
    classifier=copy.deepcopy(model.classifier)
    if backbone=='esm2':
        from transformers.models.esm.modeling_esm import EsmSelfAttention
        reference=copy.deepcopy(model.esm_mask);reference.gradient_checkpointing_disable()
        for layer in reference.esm.encoder.layer:
            layer.attention.self.forward=types.MethodType(EsmSelfAttention.forward,layer.attention.self)
        with sdpa_kernel(SDPBackend.MATH):
            refout=reference(input_ids=batch['masked_ids'],attention_mask=batch['attention_mask'],labels=batch['mlm_labels'])
            hidden=reference.esm(input_ids=batch['masked_ids'],attention_mask=batch['attention_mask'],return_dict=True).last_hidden_state
        refmlm=refout.loss;encoder=model.esm_mask
    else:
        from esm.layers.attention import MultiHeadAttention
        reference=copy.deepcopy(model.esmc)
        for block in reference.transformer.blocks:
            block.attn.forward=types.MethodType(MultiHeadAttention.forward,block.attn)
        with sdpa_kernel(SDPBackend.MATH):
            refout=reference(batch['masked_ids'],sequence_id=batch['attention_mask'].bool())
        hidden=refout.embeddings
        refmlm=F.cross_entropy(refout.sequence_logits.reshape(-1,64),batch['mlm_labels'].reshape(-1),ignore_index=-100)
        encoder=model.esmc
    reflogits=classifier(F.relu(hidden[:,0])).view(-1,2)
    refcls=F.binary_cross_entropy_with_logits(reflogits,batch['labels'][:,None].expand_as(reflogits))
    ref_loss=10*refcls+refmlm
    actual,cls,mlm,logits=model(**batch)
    torch.testing.assert_close(actual,ref_loss,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(logits,reflogits,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(cls/len(ids),refcls,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(mlm/selected.sum(),refmlm,atol=2e-5,rtol=2e-5)
    actual.backward();ref_loss.backward()
    gradient_error=0.
    for (name,p),(other,q) in zip(encoder.named_parameters(),reference.named_parameters()):
        assert name==other
        if not p.requires_grad:continue
        assert p.grad is not None and q.grad is not None,name
        assert torch.isfinite(p.grad).all(),name
        torch.testing.assert_close(p.grad,q.grad,atol=1e-4,rtol=7e-4,msg=name)
        gradient_error=max(gradient_error,float((p.grad-q.grad).abs().max()))
    for p,q in zip(model.classifier.parameters(),classifier.parameters()):
        torch.testing.assert_close(p.grad,q.grad,atol=1e-4,rtol=7e-4)
    decoder=model.esm_mask.lm_head if backbone=='esm2' else model.esmc.sequence_head
    assert all(p.requires_grad and p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters())
    assert sum(float(p.grad.abs().sum()) for p in decoder.parameters())>0
    # Both loss normalizers belong to the physical update, regardless of its microbatches.
    reference_grad={n:p.grad.detach().clone() for n,p in model.named_parameters() if p.requires_grad}
    model.zero_grad(set_to_none=True);accumulated=0.;total_tokens=train.masked_token_count(ids,cycles,2)
    for i in range(len(ids)):
        micro=train.batch(ids[i:i+1],cycles[i:i+1],2,True,'cuda')
        loss,*_=model(**micro,pair_normalizer=len(ids),mlm_normalizer=total_tokens)
        accumulated+=float(loss.detach());loss.backward()
    assert abs(accumulated-float(actual.detach()))<5e-5
    for name,p in model.named_parameters():
        if p.requires_grad:torch.testing.assert_close(p.grad,reference_grad[name],atol=2e-4,rtol=1e-3,msg=name)
    model.eval()
    with torch.no_grad():
        clean=model(**batch,compute_loss=False)
        other={**batch,'masked_ids':torch.zeros_like(batch['masked_ids'])}
        assert torch.equal(clean,model(**other,compute_loss=False))
    report=dict(passed=True,backbone=backbone,native_cls_relu_linear_head=True,readout_extensions_absent=True,
                original_backbone_forward_and_all_trainable_gradients_match=True,
                max_encoder_gradient_absolute_error=gradient_error,mlm_decoder_active=True,
                masking_stateless_and_special_tokens_excluded=True,shared_AB_BA_corruption_verified=True,
                global_masked_token_count_correct=True,loss_and_gradients_independent_of_microbatch_partition=True,
                inference_uses_clean_inputs=True,test_data_used=False,production_weights_updated=False,
                code=core_hashes(ROOT/'scripts'),source_sha256={'scripts/qualify_model.py':sha256(Path(__file__))})
    atomic_json(ROOT/'qualification'/f'model-{backbone}.json',report)
    print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
