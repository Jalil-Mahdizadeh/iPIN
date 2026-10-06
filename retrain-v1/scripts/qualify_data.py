"""Check epoch coverage, label alignment, masking and loss accumulation."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from data import PairData
from model import PairModel
from state import atomic_json

root=Path(__file__).resolve().parents[1];data=PairData(root/'data/prepared','train')
plans=[]
for epoch in range(5):
    plan=data.plan(epoch,2,64);flat=np.concatenate(plan)
    assert len(flat)==len(data) and np.array_equal(np.sort(flat),np.arange(len(data)))
    assert all(1<=len(x)<=64 for x in plan)
    assert all(np.array_equal(x,y) for x,y in zip(plan,data.plan(epoch,2,64)))
    plans.append(hashlib.sha256(flat.tobytes()).hexdigest())
assert len(set(plans))==5
ids=np.argsort(data.lengths,kind='stable')[:8]
batch=data.batch(ids,0,2,True,'cuda');same=data.batch(ids,0,2,True,'cuda')
assert all(torch.equal(batch[k],same[k]) for k in batch)
assert torch.equal(batch['labels'].cpu(),torch.tensor(data.rows[ids,2],dtype=torch.float32))
for j,idx in enumerate(ids):
    a,b=data.rows[idx,:2];la=len(data.sequence(a));lb=len(data.sequence(b))
    for key in ['input_ids','mlm_labels']:
        ab,ba=batch[key][2*j:2*j+2]
        assert torch.equal(ab[1:la+1],ba[lb+2:lb+la+2])
        assert torch.equal(ab[la+2:la+lb+2],ba[1:lb+1])
    assert int(batch['attention_mask'][2*j].sum())==la+lb+3
    assert (batch['mlm_labels'][2*j,:la+1]!=-100).any()
    assert (batch['mlm_labels'][2*j,la+2:la+lb+2]!=-100).any()
torch.manual_seed(1729);torch.backends.cuda.matmul.allow_tf32=False
model=PairModel(root/'assets/esm2',tiny=True,backend='math').cuda().train()
other=copy.deepcopy(model)
model(**batch)[0].div(len(ids)).backward()
for part in [ids[:3],ids[3:7],ids[7:]]:
    other(**data.batch(part,0,2,True,'cuda'))[0].div(len(ids)).backward()
num=den=0.
for (name,a),(othername,b) in zip(model.named_parameters(),other.named_parameters()):
    assert name==othername
    if a.grad is None:assert b.grad is None;continue
    num+=float((a.grad.double()-b.grad.double()).square().sum())
    den+=float(a.grad.double().square().sum())
relative=(num/den)**.5
assert relative<2e-5,relative
result={'passed':True,'all_five_epochs_cover_each_training_row_once':True,
        'epoch_permutation_hashes':plans,'label_alignment':True,'mask_determinism':True,
        'same_chain_masks_in_both_orientations':True,'full_sequences':True,
        'uneven_microbatch_gradient_relative_l2':relative,
        'gradient_test_scope':'small ESM architecture, FP32 math attention; same loss and batching code'}
atomic_json(root/'qualification/data-invariants.json',result)
print(json.dumps(result,indent=2))
