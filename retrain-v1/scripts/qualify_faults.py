"""Exercise checkpoint failure paths on disposable, small synthetic tensors."""
import json
import random
import shutil
from pathlib import Path
import numpy as np
import torch
from state import Checkpoints,atomic_json,capture_rng
from compare_resume import compare

root=Path(__file__).resolve().parents[1]
out=root/'qualification/checkpoint-fault-fixture'
assert not out.exists(),'Keep prior evidence; choose a new fixture path to repeat'
out.mkdir()
torch.manual_seed(1729);np.random.seed(1729);random.seed(1729)
model=torch.nn.Linear(3,2).cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=.001)
ckpt=Checkpoints(out,0,1,'synthetic-checkpoint-fault-test-v1')
def step():
    optimizer.zero_grad(set_to_none=True)
    model(torch.randn(4,3,device='cuda')).square().mean().backward()
    optimizer.step();optimizer.zero_grad(set_to_none=True)
step();state1={'update':1,'epoch':0,'next_chunk':1}
ckpt.save(model,optimizer,state1)
saved=torch.load(out/'checkpoints/update-000000001.pt',map_location='cpu',weights_only=False)
step();ckpt.save(model,optimizer,{'update':2,'epoch':0,'next_chunk':2})
# Simulated power loss leaves an unfinished newer file; it must never load.
(out/'checkpoints/update-000000003.pt.partial').write_bytes(b'incomplete synthetic checkpoint')
# A corrupted pointer must not hide its independently committed sidecar.
latest=json.loads((out/'latest.json').read_text());latest['sha256']='0'*64
atomic_json(out/'latest.json',latest)
state,selection=ckpt.load(model,optimizer)
assert state['update']==2 and selection['fallback_failures']
# A same-size corruption of the latest checkpoint forces the retained fallback.
target=out/'checkpoints/update-000000002.pt'
with target.open('r+b') as f:
    f.seek(500);byte=f.read(1);f.seek(500);f.write(bytes([byte[0]^1]))
state,selection=ckpt.load(model,optimizer)
assert state==state1 and selection['fallback_failures']
actual={k:v.detach().cpu() for k,v in model.state_dict().items()}
compare(saved['model'],actual)
def cpu_tree(x):
    if isinstance(x,torch.Tensor):return x.cpu()
    if isinstance(x,dict):return {k:cpu_tree(v) for k,v in x.items()}
    if isinstance(x,list):return [cpu_tree(v) for v in x]
    return x
compare(saved['optimizer'],cpu_tree(optimizer.state_dict()))
compare(saved['rng_by_rank'][0],capture_rng())
try:Checkpoints(out,0,1,'wrong-contract').load(model,optimizer)
except RuntimeError as e:assert 'fingerprint mismatch' in str(e)
else:raise AssertionError('Changed run contract accepted')
try:Checkpoints(out,0,2,'synthetic-checkpoint-fault-test-v1').load(model,optimizer)
except RuntimeError as e:assert 'world size changed' in str(e)
else:raise AssertionError('Changed GPU count accepted')
lost=root/'qualification/lost-checkpoint-fixture';lost.mkdir()
atomic_json(lost/'checkpoint-status.json',{'last_committed_update':1})
try:Checkpoints(lost,0,1,'synthetic-checkpoint-fault-test-v1').load(model,optimizer)
except RuntimeError as e:assert 'lost its checkpoints' in str(e)
else:raise AssertionError('Missing previously committed state silently restarted')
result={'passed':True,'partial_ignored':True,'corrupt_pointer_recovered_from_sidecar':True,
        'corrupt_latest_fell_back':True,'fallback_model_optimizer_rng_exact':True,
        'changed_contract_rejected':True,'changed_world_size_rejected':True,
        'lost_all_checkpoints_rejected':True,'scope':'small synthetic model; no production files modified',
        'fallback_selection':selection}
atomic_json(root/'qualification/checkpoint-faults.json',result)
print(json.dumps(result,indent=2))
