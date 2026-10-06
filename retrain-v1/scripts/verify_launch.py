"""Verify actual production optimizer checkpoints after their first commit."""
import argparse
import datetime
import json
import time
from pathlib import Path
import torch
from data import PairData
from state import atomic_json,sha256

root=Path(__file__).resolve().parents[1];torch.set_num_threads(4)
parser=argparse.ArgumentParser();parser.add_argument('--slurm-state',type=Path,required=True)
args=parser.parse_args()
names=['symmetric-seed2','reference-seed2'];deadline=time.monotonic()+900
while True:
    ready=True
    for name in names:
        p=root/'runs'/name/'checkpoint-status.json'
        ready=ready and p.exists() and json.loads(p.read_text())['last_committed_update']>=100
    if ready:break
    assert time.monotonic()<deadline,'First optimizer checkpoint did not arrive; inspect jobs'
    time.sleep(5)
data=PairData(root/'data/prepared','train');reports={}
for name in names:
    run=root/'runs'/name;meta=json.loads((run/'latest.json').read_text())
    p=run/'checkpoints'/meta['file'];assert p.stat().st_size==meta['bytes'] and sha256(p)==meta['sha256']
    payload=torch.load(p,map_location='cpu',mmap=True,weights_only=False)
    state=payload['training_state'];config=json.loads((run/'config.json').read_text())
    contract=json.loads((run/'contract.json').read_text())
    assert payload['fingerprint']==contract['fingerprint']==meta['fingerprint']
    assert payload['world_size']==4 and len(payload['rng_by_rank'])==4
    assert state['total_steps']==12745 and state['update']>=100
    plan=data.plan(state['epoch'],config['seed'],config['global_pairs_per_update'])
    expected_examples=state['epoch']*len(data)+sum(map(len,plan[:state['next_chunk']]))
    assert expected_examples==state['examples_seen']
    opt=payload['optimizer'];states=opt['state']
    assert len(states)==sum(len(g['params']) for g in opt['param_groups'])==538
    for x in states.values():
        assert int(x['step'].item())==state['update']
        assert x['exp_avg'].dtype==torch.float32 and x['exp_avg_sq'].dtype==torch.float32
    n=sum(x['exp_avg'].numel() for x in states.values());assert n==651043874
    reports[name]={'checkpoint':meta,'state':state,'adam_parameter_states':len(states),
                   'trainable_parameters_with_optimizer_moments':n,'saved_rank_rng_states':4,
                   'checksum_verified':True,'sampler_position_verified':True}
    del payload,states,opt
ids=[str(x['job_id']) for x in json.loads((root/'provenance/submitted-jobs.json').read_text())]
# Slurm client programs live on the host, outside this scientific container.
slurm=args.slurm_state.read_text()
running={line.split('|')[0]:line.split('|')[1] for line in slurm.strip().splitlines()}
assert all(running.get(job)=='RUNNING' for job in ids),slurm
report={'passed':True,'verified_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'runs':reports,'slurm_state':slurm,'slurm_snapshot_file':str(args.slurm_state),
        'training_complete':False,'test_evaluated':False}
atomic_json(root/'provenance/launch-verification.json',report)
print(json.dumps(report,indent=2))
