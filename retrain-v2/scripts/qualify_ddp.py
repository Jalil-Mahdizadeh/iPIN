"""Orchestrate strictly bounded four-GPU qualification, never production training."""
import json
import os
import subprocess
import sys
from pathlib import Path
from state import atomic_json, sha256

root=Path(__file__).resolve().parents[1]
base=json.loads((root/'configs/qualification.json').read_text())
qual=root/'qualification';reports=[]
assert os.environ.get('SLURM_JOB_ID'), 'Run inside the bounded qualification allocation'
def run(name,cfg,tiny=False,stop=None,updates=5):
    path=qual/(name.split('-run-')[0]+'.json')
    if path.exists():assert json.loads(path.read_text())==cfg
    else:atomic_json(path,cfg)
    args=['torchrun','--standalone','--nnodes=1','--nproc-per-node=4',str(root/'scripts/train.py'),
          '--config',str(path),'--output',str(qual/name),'--qualification','--max-updates',str(updates),
          '--train-limit','11','--val-limit','8','--qualification-max-tokens','384']
    if tiny:args+=['--tiny']
    if stop is not None:args+=['--stop-after-updates',str(stop)]
    subprocess.run(args,check=True)
def equivalent(prefix,cfg,tiny=False,stop=2,updates=5):
    a,b=prefix+'-run-full',prefix+'-run-resumed'
    run(a,cfg,tiny,updates=updates)
    run(b,cfg,tiny,stop=stop,updates=updates)
    stopped=json.loads((qual/b/'stopped.json').read_text())
    assert stopped['state']['update']==stop and stopped['checkpoint_committed']
    assert stopped['state']['pending_validation'], 'Exercise interrupted scheduled validation'
    run(b,cfg,tiny,updates=updates)
    report=qual/(prefix+'-comparison.json')
    subprocess.run([sys.executable,str(root/'scripts/compare_resume.py'),str(qual/a),str(qual/b),'--report',str(report)],check=True)
    reports.append(str(report.relative_to(root)))

# Full backbone: 3 pairs / 4 GPUs explicitly exercises a zero-contribution rank.
equivalent('ddp-reference',base)
# Uneven multi-microbatch accumulation, added head, hybrid attention, two encoder passes.
experimental={**base,'global_pairs_per_update':9,'train_max_pairs':1,
              'attention_mode':'chain_aware','readout':'residue_mean','classification_corruption':False}
equivalent('ddp-experimental',experimental,updates=3,stop=3)
# Remaining objective branches under DDP: frozen nonshared MLM parameters must
# neither freeze encoder embeddings nor leave unexpected unused trainable params.
for arm in ['masked-bce','clean-bce','positive10','cls-mlp']:
    cfg=json.loads((root/f'configs/{arm}-official-seed2.json').read_text())
    cfg.update(global_pairs_per_update=9,train_max_pairs=1,validate_every_updates=2,checkpoint_every_updates=1,warmup_updates=1)
    run('ddp-branch-'+arm,cfg,tiny=True,updates=2)
    completed=json.loads((qual/('ddp-branch-'+arm)/'completed.json').read_text())
    assert len(set(completed['model_hashes_by_rank']))==1 and not completed['state']['pending_validation']
report={'passed':True,'job_id':os.environ['SLURM_JOB_ID'],'world_size':4,
    'production_training':False,'full_650m_resume_comparisons':reports,
    'dummy_rank_exercised':True,'uneven_accumulation_exercised':True,
    'interrupted_final_validation_recovered':True,'additional_tiny_ddp_branches':['masked-bce','clean-bce','positive10','cls-mlp'],
    'data_manifest_sha256':sha256(root/'data/prepared/manifest.json'),
    'code':{n:sha256(root/'scripts'/n) for n in ['train.py','model.py','data.py','state.py','release.py']}}
atomic_json(qual/'ddp.json',report)
print(json.dumps(report,indent=2))
