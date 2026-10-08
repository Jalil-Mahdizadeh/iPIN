"""Verify frozen, qualified inputs and submit the authorized five-hour pilot once."""
import argparse
import json
import os
import subprocess
import time
from common import ROOT, PROJECT, atomic, config, now, sha, verify_freeze


def submission_environment():
    return {k:v for k,v in os.environ.items()
            if not (k.startswith(('SBATCH_','SRUN_')) or
                    (k.startswith('SLURM_') and k not in ['SLURM_CONF','SLURM_CONF_SERVER']) or
                    k in ['CUDA_VISIBLE_DEVICES','NVIDIA_VISIBLE_DEVICES'])}


def main():
    p=argparse.ArgumentParser();p.add_argument('--submit',action='store_true');args=p.parse_args()
    cfg=config();fingerprint=verify_freeze()
    for rel in ['qualification/unit.json','qualification/encoder.json','qualification/real-encoder.json']:
        assert json.loads((ROOT/rel).read_text())['passed'],rel
    assert json.loads((ROOT/'results/archive-coverage.json').read_text())['complete']
    resources=json.loads((ROOT/'provenance/resource-start.json').read_text())
    if time.time()>resources['qualification_deadline_unix']:raise RuntimeError('Qualification allocation reserve exhausted; no automatic extension')
    freeze=json.loads((ROOT/'provenance/freeze.json').read_text())
    for rel,digest in freeze['artifacts'].items():
        if sha(PROJECT/rel)!=digest:raise ValueError(f'Large artifact mismatch: {rel}')
    usage=sum(p.stat().st_size for d in [ROOT,PROJECT/'images/msa-pairformer'] for p in d.rglob('*') if p.is_file() and not p.is_symlink())
    if usage>cfg['budgets']['additional_bytes']:raise RuntimeError('Storage limit exceeded')
    command=['sbatch','--parsable',f'--output={ROOT}/logs/pilot-%j.log',str(ROOT/'slurm/pilot.sbatch'),str(ROOT)]
    plan={'at_utc':now(),'fingerprint':fingerprint,'command':command,'batch_gpu_hours_max':20,
          'qualification_gpu_hours_charged':4,'cpu_core_hours_max_reserved':648,'additional_bytes_at_launch':usage,
          'no_automatic_requeue':True,'no_production_continuation':True}
    atomic(ROOT/'provenance/launch-plan.json',plan)
    if not args.submit:print(json.dumps(plan,indent=2));return
    if time.time()>resources['qualification_deadline_unix']:raise RuntimeError('Qualification reserve expired during artifact checks')
    intent=ROOT/'provenance/submission-intent.json'
    # Exclusive creation covers the crash window between sbatch and recording its result.
    with intent.open('x') as f:json.dump(plan,f,indent=2);f.flush()
    # Do not export the interactive allocation's GPU visibility or resource variables.
    # Retain site connection configuration while Slurm defines the new job's resources.
    submit_env=submission_environment()
    result=subprocess.run(command,text=True,capture_output=True,env=submit_env)
    if result.returncode:
        atomic(ROOT/'provenance/submission-error.json',{'at_utc':now(),'returncode':result.returncode,'stderr':result.stderr})
        raise RuntimeError(result.stderr)
    job=result.stdout.strip().split(';')[0]
    if not job.isdigit():raise RuntimeError('Unexpected scheduler response; submission intent retained')
    atomic(ROOT/'provenance/submission.json',{**plan,'job_id':job,'scheduler_response':result.stdout.strip()})
    print(json.dumps({'job_id':job,'batch_gpu_hours_max':20,'total_gpu_hours_ceiling':24}),flush=True)


if __name__=='__main__':main()
