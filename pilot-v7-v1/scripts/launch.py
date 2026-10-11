"""Launch once in the verified existing idle GH200 allocation; never submit/requeue."""
import argparse
import json
import os
import subprocess
from study import ROOT, PROJECT, config, read, atomic, now, sha, verify_freeze, remaining, owned_bytes


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--start',action='store_true');args=ap.parse_args()
    fp=verify_freeze();cfg=config();q=read(ROOT/'qualification/real.json');resource=read(ROOT/'provenance/resource-start.json')
    if not q['passed'] or not read(ROOT/'qualification/unit.json')['passed']:raise ValueError('Not qualified')
    if os.environ.get('SLURM_JOB_ID')!=resource['job_id'] or os.environ.get('CUDA_VISIBLE_DEVICES')!=resource['cuda_visible_devices']:
        raise ValueError('Allocation/device changed')
    if q['conservative_total_seconds']>remaining(cfg['budgets']['shutdown_reserve_seconds']):raise TimeoutError('Insufficient study budget')
    time_left=subprocess.check_output(['squeue','--noheader','--jobs',resource['job_id'],'--format=%L'],text=True).strip()
    days,clock=time_left.split('-',1) if '-' in time_left else ('0',time_left)
    allocation_seconds=int(days)*86400+sum(int(v)*scale for v,scale in zip(reversed(clock.split(':')),[1,60,3600]))
    if allocation_seconds<q['conservative_total_seconds']+cfg['budgets']['shutdown_reserve_seconds']:raise TimeoutError('Allocation expires too early')
    apps=subprocess.check_output(['nvidia-smi','-i',resource['cuda_visible_devices'],'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    if apps:raise RuntimeError('Allocated GPU is occupied; do not interfere')
    if sha(PROJECT/cfg['image'])!=cfg['image_sha256']:raise ValueError('Container checksum mismatch')
    if owned_bytes()>=cfg['budgets']['additional_bytes_ceiling']:raise RuntimeError('Storage ceiling exhausted')
    cmd=['apptainer','exec','--nv','--env','OPENBLAS_NUM_THREADS=1','--env','OMP_NUM_THREADS=8',str(PROJECT/cfg['image']),
         'python','-B',str(ROOT/'scripts/run.py')]
    plan={'at_utc':now(),'fingerprint':fp,'job_id':resource['job_id'],'command':cmd,'new_scheduler_submission':False,
        'conservative_seconds':q['conservative_total_seconds'],'allocation_remaining_seconds':allocation_seconds,
        'separate_gpu_hour_ceiling':cfg['budgets']['allocated_gpu_hours_ceiling'],'test_accessed':False}
    atomic(ROOT/'provenance/launch-plan.json',plan)
    if not args.start:print(plan,flush=True);return
    with (ROOT/'provenance/execution-intent.json').open('x') as f:
        json.dump(plan,f,indent=2);f.flush();os.fsync(f.fileno())
    with (ROOT/'logs/controller.log').open('xb') as log:
        p=subprocess.Popen(cmd,cwd=PROJECT,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
    atomic(ROOT/'provenance/launch.json',{**plan,'controller_pid':p.pid})
    print({'started':True,'controller_pid':p.pid,'job_id':resource['job_id'],'fingerprint':fp},flush=True)


if __name__=='__main__':main()
