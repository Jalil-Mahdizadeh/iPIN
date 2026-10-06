"""Verify the frozen campaign and print submission commands by default (stdlib only)."""
import argparse
import datetime
import hashlib
import json
import shlex
import subprocess
from pathlib import Path

ROOT=Path('/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v2')
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
    return h.hexdigest()
def verify(release):
    assert release.is_relative_to(ROOT/'releases') and release.is_dir()
    m=json.loads((release/'release.json').read_text())
    assert m['ready_for_submission'] and not m['production_submitted_at_preparation']
    for name,digest in m['files'].items():assert sha(release/name)==digest,('Release changed',name)
    assert sha(ROOT/'data/prepared/manifest.json')==m['data_manifest_sha256']
    for name,digest in json.loads((ROOT/'data/prepared/manifest.json').read_text())['files'].items():
        assert sha(ROOT/'data/prepared'/name)==digest,('Data changed',name)
    assert sha(ROOT/'provenance/downloads.json')==m['initialization_manifest_sha256']
    for item in json.loads((ROOT/'provenance/downloads.json').read_text()):
        assert sha(ROOT/item['local_path'])==item['sha256'],item['local_path']
    for name,digest in m['qualification_reports'].items():
        assert sha(ROOT/name)==digest and json.loads((ROOT/name).read_text())['passed'],name
    assert sha(ROOT.parent/'images/plm-interact/plm-interact-native-arm64-v1.sif')==m['sif_sha256']
    return m
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--release',default=None)
    p.add_argument('--run',help='One enabled run, including exact-contract continuation')
    p.add_argument('--max-concurrent',type=int,choices=range(1,5),default=2,
                   help='Maximum simultaneously running array tasks (1–4); each task uses four GPUs')
    p.add_argument('--submit',action='store_true',help='Actually submit; absent means dry run')
    args=p.parse_args()
    release=Path(args.release).resolve() if args.release else ROOT/'releases'/(ROOT/'releases/CURRENT').read_text().strip()
    m=verify(release)
    runs=[args.run] if args.run else m['enabled_runs']
    assert all(name in m['enabled_runs'] for name in runs)
    for name in runs:
        out=ROOT/'runs'/name
        assert not (out/'completed.json').exists(),f'{name} is already complete; select an unfinished run'
        assert not (out/'REQUEST_STOP').exists(),f'{name} is manually stopped'
        if (out/'contract.json').exists():
            contract=json.loads((out/'contract.json').read_text())
            assert contract['configuration']==json.loads((release/'configs'/f'{name}.json').read_text()),'Resume configuration mismatch'
            assert contract['code']=={n:sha(release/'code'/n) for n in contract['code']},'Resume code mismatch'
    command=['sbatch','--parsable']
    if not args.run:command+=[f'--array=0-{len(runs)-1}%{args.max_concurrent}']
    command+=[str(release/'slurm/train.sbatch'),str(release)]
    if args.run:command+=[args.run]
    plan={'time_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'dry_run':not args.submit,
        'release':str(release),'runs':runs,'command':command,'shell_command':shlex.join(command),
        'gpus_per_run':4,'maximum_concurrent_runs':1 if args.run else args.max_concurrent,
        'launcher_path':str(Path(__file__).resolve()),'launcher_sha256':sha(Path(__file__).resolve()),
        'production_jobs_submitted_by_this_call':0,'qualified':True}
    if args.submit:
        # Prevent duplicate queue submissions; run-level filesystem locks provide
        # a second check inside the trainer if a launch races this read.
        active=subprocess.run(['squeue','--me','--noheader','--format=%j'],text=True,capture_output=True,check=True).stdout
        assert 'plmi-v2-screen' not in active,'An active v2 campaign exists; inspect it before resubmitting'
        job=subprocess.run(command,text=True,capture_output=True,check=True).stdout.strip()
        plan.update(slurm_job_id=job,production_jobs_submitted_by_this_call=len(runs))
        destination=ROOT/'provenance'/f'submission-{job.split(";")[0]}.json'
    else:destination=ROOT/'provenance/production-launch-plan.json'
    destination.write_text(json.dumps(plan,indent=2)+'\n')
    print(json.dumps(plan,indent=2))
if __name__=='__main__':main()
