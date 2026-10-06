"""Resume only unfinished sampling; preserve completed original stages verbatim."""
import fcntl
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from common import read_json,write_json,sha

CHILDREN={}
FINISHED=['bootstrap','uniprot','normalize','historical_exposure','homology','negative_test-ilp','positive_split','boundary_verification']


def state(status='running',error=None):
    write_json(ROOT/'pipeline-state.json',dict(status=status,recovery='v1',pid=os.getpid(),host=platform.node(),
        slurm_job_id=os.environ.get('SLURM_JOB_ID'),updated_utc=datetime.now(timezone.utc).isoformat(),
        active_stages={k:p.pid for k,p in CHILDREN.items()},finished_stages=FINISHED.copy(),error=error))


def stop():
    for p in CHILDREN.values():
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass


def freeze_contract():
    original=read_json(ROOT/'provenance/execution-contract.json')
    for name,expected in original['identities'].items():assert sha(ROOT/name)==expected,name
    for name in FINISHED:
        receipt=read_json(ROOT/f'work/stages/{name}.json')
        for path,expected in receipt['outputs'].items():assert sha(ROOT/path)==expected,path
    qualification=read_json(ROOT/'reports/recovery-v1-qualification.json');assert qualification['passed']
    paths=[*sorted((ROOT/'recovery-v1').glob('*.py')),ROOT/'provenance/execution-contract.json',
           ROOT/'reports/recovery-v1-qualification.json',ROOT/'configuration.json',
           ROOT/'work/protein-registry.json',ROOT/'work/known-positive-families.tsv']
    for split in ['train','val']:
        paths += [ROOT/f'work/{split}-positives.csv']
        paths += sorted((ROOT/f'work/negative-{split}').glob('*'))
    identities={os.path.relpath(p,ROOT):sha(p) for p in paths if p.is_file()}
    policy=dict(solver_seconds_per_split=1800,no_improvement_seconds=300,threads_per_split=8,
                max_concurrent_samplers=2,watchdog_grace_seconds=90,checkpoint_every_improving_incumbent=True,
                candidate_pools_regenerated=False,positive_partition_changed=False,test_changed=False,
                unchanged_initialization_without_optimality_not_published=True)
    fingerprint=hashlib.sha256(json.dumps(dict(identities=identities,policy=policy),sort_keys=True).encode()).hexdigest()
    contract=dict(fingerprint=fingerprint,identities=identities,policy=policy,
                  parent_contract=original['fingerprint'],created_utc=datetime.now(timezone.utc).isoformat(),
                  correction='Explicit complete feasible MIP start; equivalent equality/deviation model with scaled objective; IPM relaxation; durable checked incumbents',
                  original_runtime_failure='Validation reached time limit without an integer incumbent; original driver cancelled training',
                  models_trained=False,model_inference=False)
    path=ROOT/'provenance/recovery-v1/contract.json'
    if path.exists():assert read_json(path)['fingerprint']==fingerprint,'Recovery contract changed; refuse silent resume'
    else:write_json(path,contract)
    return contract


def main():
    lock=(ROOT/'work/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ.get('SLURM_JOB_ID') and len(os.sched_getaffinity(0))>=16
    contract=freeze_contract();state()
    failures={};opened=[];starts={};signalled={}
    try:
        for split in ['train','val']:
            name=f'negative_{split}';receipt=ROOT/f'work/recovery-v1/{split}/completed.json'
            if receipt.exists():
                completed=read_json(receipt);assert completed['contract']==contract['fingerprint']
                for path,digest in completed['outputs'].items():assert sha(ROOT/path)==digest,path
                FINISHED.append(name);continue
            log=(ROOT/f'logs/recovery-v1-{split}.log').open('a');opened.append(log)
            command=['bash',str(ROOT/'scripts/python.sh'),str(ROOT/'recovery-v1/worker.py'),split]
            proc=subprocess.Popen(command,cwd=ROOT.parent,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            CHILDREN[name]=proc;starts[name]=time.monotonic();state()
        while CHILDREN:
            for name,p in list(CHILDREN.items()):
                code=p.poll()
                if code is None:
                    # Bound setup + solve time independently of solver callbacks.
                    maximum=contract['policy']['solver_seconds_per_split']+180
                    elapsed=time.monotonic()-starts[name]
                    if elapsed>maximum:
                        if name not in signalled:os.killpg(p.pid,signal.SIGTERM);signalled[name]=time.monotonic()
                        elif time.monotonic()-signalled[name]>contract['policy']['watchdog_grace_seconds']:os.killpg(p.pid,signal.SIGKILL)
                    continue
                CHILDREN.pop(name);split=name.removeprefix('negative_')
                if code:
                    failures[name]=f'exit {code}; checked checkpoint retained in work/recovery-v1/{split}'
                else:
                    outputs={path:sha(ROOT/path) for path in [f'prepared/{split}.csv',f'reports/{split}-negative-sampling.json']}
                    write_json(ROOT/f'work/recovery-v1/{split}/completed.json',dict(contract=contract['fingerprint'],outputs=outputs))
                    FINISHED.append(name)
                state(error=failures or None)
            if CHILDREN:time.sleep(2)
        if failures:raise RuntimeError(json.dumps(failures))
        with (ROOT/'logs/recovery-v1-finalize.log').open('a') as log:
            p=subprocess.Popen(['bash',str(ROOT/'scripts/python.sh'),str(ROOT/'recovery-v1/finalize.py')],
                               cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            CHILDREN['finalize']=p;state();code=p.wait();CHILDREN.pop('finalize')
            assert code==0,'Final checks failed; see logs/recovery-v1-finalize.log'
        FINISHED.append('finalize');state('complete')
    except BaseException as exc:
        stop();state('failed',str(exc));raise
    finally:
        for f in opened:f.close()


if __name__=='__main__':
    def interrupted(signum,frame):stop();raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    main()
