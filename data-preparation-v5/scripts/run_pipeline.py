"""Stage-resumable CPU preparation, with immutable inputs and checked outputs."""
import concurrent.futures
import fcntl
import hashlib
import json
import os
import platform
import signal
import subprocess
import threading
from datetime import datetime,timezone
from common import ROOT,PROJECT,read_json,write_json,sha

LOCK=threading.RLock();CHILDREN={};FAILED=threading.Event();FINISHED=[]

def state(status='running',error=None):
    with LOCK:
        write_json(ROOT/'pipeline-state.json',dict(status=status,pid=os.getpid(),host=platform.node(),
            slurm_job_id=os.environ.get('SLURM_JOB_ID'),updated_utc=datetime.now(timezone.utc).isoformat(),
            active_stages={k:v.pid for k,v in CHILDREN.items()},finished_stages=FINISHED.copy(),error=error))

def stop_children():
    FAILED.set()
    with LOCK:processes=list(CHILDREN.values())
    for p in processes:
        if p.poll() is None:
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass

def stage(name,command,outputs,fingerprint,adopt_status=None):
    try:
        return run_stage(name,command,outputs,fingerprint,adopt_status)
    except BaseException:
        stop_children()
        raise

def run_stage(name,command,outputs,fingerprint,adopt_status=None):
    if FAILED.is_set():raise RuntimeError('An earlier stage failed; dependent work cancelled')
    receipt=ROOT/'work/stages'/f'{name}.json';receipt.parent.mkdir(exist_ok=True)
    if receipt.exists():
        previous=read_json(receipt)
        assert previous['fingerprint']==fingerprint,f'Stage contract changed: {name}'
        for path,digest in previous['outputs'].items():assert sha(ROOT/path)==digest,f'Completed output changed: {path}'
    else:
        adopted=False
        if adopt_status:
            p=ROOT/'work'/f'{adopt_status}.status.json'
            adopted=p.exists() and read_json(p)['status']=='complete' and all((ROOT/o).exists() for o in outputs)
        if not adopted:
            log=ROOT/'logs'/f'{name}.log'
            with log.open('a') as out:
                out.write(f'\nSTART {datetime.now(timezone.utc).isoformat()} {command!r}\n');out.flush()
                proc=subprocess.Popen(command,cwd=ROOT,stdout=out,stderr=subprocess.STDOUT,start_new_session=True)
                with LOCK:CHILDREN[name]=proc
                state()
                try:
                    code=proc.wait()
                    if code:raise RuntimeError(f'{name} exited {code}; see {log.relative_to(ROOT)}')
                except BaseException:
                    stop_children();raise
                finally:
                    with LOCK:CHILDREN.pop(name,None)
        for o in outputs:assert (ROOT/o).is_file(),f'Missing stage output: {o}'
        write_json(receipt,dict(fingerprint=fingerprint,adopted_verified_prelaunch_stage=adopted,command=command,
                    completed_utc=datetime.now(timezone.utc).isoformat(),outputs={o:sha(ROOT/o) for o in outputs}))
    with LOCK:FINISHED.append(name)
    state()

def main():
    lock=(ROOT/'work/pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert os.environ.get('SLURM_JOB_ID'),'Run preparation inside a SLURM compute allocation'
    assert len(os.sched_getaffinity(0))>=24,'Preparation requires at least 24 allocated CPU cores'
    assert read_json(ROOT/'reports/qualification.json')['passed']
    paths=[ROOT/'configuration.json',ROOT/'environment/requirements.lock',ROOT/'environment/vocabularies.json',
           ROOT/'environment/runtime.json',ROOT/'environment/kahip-smoke.json',ROOT/'provenance/mmseqs-binary.sha256',
           PROJECT/'images/plm-interact-esmc/plm-interact-esmc-arm64-v1.sif',
           ROOT/'sources/hippie-v3-snapshot.txt.gz',ROOT/'sources/author-hippie-v3.csv',ROOT/'provenance/uniprot-downloads.json',
           PROJECT/'retrain-v1/data/raw/pairs_uniprot_seqs_train.csv',PROJECT/'retrain-v1/data/raw/pairs_uniprot_seqs_val.csv',
           ROOT/'sources/author-pipeline/bin/sample_negatives_ilp.py',ROOT/'sources/author-pipeline/bin/utils.py',
           *sorted((ROOT/'scripts').glob('*.py')),*sorted((ROOT/'scripts').glob('*.sh'))]
    identities={os.path.relpath(p,ROOT):sha(p) for p in paths}
    fingerprint=hashlib.sha256(json.dumps(identities,sort_keys=True).encode()).hexdigest()
    contract_path=ROOT/'provenance/execution-contract.json'
    contract=dict(fingerprint=fingerprint,identities=identities,slurm_job_id=os.environ.get('SLURM_JOB_ID'),host=platform.node(),
                  resource_policy='CPU-only: 16-thread sequence search, up to three 8-thread ILPs; no new GPU or training job',
                  solver_time_limits_seconds=7200,models_trained=False,model_inference=False)
    if contract_path.exists():assert read_json(contract_path)['fingerprint']==fingerprint,'Execution inputs changed; refuse silent resume'
    else:write_json(contract_path,contract)
    py=['bash',str(ROOT/'scripts/python.sh')]
    try:
        state()
        stage('bootstrap',py+[str(ROOT/'scripts/bootstrap.py')],['frozen-tests/original-test.csv','frozen-tests/original-manifest.json','work/hippie-positive-evidence.csv','work/requested-accessions.json'],fingerprint)
        stage('uniprot', ['python',str(ROOT/'scripts/fetch_uniprot.py')],['provenance/uniprot-downloads.json'],fingerprint,adopt_status='uniprot')
        stage('normalize',py+[str(ROOT/'scripts/normalize.py')],['work/protein-registry.json','work/test-protein-metadata.json','work/known-positive-families.tsv','work/eligible-before-homology.csv','work/eligible-before-homology.fasta','reports/normalization-audit.json'],fingerprint,adopt_status='normalize')
        stage('historical_exposure',py+[str(ROOT/'scripts/audit_history.py')],['reports/historical-exposure.json'],fingerprint)
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
            test_future=pool.submit(stage,'negative_test-ilp',py+[str(ROOT/'scripts/sample_ilp.py'),'test-ilp'],['frozen-tests/test-ilp.csv','reports/test-ilp-negative-sampling.json'],fingerprint)
            try:
                stage('homology',['bash','-l',str(ROOT/'scripts/homology.sh')],['work/eligible-positives.csv','work/eligible.fasta','work/protein-groups.json','work/development-all-hits.tsv','reports/test-homology-filter.json','reports/grouping.json'],fingerprint)
                stage('positive_split',py+[str(ROOT/'scripts/split_positives.py')],['work/train-positives.csv','work/val-positives.csv','work/train.fasta','work/val.fasta','reports/positive-split.json'],fingerprint)
                stage('boundary_verification',['bash','-l',str(ROOT/'scripts/verify_boundaries.sh')],['reports/final-development-homology.json','work/boundary_verification.status.json'],fingerprint)
                jobs=[test_future]
                for split in ['train','val']:
                    jobs.append(pool.submit(stage,f'negative_{split}',py+[str(ROOT/'scripts/sample_ilp.py'),split],[f'prepared/{split}.csv',f'reports/{split}-negative-sampling.json'],fingerprint))
                for job in concurrent.futures.as_completed(jobs):job.result()
            except BaseException:stop_children();raise
        stage('finalize',py+[str(ROOT/'scripts/finalize.py')],['completed.json','reports/final-audit.json','REPORT.md'],fingerprint)
        state('complete')
    except BaseException as exc:
        stop_children();state('failed',str(exc));raise

if __name__=='__main__':
    def interrupted(signum,frame):
        stop_children();state('interrupted',f'signal {signum}');raise SystemExit(128+signum)
    signal.signal(signal.SIGTERM,interrupted);signal.signal(signal.SIGINT,interrupted)
    main()
