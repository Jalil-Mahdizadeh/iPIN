"""Finish the authorized historical comparison once all required scores exist."""
import datetime, fcntl, json, os, subprocess, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
EXT=ROOT/'v1-v4-comparison'


def save(status):
    status['at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    path=EXT/'STATUS.json';tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(status,indent=2)+'\n');os.replace(tmp,path)


def main():
    lock=(EXT/'results/finalizer.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    while True:
        states={}
        for name in ['v2-capped','v2-clean-bce','native-human','native-plm','ipin-esm2','ipin-esmc']:
            base=EXT if name.startswith('v2-') else ROOT
            states[name]=sum((base/'predictions'/name/f'rank-{i:02d}.done.json').exists() for i in range(4))
        status={'phase':'waiting_for_complete_inference','complete':False,'finished_shards':states}
        save(status)
        if all(value==4 for value in states.values()):break
        time.sleep(30)
    status['phase']='analysis_and_verification';save(status)
    try:
        with (EXT/'logs/final-analysis.log').open('a') as log:
            subprocess.run(['bash',str(ROOT/'scripts/container.sh'),'analysis','python','scripts/historical_analyze.py'],
                           cwd=ROOT.parent,stdout=log,stderr=subprocess.STDOUT,check=True)
        assert json.loads((EXT/'results/COMPLETE.json').read_text())['complete']
    except Exception as error:
        status.update(phase='analysis_failed',error=str(error));save(status);raise
    status.update(phase='complete',complete=True,report=str(EXT/'REPORT.md'));save(status)


if __name__=='__main__':main()
