"""Finalize automatically when every frozen predictor finishes; never publish partial coverage."""
import datetime,fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def save(value):
 p=ROOT/'STATUS.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,p)
def main():
 lock=(ROOT/'results/finalizer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 roster=json.loads((ROOT/'provenance/roster.json').read_text())['models'];started=now()
 sharded=['ipin-esm2','ipin-esmc','native-human','native-plm','xpair-bernett','xpair-default','dscript']
 while True:
  states={}
  for name in roster:
   folder=ROOT/'predictions'/name
   if name in sharded:
    count=sum((folder/f'rank-{r:02d}.done.json').exists() for r in range(4));states[name]={'done':count==4,'finished_shards':count,'required_shards':4}
   elif name=='sprint':states[name]={'done':(folder/'prediction-command-succeeded').exists()}
   else:states[name]={'done':(folder/'done.json').exists()}
  status={'at_utc':now(),'started_at_utc':started,'phase':'waiting_for_complete_inference','complete':False,'models':states}
  save(status)
  if all(item['done'] for item in states.values()):break
  print(json.dumps({'at_utc':now(),'finished_models':[n for n,v in states.items() if v['done']], 'waiting_for':[n for n,v in states.items() if not v['done']]}),flush=True)
  time.sleep(30)
 status['phase']='analysis_and_verification';save(status)
 try:
  with (ROOT/'logs/final-analysis.log').open('a') as stream:
   subprocess.run(['bash',str(ROOT/'scripts/finish_analysis.sh')],stdout=stream,stderr=subprocess.STDOUT,check=True)
 except Exception as e:
  status.update(at_utc=now(),phase='analysis_failed',error=str(e));save(status);raise
 assert json.loads((ROOT/'results/COMPLETE.json').read_text())['complete']
 status.update(at_utc=now(),phase='complete',complete=True,report=str(ROOT/'REPORT.md'));save(status)
 print('Benchmark complete; see REPORT.md',flush=True)
if __name__=='__main__':main()
