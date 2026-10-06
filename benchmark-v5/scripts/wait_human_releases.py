"""Keep the explicitly requested two-release extension running through final analysis."""
import datetime,fcntl,json,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def write(value):
    value['at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat()
    p=ROOT/'human-releases-status.json';tmp=p.with_name(p.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n');os.replace(tmp,p)
lock=(ROOT/'logs/human-finalizer.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
while True:
    done=sum((ROOT/'predictions/native-human'/f'rank-{r:02d}.done.json').exists() for r in range(4))
    tuna=(ROOT/'predictions/tuna-human/done.json').exists()
    committed=sum(json.loads(p.read_text())['rows'] for p in (ROOT/'predictions/native-human').glob('rank-*-chunk-*.json'))
    status={'complete':False,'phase':'inference','native_human_shards':done,'native_human_committed_pairs':committed,
            'native_human_total_pairs':76918,'tuna_human_complete':tuna,'native_job':'3396514'}
    write(status);print(status,flush=True)
    if done==4 and tuna:break
    time.sleep(30)
write({**status,'phase':'analysis'})
with (ROOT/'logs/human-releases-analysis.log').open('a') as out:
    result=subprocess.run(['bash',str(ROOT/'scripts/finish_human_releases.sh')],stdout=out,stderr=subprocess.STDOUT)
if result.returncode:
    write({**status,'phase':'analysis_failed','returncode':result.returncode});raise SystemExit(result.returncode)
complete=json.loads((ROOT/'completed.json').read_text());assert complete['complete'] and complete['checks']['predictors']==11
write({**status,'complete':True,'phase':'complete','completion':'completed.json','report':'REPORT.md'})
print('Eleven-model benchmark extension complete',flush=True)
