"""CPU qualification against unmodified native scorer, no third-party Python needed."""
import hashlib,json,subprocess,datetime,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'benchmark-v5'
OUT=ROOT/'qualification/sprint-requested'
def record(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def run(binary,name,proteins,hsp,train,pairs,empty):
 dest=OUT/(name+'.scores')
 for suffix in ['', '.pos','.neg']:
  p=Path(str(dest)+suffix)
  if p.exists():p.rename(p.with_name(p.name+'.previous-'+str(os.getpid())))
 cmd=[str(binary),'-p',str(proteins),'-h',str(hsp),'-tr',str(train),'-pos',str(pairs),'-neg',str(empty),'-o',str(dest),'-Thc','40']
 with (OUT/(name+'.log')).open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 return dest
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 fixture=OLD/'qualification/sprint-requested'
 args=[OLD/'data/sprint/proteins.fasta',OLD/'predictions/sprint/canonical.hsp',fixture/'train.txt',fixture/'pairs.txt',fixture/'empty.txt']
 native=run(ROOT/'bin/sprint_predict_native_cpu','native-cpu',*args)
 sparse=run(ROOT/'bin/sprint_predict_sparse_cpu','sparse-cpu',*args)
 assert native.read_bytes()==sparse.read_bytes(),'Sparse/native score mismatch'
 reference=fixture/'native.scores'
 assert native.read_bytes()==reference.read_bytes(),'CPU/SIF archived native score mismatch'
 # Exercise original self-training branch as well as reverse/duplicate candidates.
 train=OUT/'self-training.txt';old=(fixture/'train.txt').read_text();seen=[]
 for line in (fixture/'pairs.txt').read_text().splitlines():
  for name in line.split():
   if name not in seen:seen.append(name)
   if len(seen)>=6:break
  if len(seen)>=6:break
 train.write_text(old+''.join(f'{p} {p}\n' for p in seen))
 args[2]=train
 self_native=run(ROOT/'bin/sprint_predict_native_cpu','native-cpu-self',*args)
 self_sparse=run(ROOT/'bin/sprint_predict_sparse_cpu','sparse-cpu-self',*args)
 assert self_native.read_bytes()==self_sparse.read_bytes(),'Self-training score mismatch'
 rows=[float(line.split()[0]) for line in native.read_text().splitlines()]
 result={'passed':True,'at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
  'native_output_identical_bytes':True,'native_cpu_matches_archived_sif_bytes':True,'self_training_identical_bytes':True,
  'pairs':len(rows),'nonzero_pairs':sum(v!=0 for v in rows),'training_edges':len(old.splitlines()),
  'full_corpus_and_full_native_hsp_preprocessing':True,'self_reversed_duplicate_candidates_included':True,
  'only_change':'Native serial arithmetic for requested unordered cells; preserve training, outer HSP and per-cell inner HSP order; unchanged native high-count preprocessing and normalization.',
  'outputs':[record(p) for p in [native,sparse,self_native,self_sparse,reference]],
  'binaries':[record(ROOT/'bin'/n) for n in ['sprint_predict_native_cpu','sprint_predict_sparse_cpu']],
  'sources':[record(p) for p in sorted((ROOT/'scripts/sprint_sparse').glob('*'))],
  'fixtures':[record(p) for p in [*args,fixture/'train.txt']]}
 p=OUT/'qualification.json';tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');os.replace(tmp,p)
 print(json.dumps({k:v for k,v in result.items() if k not in ['outputs','binaries','sources','fixtures']}),flush=True)
if __name__=='__main__':main()
