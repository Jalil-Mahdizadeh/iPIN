"""Complete native SPRINT CPU stages after HSP construction and qualification."""
import datetime,hashlib,json,os,socket,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'predictions/sprint'
def record(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
 return {'path':str(p),'bytes':p.stat().st_size,'sha256':h.hexdigest()}
def atomic(p,d):
 temp=p.with_name(p.name+'.tmp-'+str(os.getpid()));temp.write_text(json.dumps(d,indent=2)+'\n');os.replace(temp,p)
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def canonicalize():
 assert (OUT/'hsp-command-succeeded').exists()
 raw=OUT/'raw.hsp';canonical=OUT/'canonical.hsp';contract=record(ROOT/'provenance/sprint-input.json')['sha256']
 marker=OUT/'hsp-complete.json'
 if marker.exists():
  m=json.loads(marker.read_text());assert m['input_sha256']==contract and record(canonical)['sha256']==m['file']['sha256'];return
 sequences={};name=None
 for line in (ROOT/'data/sprint/proteins.fasta').open():
  if line.startswith('>'):name=line[1:].strip();sequences[name]=''
  else:sequences[name]+=line.strip()
 lengths={n:len(s) for n,s in sequences.items()};del sequences
 blocks={};key=None;short=0;total=0;selfs=set();started=time.monotonic()
 with raw.open('rb') as stream:
  while True:
   offset=stream.tell();line=stream.readline()
   if not line:break
   fields=line.decode().split();assert fields
   if fields[0]=='>':
    if key is not None:blocks[key][1]=offset-blocks[key][0]
    assert len(fields)==4 and fields[2]=='and';key=(fields[1],fields[3]);assert key not in blocks
    blocks[key]=[offset,0,0]
   else:
    assert key is not None and len(fields)==3
    a,b,n=map(int,fields);assert min(a,b)>=0 and n>0 and a+n<=lengths[key[0]] and b+n<=lengths[key[1]]
    if n<20:
     assert key[0]==key[1] and a==b==0 and n==lengths[key[0]];short+=1
    if key[0]==key[1] and a==b==0 and n==lengths[key[0]]:selfs.add(key[0])
    blocks[key][2]+=1;total+=1
    if total%5000000==0:print({'stage':'validate_hsp','records':total,'seconds':time.monotonic()-started},flush=True)
  if key is not None:blocks[key][1]=stream.tell()-blocks[key][0]
 assert blocks and all(v[2]>0 for v in blocks.values()) and selfs==set(lengths)
 tmp=canonical.with_suffix('.tmp')
 with raw.open('rb') as source,tmp.open('wb') as target:
  for key,(offset,size,count) in sorted(blocks.items()):
   source.seek(offset)
   while size:
    b=source.read(min(size,8*1024**2));assert b;target.write(b);size-=len(b)
 os.replace(tmp,canonical)
 atomic(marker,{'at_utc':now(),'input_sha256':contract,'raw_file':record(raw),'file':record(canonical),
  'census':{'protein_pair_blocks':len(blocks),'hsp_records':total,'proteins_with_full_self_hsp':len(lengths),
   'native_short_full_self_hsps_preserved':short,'within_block_order_preserved':True},'script':record(__file__)})
 print({'stage':'canonical_hsp_complete','seconds':time.monotonic()-started},flush=True)
def main():
 qpath=ROOT/'qualification/sprint-requested/qualification.json';q=json.loads(qpath.read_text());assert q['passed']
 binary=ROOT/'bin/sprint_predict_sparse_cpu';expected=next(r for r in q['binaries'] if r['path']==str(binary));assert record(binary)==expected
 for r in q['sources']:assert record(r['path'])==r
 contract=json.loads((ROOT/'provenance/sprint-input.json').read_text())
 for path,h in contract['files'].items():assert record(ROOT/path)['sha256']==h
 canonicalize()
 marker=OUT/'prediction-command-succeeded';output=OUT/'scores.txt'
 if marker.exists():
  m=json.loads((OUT/'execution.json').read_text());assert record(output)==m['output'];return
 for suffix in ['', '.pos','.neg']:
  p=Path(str(output)+suffix)
  if p.exists():p.rename(ROOT/'logs'/('sprint-incomplete-'+str(time.time_ns())+suffix))
 data=ROOT/'data/sprint'
 cmd=[str(binary),'-p',str(data/'proteins.fasta'),'-h',str(OUT/'canonical.hsp'),'-tr',str(data/'train-positive.txt'),
  '-pos',str(data/'pairs.txt'),'-neg',str(data/'empty.txt'),'-o',str(output),'-Thc','40']
 started=time.monotonic()
 with (ROOT/'logs/sprint-predict.log').open('w') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
 expected_rows=sum(1 for _ in (data/'pairs.txt').open());rows=0
 import math
 for line in output.open():
  score,label=line.split();assert label=='1' and math.isfinite(float(score)) and float(score)>=0;rows+=1
 assert rows==expected_rows
 atomic(OUT/'execution.json',{'at_utc':now(),'mode':'Native serial SPRINT arithmetic, qualified sparse requested-pair accumulation',
  'command':cmd,'binary':record(binary),'qualification':record(qpath),'input_contract':record(ROOT/'provenance/sprint-input.json'),
  'output':record(output),'rows':rows,'seconds':time.monotonic()-started,'hostname':socket.gethostname(),'script':record(__file__)})
 marker.write_text(now()+'\n');print({'stage':'sprint_prediction_complete','rows':rows,'seconds':time.monotonic()-started},flush=True)
if __name__=='__main__':main()
