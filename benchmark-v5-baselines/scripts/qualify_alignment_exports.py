"""Qualify integer diagnostics without changing any frozen primary alignment field."""
import argparse,csv,hashlib,json,math,os,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('reference',choices=['v5','bernett']);args=p.parse_args();ref=args.reference
    primary=json.loads((ROOT/'provenance/nonhuman'/('search-'+ref+'.json')).read_text())
    source=Path(primary['output_path']);assert sha(source)==primary['output_sha256']
    marker=ROOT/'provenance/nonhuman'/('alignment-precision-'+ref+'.json')
    output=ROOT/'work/nonhuman'/('backtrace-'+ref+'.tsv')
    if marker.exists():
        old=json.loads(marker.read_text());assert old['primary_sha256']==sha(source) and old['backtrace_sha256']==sha(output)
        print('Verified alignment-export qualification',ref);return
    command=list(primary['identity']['command']);command[4]=str(output.with_suffix('.partial'))
    command[5]=str(ROOT/'work/nonhuman'/('backtrace-tmp-'+ref));command+=['-a','1']
    start=time.monotonic()
    with (ROOT/'logs/nonhuman'/('backtrace-'+ref+'.log')).open('w') as log:
        subprocess.run(['/usr/bin/time','-v','-o',str(ROOT/'logs/nonhuman'/('backtrace-'+ref+'.resources.txt')),*command],
                       check=True,stdout=log,stderr=subprocess.STDOUT)
    os.replace(output.with_suffix('.partial'),output)
    def rows(path):
        with path.open() as f:
            for row in csv.reader(f,delimiter='\t'):yield row
    original=list(rows(source));precise=list(rows(output));assert len(original)==len(precise)
    old={tuple(r[:9]):r for r in original};new={tuple(r[:9]):r for r in precise}
    assert len(old)==len(original) and old.keys()==new.keys(), 'Backtrace changed a primary field'
    maximum=0.;identity_below=0;coverage_below=0
    for r in precise:
        ni,al,qs,qe,ts,te=map(int,r[9:]);ql,tl=map(int,r[7:9]);assert ni>0
        identity_below+=4*ni<al;coverage_below+=2*(qe-qs+1)<ql or 2*(te-ts+1)<tl
        exact=ni/al*math.sqrt((qe-qs+1)/ql*(te-ts+1)/tl)
        rounded=float(r[2])*math.sqrt(float(r[5])*float(r[6]));maximum=max(maximum,abs(exact-rounded))
    result={'reference':ref,'primary_sha256':sha(source),'backtrace_sha256':sha(output),'command':command,
       'primary_first_nine_fields_identical_for_every_alignment':True,'rows':len(precise),
       'integer_below_requested_identity':identity_below,'integer_below_requested_bilateral_coverage':coverage_below,
       'maximum_absolute_weight_rounding_difference':maximum,'wall_seconds':time.monotonic()-start,
       'correction':'The initial search omitted backtraces, so its nident export was zero (unavailable). '
       'The raw integer-identity diagnostic fields in predictions-frozen.json are superseded by this qualification. '
       'All nine primary fields are identical; no retained hit, baseline weight, fitted model or prediction changed.'}
    temp=marker.with_suffix('.partial');temp.write_text(json.dumps(result,indent=2)+'\n');os.replace(temp,marker)
    print('Integer audit qualified',ref,result,flush=True)
if __name__=='__main__':main()
