"""Resumable, release-checked UniProt JSON and isoform FASTA snapshots."""
import concurrent.futures
import gzip
import hashlib
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime,timezone
from common import ROOT, read_json, write_json, atomic_bytes, sha, mark

RELEASE='2026_03'
BATCH_SIZE=30  # Validated with both primary and secondary-accession clauses.

def fetch_one(item):
    idx,accessions,fmt=item
    key=hashlib.sha256('\n'.join(accessions).encode()).hexdigest()[:20]
    path=ROOT/'sources/uniprot'/f'query-{key}.{fmt}.gz'
    receipt=path.with_suffix(path.suffix+'.receipt.json')
    if path.exists():
        r=read_json(receipt)
        assert r['accessions']==accessions and r['sha256']==sha(path) and r['release']==RELEASE
        return r
    query='('+' OR '.join(f'(accession:{p} OR sec_acc:{p})' for p in accessions)+')'
    params={'query':query,'format':fmt}
    if fmt=='fasta':params['includeIsoform']='true'
    url='https://rest.uniprot.org/uniprotkb/stream?'+urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            req=urllib.request.Request(url,headers={'User-Agent':'iPIN-v5-data-preparation/1.0'})
            with urllib.request.urlopen(req,timeout=120) as response:
                data=response.read()
                release=response.headers.get('X-UniProt-Release')
                assert release==RELEASE,(release,RELEASE)
                release_date=response.headers.get('X-UniProt-Release-Date')
            if fmt=='json':
                obj=json.loads(data);n=len(obj['results'])
            else:
                assert not data or data.startswith(b'>'),data[:80]
                n=sum(line.startswith(b'>') for line in data.splitlines())
            atomic_bytes(path,gzip.compress(data,mtime=0))
            r=dict(path=str(path.relative_to(ROOT)),accessions=accessions,query=query,format=fmt,include_isoforms=fmt=='fasta',release=release,release_date=release_date,
                   retrieved_utc=datetime.now(timezone.utc).isoformat(),records=n,sha256=sha(path),bytes=path.stat().st_size)
            write_json(receipt,r)
            print(f'batch={idx} format={fmt} records={n}',flush=True)
            return r
        except urllib.error.HTTPError as exc:
            if exc.code in (400,401,403,404) or attempt==4:
                raise RuntimeError(f'UniProt HTTP {exc.code}: {exc.read().decode()[:1000]}') from exc
            time.sleep(min(2**attempt,15))
        except Exception:
            if attempt==4:raise
            time.sleep(min(2**attempt,15))

def main():
    ids=read_json(ROOT/'work/requested-accessions.json')['base_accessions']
    tasks=[(i//BATCH_SIZE,ids[i:i+BATCH_SIZE],fmt) for i in range(0,len(ids),BATCH_SIZE) for fmt in ['json','fasta']]
    mark('uniprot','running',base_accessions=len(ids),requests=len(tasks),release=RELEASE)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(fetch_one,tasks))
    write_json(ROOT/'provenance/uniprot-downloads.json',results)
    mark('uniprot','complete',base_accessions=len(ids),requests=len(tasks),release=RELEASE)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        mark('uniprot','failed',error=str(exc))
        raise
