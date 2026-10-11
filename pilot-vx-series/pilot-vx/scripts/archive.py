"""Stream the verified archive; retain bounded TRAIN/DEV monomer caches only."""
import gzip
import json
import os
import tarfile
import time
from collections import Counter
from pathlib import Path
from common import ROOT, PROJECT, atomic, config, now, sha, text_hash
from msa import records, cached_monomer


def main():
    cfg=config(); path=ROOT/'data'/cfg['archive']['filename']
    if path.stat().st_size!=cfg['archive']['bytes'] or sha(path)!=cfg['archive']['sha256']:
        raise ValueError('Archive checksum/size mismatch; no parsing authorized')
    proteins=json.loads((ROOT/'data/proteins.json').read_text()); sample=json.loads((ROOT/'data/sample.json').read_text())
    targets=set(json.loads((ROOT/'data/coverage-proteins.json').read_text()))
    targets.update(p for r in sample for p in [r['a'],r['b']])
    by_sequence={proteins[str(i)]['sequence']:str(i) for i in sorted(targets)}
    forbidden={'train':set(json.loads((ROOT/'data/test-sequence-hashes.json').read_text())), 'val':set()}
    for p in proteins.values(): forbidden['val' if p['split']=='train' else 'train'].add(p['sha256'])
    out=ROOT/'data/monomers';out.mkdir(exist_ok=True);catalog={};counts=Counter();begin=time.monotonic()
    resource=json.loads((ROOT/'provenance/resource-start.json').read_text())
    (ROOT/'results/archive-incomplete.json').write_text(json.dumps({'started':now(),'targets':len(targets)}))
    with tarfile.open(path,'r|gz') as tf:
        for member in tf:
            if time.monotonic()-begin>6*3600 or time.time()>resource['qualification_deadline_unix']:
                raise TimeoutError('Preprocessing/qualification resource window exhausted')
            if not member.isfile() or not member.name.endswith(('.a3m','.a3m.gz')):continue
            if member.name.endswith('.gz'):raise ValueError('Nested compressed member requires explicit format qualification')
            counts['members']+=1
            with tf.extractfile(member) as fh:
                # tarfile's streaming _Stream lacks seekable(); TextIOWrapper probes it.
                # Decode binary lines directly so the reader never asks for seeking.
                iterator=records(line.decode('ascii',errors='strict') for line in fh)
                first=next(iterator);second=next(iterator)
                if first[0].split()[0]!='mask' or second[0].split()[0]!='query':
                    raise ValueError(f'Unexpected archive mask/query ordering: {member.name}')
                quality,query=first[1],second[1]
                if len(quality)!=len(query):raise ValueError('Query/mask length mismatch')
                i=by_sequence.get(query)
                if i is None:continue
                counts['exact_matching_members']+=1
                if i in catalog and catalog[i]['member']<=member.name:continue
                m=cached_monomer(iterator,query,quality,forbidden[proteins[i]['split']],cfg['maximum_cached_homologs_per_protein'])
                dest=out/f'{i}.json.gz';tmp=dest.with_name(dest.name+'.partial')
                with tmp.open('wb') as raw:
                    with gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz:
                        gz.write(json.dumps(m,separators=(',',':')).encode())
                    raw.flush();os.fsync(raw.fileno())
                tmp.replace(dest)
                catalog[i]={'member':member.name,'path':str(dest.relative_to(ROOT)),'sha256':sha(dest),
                            'query_sha256':m['query_sha256'],'stats':m['stats'],'quality_fraction':quality.count('*')/len(quality)}
            if counts['members']%100==0:
                atomic(ROOT/'results/archive-progress.json',{'at_utc':now(),'counts':dict(counts),'matched_proteins':len(catalog),'seconds':time.monotonic()-begin})
                print(json.dumps({'members':counts['members'],'matched':len(catalog)}),flush=True)
                usage=sum(p.stat().st_size for d in [ROOT,PROJECT/'images/msa-pairformer'] for p in d.rglob('*') if p.is_file() and not p.is_symlink())
                if usage>cfg['budgets']['additional_bytes']:raise RuntimeError('Additional storage budget exceeded')
    atomic(ROOT/'data/monomer-catalog.json',catalog)
    audit_ids=json.loads((ROOT/'data/coverage-proteins.json').read_text())
    audit={}
    for split in ['train','val']:
        subset=[i for i in audit_ids if proteins[str(i)]['split']==split]
        audit[split]={'total':len(subset),'exact_matches':sum(str(i) in catalog for i in subset)}
    atomic(ROOT/'results/archive-coverage.json',{'complete':True,'at_utc':now(),'archive_sha256':cfg['archive']['sha256'],
           'counts':dict(counts),'target_proteins':len(targets),'matched_target_proteins':len(catalog),
           'coverage_audit':audit,'unmatched_ids':sorted(targets-set(map(int,catalog))),'seconds':time.monotonic()-begin})
    (ROOT/'results/archive-incomplete.json').unlink()
    print(json.dumps({'complete':True,'targets':len(targets),'matched':len(catalog),'coverage':audit}),flush=True)


if __name__=='__main__':main()
