"""Fetch immutable public human-PPI inputs and original ESM-2 initialization."""
import hashlib
import json
import os
import time
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parents[1]
BASE_REV = '08e4846e537177426273712802403f7ba8261b6c'
DATA_REV = '5d2ad03baa165c27df32a2eadf066462a2a83073'

def digest(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(16*1024**2), b''): h.update(b)
    return h.hexdigest()

def get_json(url):
    r = requests.get(url, timeout=60); r.raise_for_status(); return r.json()

def main():
    (ROOT/'provenance').mkdir(parents=True, exist_ok=True)
    records = []
    specs = [
        ('models', 'facebook/esm2_t33_650M_UR50D', BASE_REV, ROOT/'assets/esm2',
         ['config.json','model.safetensors','special_tokens_map.json','tokenizer_config.json','vocab.txt','README.md']),
        ('datasets', 'danliu1226/Bernett_benchmarking', DATA_REV, ROOT/'data/raw',
         ['pairs_uniprot_seqs_train.csv','pairs_uniprot_seqs_val.csv','pairs_uniprot_seqs_test.csv','README.md'])]
    for kind, repo, rev, dest, names in specs:
        dest.mkdir(parents=True, exist_ok=True)
        tree = get_json(f'https://huggingface.co/api/{kind}/{repo}/tree/{rev}?recursive=true')
        (ROOT/'provenance'/f'{kind}-tree.json').write_text(json.dumps(tree,indent=2)+'\n')
        meta = {x['path']:x for x in tree}
        for name in names:
            m=meta[name]; target=dest/name
            url=f'https://huggingface.co/{"datasets/" if kind=="datasets" else ""}{repo}/resolve/{rev}/{name}'
            expected=m.get('lfs',{}).get('oid')
            if not (target.exists() and target.stat().st_size==m['size'] and (not expected or digest(target)==expected)):
                partial=target.with_suffix(target.suffix+'.partial')
                for attempt in range(4):
                    try:
                        with requests.get(url,params={'download':'true','cachebust':str(time.time_ns())},stream=True,timeout=(30,180)) as r:
                            r.raise_for_status()
                            with open(partial,'wb') as f:
                                for b in r.iter_content(8*1024**2):
                                    if b:f.write(b)
                                f.flush(); os.fsync(f.fileno())
                        assert partial.stat().st_size==m['size'],(name,partial.stat().st_size,m['size'])
                        assert not expected or digest(partial)==expected, name
                        os.replace(partial,target)
                        break
                    except Exception:
                        if attempt==3:raise
                        time.sleep(2**attempt)
            record={'repository':repo,'revision':rev,'file':name,'local_path':str(target.relative_to(ROOT)),
                    'url':url,'bytes':target.stat().st_size,'sha256':digest(target),'publisher_lfs_sha256':expected}
            records.append(record)
            (ROOT/'provenance/downloads.json').write_text(json.dumps(records,indent=2)+'\n')
            print(json.dumps(record),flush=True)

if __name__=='__main__':main()
