"""Fetch only cross-species identity/masking CSV members from publisher ZIP."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import struct
import zlib
import requests

ROOT=Path(__file__).resolve().parents[1]
LIT=ROOT.parent/'literature/plm-interact'
URL=next(x['url'] for x in json.loads((LIT/'sources.json').read_text())['sources'] if x['file']=='source-data-index.json')
entries=[e for e in json.loads((LIT/'source-data-index.json').read_text()) if e['path'].endswith('.csv') and e['path'].startswith(('Source Data/Supplementary Figure7/','Source Data/Supplementary Figure2/'))]

def byte_range(start,end):
    response=requests.get(URL+f'?range={start}-{end}',headers={'Range':f'bytes={start}-{end}'},timeout=120)
    response.raise_for_status()
    assert response.status_code==206 and len(response.content)==end-start+1
    assert response.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
    return response.content

def get(entry):
    path=ROOT/'data/published-extra'/Path(entry['path']).parent.name/Path(entry['path']).name
    path.parent.mkdir(parents=True,exist_ok=True)
    if not path.exists():
        offset=entry['header_offset']
        h=struct.unpack('<IHHHHHIIIHH',byte_range(offset,offset+29))
        assert h[0]==0x04034b50 and h[3]==8
        start=offset+30+h[-2]+h[-1]
        raw=zlib.decompress(byte_range(start,start+entry['compressed_bytes']-1),-15)
        assert len(raw)==entry['bytes']
        if h[6]:assert (zlib.crc32(raw)&0xffffffff)==h[6]
        path.write_bytes(raw)
    assert path.stat().st_size==entry['bytes']
    item=dict(**entry,local_path=str(path.relative_to(ROOT)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),archive_url=URL)
    print('Verified',path.name,flush=True)
    return item

with ThreadPoolExecutor(max_workers=3) as pool:records=list(pool.map(get,entries))
(ROOT/'provenance/publisher-extra-downloads.json').write_text(json.dumps(records,indent=2)+'\n')
