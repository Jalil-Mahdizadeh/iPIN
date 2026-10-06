"""Fetch immutable snapshots; verify Figshare's published digest before use."""
import concurrent.futures
import hashlib
import json
import os
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '8bf9adb183b702c88db62ff878de482e2f20e4b2'
SOURCES = [
    ('hippie-v3-snapshot.txt.gz', 'https://cbdm-01.zdv.uni-mainz.de/hippienew/downloads/HIPPIE-current.txt.gz', None, None),
    ('bernett-2026-data-v1.zip', 'https://ndownloader.figshare.com/files/68110393', 190914297, '37f2bde59827ec571265cf5f91c3774d'),
    ('author-pipeline.zip', f'https://codeload.github.com/bionetslab/ppi-splitting-pipeline/zip/{COMMIT}', None, None),
]

def digest(path, algorithm='sha256'):
    h = hashlib.new(algorithm)
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

def fetch(item):
    name, url, expected_size, expected_md5 = item
    path = ROOT/'sources'/name
    receipt = ROOT/'provenance'/f'{name}.json'
    if path.exists():
        old = json.loads(receipt.read_text())
        assert digest(path) == old['sha256'], f'Existing source changed: {name}'
        return old
    part = path.with_name(path.name + '.part')
    for attempt in range(3):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'iPIN-v5-reproducibility/1.0'})
            with urllib.request.urlopen(request, timeout=90) as response, part.open('wb') as out:
                headers = {k: response.headers.get(k) for k in ['Content-Length', 'Content-Type', 'Last-Modified', 'ETag']}
                for block in iter(lambda: response.read(4 * 1024**2), b''):
                    out.write(block)
                out.flush()
                os.fsync(out.fileno())
            if expected_size is not None:
                assert part.stat().st_size == expected_size, (name, part.stat().st_size, expected_size)
            md5 = digest(part, 'md5')
            if expected_md5 is not None:
                assert md5 == expected_md5, (name, md5, expected_md5)
            result = dict(name=name, url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(),
                          bytes=part.stat().st_size, sha256=digest(part), md5=md5,
                          expected_md5=expected_md5, expected_bytes=expected_size, headers=headers)
            part.replace(path)
            receipt.write_text(json.dumps(result, indent=2)+'\n')
            print(json.dumps(result), flush=True)
            return result
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2 * (attempt+1))

if __name__ == '__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, SOURCES))
    (ROOT/'provenance/source-downloads.json').write_text(json.dumps(results, indent=2)+'\n')
