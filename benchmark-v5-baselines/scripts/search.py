"""Run the pinned MMseqs2 binary on CPU; restart only incomplete searches."""
from pathlib import Path
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]

def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as inp:
        for b in iter(lambda: inp.read(8*1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def main():
    start = time.monotonic()
    frozen = json.loads((ROOT / 'provenance/protocol-freeze.json').read_text())
    assert sha(ROOT / 'PROTOCOL.md') == frozen['protocol_sha256']
    cfg = frozen['config']['search']
    binary = shutil.which('mmseqs')
    assert binary, 'Load the recorded MMseqs2 module first.'
    # Same installed executable used and recorded by the preceding data-preparation audit.
    assert sha(binary) == 'bbe29a5b59b268d13f1237a8a8dd47c7d2a0d74103aab474c6d532e885ea58ac'
    query, target = ROOT / 'data/queries.fasta', ROOT / 'data/train.fasta'
    output = ROOT / 'work/alignments.tsv'
    partial = output.with_suffix('.tsv.partial')
    command = [binary, 'easy-search', str(query), str(target), str(partial), str(ROOT / 'work/mmseqs'),
               '--threads', str(frozen['config']['threads']), '--gpu', '0', '-s', str(cfg['sensitivity']),
               '-e', str(cfg['evalue']), '--min-seq-id', str(cfg['min_identity']),
               '-c', str(cfg['min_coverage']), '--cov-mode', '0', '--max-seqs', str(cfg['max_prefilter']),
               '--alignment-mode', '3', '--num-iterations', '1', '--split-memory-limit', '16G',
               '--format-output', 'query,target,fident,bits,evalue,qcov,tcov,qlen,tlen', '--remove-tmp-files', '1']
    identity = {'query_sha256': sha(query), 'target_sha256': sha(target), 'binary_sha256': sha(binary),
                'protocol_sha256': frozen['protocol_sha256'], 'command': command}
    marker = ROOT / 'provenance/search.json'
    if marker.exists():
        old = json.loads(marker.read_text())
        assert old['identity'] == identity and old['output_sha256'] == sha(output)
        print('Search already complete; input, binary, and output hashes verified.', flush=True)
        return
    print(json.dumps({'search_command': command, 'gpu': False}), flush=True)
    subprocess.run(['/usr/bin/time', '-v', '-o', str(ROOT / 'logs/search.resources.txt'), *command], check=True)
    assert partial.is_file() and partial.stat().st_size > 0
    os.replace(partial, output)
    version = subprocess.check_output([binary, 'version'], text=True).strip()
    result = {'identity': identity, 'version_reported': version, 'module': 'mmseqs2-gpu/18-8cc5c (CPU mode)',
              'output_sha256': sha(output), 'wall_seconds': time.monotonic()-start,
              'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    marker.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'identity'}), flush=True)

if __name__ == '__main__':
    main()
