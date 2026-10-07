"""Two independent CPU searches using the inherited baseline settings."""
import argparse, hashlib, json, os, shutil, subprocess, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser(); parser.add_argument('reference', choices=['v5', 'bernett']); args = parser.parse_args()
    start = time.monotonic(); ref = args.reference
    frozen = json.loads((ROOT / 'provenance/nonhuman/protocol-freeze.json').read_text())['identity']
    assert sha(ROOT / 'NONHUMAN-ADDENDUM.md') == frozen['addendum_sha256']
    cfg = frozen['config']['search']; threads = frozen['config']['threads']
    binary = shutil.which('mmseqs'); assert binary and sha(binary) == 'bbe29a5b59b268d13f1237a8a8dd47c7d2a0d74103aab474c6d532e885ea58ac'
    query = ROOT / 'data/nonhuman/queries.fasta'; target = ROOT / 'data/nonhuman' / (ref + '-train.fasta')
    output = ROOT / 'work/nonhuman' / ('alignments-' + ref + '.tsv'); partial = output.with_suffix('.partial')
    command = [binary, 'easy-search', str(query), str(target), str(partial), str(ROOT / 'work/nonhuman' / ('mmseqs-' + ref)),
               '--threads', str(threads), '--gpu', '0', '-s', str(cfg['sensitivity']), '-e', str(cfg['evalue']),
               '--min-seq-id', str(cfg['min_identity']), '-c', str(cfg['min_coverage']), '--cov-mode', '0',
               '--max-seqs', str(cfg['max_prefilter']), '--alignment-mode', '3', '--num-iterations', '1',
               '--split-memory-limit', '16G', '--format-output',
               'query,target,fident,bits,evalue,qcov,tcov,qlen,tlen,nident,alnlen,qstart,qend,tstart,tend',
               '--remove-tmp-files', '1']
    identity = {'query_sha256': sha(query), 'target_sha256': sha(target), 'binary_sha256': sha(binary),
                'protocol_sha256': frozen['addendum_sha256'], 'script_sha256': sha(__file__), 'command': command}
    marker = ROOT / 'provenance/nonhuman' / ('search-' + ref + '.json')
    if marker.exists():
        old = json.loads(marker.read_text()); assert old['identity'] == identity and sha(output) == old['output_sha256']
        print('Verified cached nonhuman search', ref, flush=True); return
    print('CPU search', ref, flush=True)
    with (ROOT / 'logs/nonhuman' / ('search-' + ref + '.log')).open('w') as log:
        subprocess.run(['/usr/bin/time', '-v', '-o', str(ROOT / 'logs/nonhuman' / ('search-' + ref + '.resources.txt')), *command],
                       check=True, stdout=log, stderr=subprocess.STDOUT)
    assert partial.exists() and partial.stat().st_size > 0; os.replace(partial, output)
    result = {'identity': identity, 'output_sha256': sha(output), 'output_path': str(output),
              'wall_seconds': time.monotonic()-start, 'module': 'mmseqs2-gpu/18-8cc5c (CPU)',
              'reported_version': subprocess.check_output([binary, 'version'], text=True).strip(), 'gpu_used': False}
    temp = marker.with_suffix('.partial'); temp.write_text(json.dumps(result, indent=2)+'\n'); os.replace(temp, marker)
    print('Search complete', ref, 'seconds', result['wall_seconds'], flush=True)

if __name__ == '__main__': main()
