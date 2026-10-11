"""Fetch pinned public sequence/model artifacts; no PPI scores or structures."""
import argparse
import hashlib
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def fetch(url, path, expected=None, size=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        digest = sha(path)
        if expected and digest != expected:
            raise ValueError(f'Existing file hash mismatch: {path}')
        if size is not None and path.stat().st_size != size:
            raise ValueError(f'Existing file size mismatch: {path}')
        return digest
    partial = path.with_name(path.name + '.partial')
    tls = []
    if url.startswith('https://conglab.swmed.edu/'):
        import ssl
        # The author server omits intermediates. Their chain was verified against system roots;
        # hostname and certificate verification remain enabled.
        chain = ROOT/'provenance/mirror-untrusted-chain.pem'
        subprocess.run(['openssl','verify','-untrusted',str(chain),str(ROOT/'provenance/mirror-leaf.pem')],check=True)
        bundle=ROOT/'provenance/mirror-ca-bundle.pem'
        roots=Path(ssl.get_default_verify_paths().cafile)
        bundle.write_bytes(roots.read_bytes()+b'\n'+chain.read_bytes())
        tls=['--cacert',str(bundle)]
    subprocess.run(['curl', *tls, '--silent', '--show-error', '--fail', '--location', '--retry', '5', '--retry-delay', '10',
                    '--connect-timeout', '30', '--max-time', '14400', '--continue-at', '-',
                    '--output', str(partial), url], check=True)
    digest = sha(partial)
    if expected and digest != expected:
        raise ValueError(f'Download hash mismatch: {path}')
    if size is not None and partial.stat().st_size != size:
        raise ValueError(f'Download size mismatch: {path}')
    partial.replace(path)
    return digest


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('kind', choices=['archive', 'encoder']); args = ap.parse_args()
    cfg = json.loads((ROOT/'config.json').read_text()); records = []
    if args.kind == 'archive':
        a = cfg['archive']
        # Published author mirror: same bytes must match the pinned Dryad digest and size.
        url = 'https://conglab.swmed.edu/humanPPI/downloads/protein_omicMSAs.tar.gz'
        items = [(url, ROOT/'data'/a['filename'], a['sha256'], a['bytes'])]
    else:
        e = cfg['encoder']; dest = PROJECT/'images/msa-pairformer'
        items = [(f"https://api.github.com/repos/yoakiyama/MSA_Pairformer/tarball/{e['commit']}",
                  dest/'sources/source.tar.gz', None, None)]
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{e['weights_repository']}/revision/{e['weights_revision']}?blobs=true", timeout=60) as f:
            meta = json.load(f)
        assert meta['sha'] == e['weights_revision']
        (ROOT/'provenance').mkdir(exist_ok=True)
        (ROOT/'provenance/encoder-remote.json').write_text(json.dumps(meta, indent=2)+'\n')
        for name in ['model_cuex.bin', 'contact.bin', 'confind_contact.bin', 'LICENSE', 'config.json', 'README.md']:
            entry = next(x for x in meta['siblings'] if x['rfilename'] == name)
            items.append((f"https://huggingface.co/{e['weights_repository']}/resolve/{e['weights_revision']}/{name}",
                          dest/'assets'/name, entry.get('lfs', {}).get('sha256'), entry.get('size')))
    for url, path, expected, size in items:
        digest = fetch(url, path, expected, size)
        records.append({'url': url, 'path': str(path.relative_to(PROJECT)), 'sha256': digest, 'bytes': path.stat().st_size})
        print(json.dumps(records[-1]), flush=True)
    (ROOT/'provenance').mkdir(exist_ok=True)
    out = ROOT/'provenance'/f'{args.kind}-download.json'
    out.write_text(json.dumps({'completed_unix': time.time(), 'files': records}, indent=2)+'\n')


if __name__ == '__main__':
    main()
