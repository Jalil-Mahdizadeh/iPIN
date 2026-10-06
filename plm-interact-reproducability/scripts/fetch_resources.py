"""Download only pinned released checkpoints, test sets and metadata. No training."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parents[1]
SELECTION = {
    'cross_species_benchmarking': lambda p: p.startswith('test/') or p == 'README.md',
    'Bernett_benchmarking': lambda p: p in ('pairs_uniprot_seqs_test.csv', 'README.md'),
    'Mutation_effect_dataset': lambda p: p in ('test_mutation_data.csv', 'README.md'),
    'PLM-interact-650M-Leakage-Free-Dataset': lambda p: p != '.gitattributes',
    'PLM-interact-650M-Mutation': lambda p: p != '.gitattributes',
}

def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()

def fetch(task):
    name, revision, entry = task
    is_model = name.startswith('PLM-')
    prefix = '' if is_model else 'datasets/'
    relative = entry['path']
    url = f'https://huggingface.co/{prefix}danliu1226/{name}/resolve/{revision}/{relative}'
    target = ROOT / ('checkpoints' if is_model else 'data') / name / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    expected = entry.get('lfs', {}).get('oid')
    if not target.exists():
        partial = target.with_suffix(target.suffix + '.partial')
        for attempt in range(3):
            try:
                start = time.monotonic()
                response = requests.get(url, stream=True, timeout=(30, 120))
                response.raise_for_status()
                with partial.open('wb') as handle:
                    for block in response.iter_content(8 << 20):
                        handle.write(block)
                response.close()
                if partial.stat().st_size != entry['size']:
                    raise RuntimeError(f'Wrong download size: {partial}')
                digest = sha256(partial)
                if expected and digest != expected:
                    raise RuntimeError(f'Wrong download hash: {partial}')
                partial.replace(target)
                print(f'Downloaded {name}/{relative}: {entry["size"]:,} bytes in {time.monotonic()-start:.1f}s', flush=True)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    digest = sha256(target)
    if target.stat().st_size != entry['size'] or (expected and digest != expected):
        raise RuntimeError(f'Existing file disagrees with pinned release: {target}')
    return {'repository': f'danliu1226/{name}', 'revision': revision,
            'source_path': relative, 'local_path': str(target.relative_to(ROOT)),
            'url': url, 'bytes': target.stat().st_size, 'sha256': digest,
            'publisher_lfs_sha256': expected}

def main():
    tasks = []
    for name, include in SELECTION.items():
        meta = json.loads((ROOT / 'provenance' / f'hf-{name}.json').read_text())
        entries = json.loads((ROOT / 'provenance' / f'hf-tree-{name}.json').read_text())
        tasks.extend((name, meta['sha'], entry) for entry in entries
                     if entry['type'] == 'file' and include(entry['path']))
    records = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        for future in as_completed([pool.submit(fetch, task) for task in tasks]):
            records.append(future.result())
            manifest = {'retrieved_utc': datetime.now(timezone.utc).isoformat(),
                        'no_retraining': True, 'files': sorted(records, key=lambda x: x['local_path'])}
            (ROOT / 'provenance' / 'downloads.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'All {len(records)} resources downloaded and verified.', flush=True)

if __name__ == '__main__':
    main()
