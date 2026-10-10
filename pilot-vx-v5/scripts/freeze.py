"""Freeze one qualified pilot, before fitting any new outcome head."""
import ast
import subprocess
import sys
from pathlib import Path
from study import ROOT, PROJECT, config, read, sha, atomic, now, check_budget


def main():
    if (ROOT / 'provenance/freeze.json').exists() or (ROOT / 'results/fit-start.json').exists():
        raise RuntimeError('Study already frozen or outcome fitting started')
    cfg = config(); qualification = read(ROOT / 'qualification/real.json')
    if not qualification['passed'] or qualification['labels_used']:
        raise ValueError('Missing label-free qualification')
    if qualification['conservative_total_seconds'] > check_budget(cfg['budgets']['shutdown_reserve_seconds']):
        raise TimeoutError('Conservative workload no longer fits the frozen budget')
    for rel, digest in qualification['code_sha256'].items():
        if sha(ROOT / rel) != digest:
            raise ValueError('Qualified code changed: ' + rel)
    for p in (ROOT / 'scripts').glob('*.py'):
        ast.parse(p.read_text())
    with (ROOT / 'qualification/unit.log').open('w') as log:
        result = subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'],
                                stdout=log, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError('Qualification tests failed; see qualification/unit.log')
    atomic(ROOT / 'qualification/unit.json', {'at_utc': now(), 'passed': True,
        'tests_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'tests').glob('*.py'))},
        'output_sha256': sha(ROOT / 'qualification/unit.log')})
    inputs = read(ROOT / 'provenance/inputs.json')
    pairs = {p['uid']: p for p in read(ROOT / 'data/pairs.json')}
    for case in qualification['cases']:
        p = pairs[case['uid']]
        if any(case[k] != p[k] for k in ('length', 'paired_depth', 'depth_256', 'T256_tokens_sha256', 'S256_tokens_sha256')):
            raise ValueError('Prepared input differs from GPU qualification: ' + case['uid'])
    for rel, digest in inputs['source_files'].items():
        if sha(PROJECT / rel) != digest:
            raise ValueError('Source changed: ' + rel)
    for rel,digest in read(ROOT / 'provenance/preparation-start.json')['preparation_code_sha256'].items():
        if sha(ROOT / rel) != digest: raise ValueError('Preparation code changed: '+rel)
    data = sorted(p for p in (ROOT / 'data').rglob('*') if p.is_file())
    paths = [ROOT / 'README.md', ROOT / 'config.json', *sorted((ROOT / 'scripts').glob('*.py')),
        *sorted((ROOT / 'tests').glob('*.py')), *data, ROOT / 'qualification/real.json', ROOT / 'qualification/unit.json',
        ROOT / 'provenance/inputs.json', ROOT / 'provenance/source-artifact-digests.json',
        ROOT / 'provenance/proposal-at-start.md', ROOT / 'provenance/resource-start.json',
        ROOT / 'provenance/preparation-start.json', ROOT / 'provenance/preflight-start.json']
    files = {str(p.relative_to(PROJECT)): sha(p) for p in paths}; files.update(inputs['source_files'])
    # A TRAIN/DEV allowlist is used instead of traversing the older manifests' TEST exclusion inputs.
    if any('test-sequence' in rel or '/test.' in rel for rel in files):
        raise ValueError('Unexpected TEST path in current freeze')
    atomic(ROOT / 'provenance/freeze.json', {'at_utc': now(), 'files': files, 'image': inputs['image'],
        'image_sha256': inputs['image_sha256'], 'test_accessed': False, 'outcome_heads_fitted': False,
        'production_authorized': False, 'resource_contract': cfg['budgets']})
    print({'frozen': True, 'fingerprint': sha(ROOT / 'provenance/freeze.json'), 'bound_files': len(files)}, flush=True)


if __name__ == '__main__':
    main()
