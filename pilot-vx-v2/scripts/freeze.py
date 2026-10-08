"""Freeze the single qualified ablation before fitting any new outcome head."""
import ast
import heapq
import subprocess
import sys
import time
from study import ROOT, BASE, PROJECT, config, read, sha, atomic, now


def main():
    if (ROOT / 'provenance/freeze.json').exists():
        raise RuntimeError('Already frozen; no overwrite')
    cfg = config(); resource = read(ROOT / 'provenance/resource-start.json')
    if time.time() > resource['qualification_deadline_unix']:
        raise TimeoutError('Qualification reserve exhausted')
    for p in (ROOT / 'scripts').glob('*.py'): ast.parse(p.read_text())
    subprocess.run(['bash', '-n', str(ROOT / 'slurm/pilot.sbatch')], check=True)
    subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'], check=True)
    unit = {'passed': True, 'at_utc': now(), 'tests_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (ROOT / 'tests').glob('*.py')}}
    atomic(ROOT / 'qualification/unit.json', unit)
    real = read(ROOT / 'qualification/real.json'); assert real['passed'] and not real['labels_used']
    for rel, digest in real['code_sha256'].items():
        assert sha(ROOT / rel) == digest, ('Qualification code changed', rel)
    inputs = read(ROOT / 'provenance/inputs.json')
    for rel, digest in inputs['source_files'].items():
        assert sha(PROJECT / rel) == digest, ('Source input changed', rel)
    assert sha(BASE / 'provenance/freeze.json') == cfg['source_fingerprint']
    source = read(ROOT / 'data/source_metadata.json'); sample = read(ROOT / 'data/sample.json')
    jobs = sorted(range(len(sample)), key=lambda i: (-source[i]['elapsed_seconds'], sample[i]['uid']))
    heap = [(0., r) for r in range(cfg['budgets']['world_size'])]; heapq.heapify(heap)
    assignment = {str(r): [] for r in range(cfg['budgets']['world_size'])}
    for i in jobs:
        load, rank = heapq.heappop(heap); assignment[str(rank)].append(i)
        heapq.heappush(heap, (load + max(.01, source[i]['elapsed_seconds']), rank))
    assert sorted(i for a in assignment.values() for i in a) == list(range(8000))
    atomic(ROOT / 'data/worker_assignment.json', assignment)
    paths = [ROOT / 'README.md', ROOT / 'config.json', *sorted((ROOT / 'scripts').glob('*.py')),
        *sorted((ROOT / 'tests').glob('*.py')), ROOT / 'slurm/pilot.sbatch',
        *sorted((ROOT / 'data').glob('*')), ROOT / 'qualification/unit.json', ROOT / 'qualification/real.json',
        ROOT / 'provenance/inputs.json', ROOT / 'provenance/source-feature-digests.json', ROOT / 'provenance/resource-start.json']
    files = {str(p.relative_to(PROJECT)): sha(p) for p in paths if p.is_file()}
    files.update(inputs['source_files'])
    files[str(BASE / 'provenance/freeze.json').replace(str(PROJECT) + '/', '')] = cfg['source_fingerprint']
    atomic(ROOT / 'provenance/freeze.json', {'at_utc': now(), 'files': files,
        'image': inputs['image'], 'image_sha256': inputs['image_sha256'],
        'test_accessed': False, 'new_outcome_heads_fitted': False, 'production_authorized': False})
    print({'frozen': True, 'fingerprint': sha(ROOT / 'provenance/freeze.json'), 'assignment_estimated_seconds': sorted(heap)}, flush=True)


if __name__ == '__main__': main()
