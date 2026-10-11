"""Freeze inference-only TEST protocol and all inputs after qualification."""
import subprocess
import sys
from shared import ROOT, PROJECT, TRAIN, BASE, read, config, sha, now, atomic, source_contract


def main():
    if (ROOT / 'provenance/freeze.json').exists() or (ROOT / 'results/evaluation-start.json').exists():
        raise RuntimeError('Cannot replace an existing freeze or evaluated benchmark')
    cfg = config(); source_contract(); prep = read(ROOT / 'provenance/preparation.json')
    assert prep['complete'] and prep['rows'] == cfg['test_pairs'] and prep['source_trainval_freeze_verified']
    assert not prep['heads_fitted'] and not prep['new_performance_computed'] and not prep['worker_manifest_contains_labels']
    q = read(ROOT / 'qualification/encoder.json'); b = read(ROOT / 'provenance/baseline-reuse.json')
    assert q['passed'] and not q['test_scores_computed'] and not q['heads_fitted']
    assert q['conservative_wall_hours'] <= cfg['budgets']['batch_wall_hours']
    assert b['rows_reused'] == cfg['test_pairs'] and not b['missing_rows']
    assert b['baseline_sha256'] == sha(ROOT / 'data/baseline.npz')
    for name, digest in q['code_sha256'].items():
        assert sha(PROJECT / name) == digest, name
    for name, digest in read(ROOT / 'provenance/preparation-start.json')['code'].items():
        assert sha(PROJECT / name) == digest, name
    log = ROOT / 'qualification/unit.log'
    with log.open('wb') as handle:
        subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'],
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    atomic(ROOT / 'qualification/unit.json', {'at_utc': now(), 'passed': True, 'log_sha256': sha(log),
        'test_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'tests').glob('*.py'))}})
    assert sha(PROJECT / cfg['image']) == cfg['image_sha256']
    # Pin every imported frozen numerical implementation against its source manifest.
    old = read(TRAIN / 'provenance/freeze.json')
    dependencies = [TRAIN / 'scripts' / n for n in ['inputs.py', 'guarded_encoder.py', 'heads.py', 'chunks.py', 'statistics_rx.py']]
    dependencies += [BASE / 'scripts' / n for n in ['common.py', 'msa.py', 'encoder.py']]
    dependencies += [BASE / 'config.json', BASE / 'provenance/freeze.json', BASE / 'data/sample.json']
    for p in dependencies:
        assert sha(p) == old['files'][str(p.relative_to(PROJECT))], str(p)
    paths = [ROOT / 'README.md', ROOT / 'config.json'] + dependencies
    paths += list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'tests').glob('*.py')) + list((ROOT / 'slurm').glob('*.sbatch'))
    paths += [p for p in (ROOT / 'data').iterdir() if p.is_file()]
    paths += [ROOT / 'qualification' / n for n in ['encoder.json', 'timing-estimate.json', 'unit.json']]
    paths += [ROOT / 'provenance' / n for n in ['authorization.json', 'preparation-start.json', 'preparation.json', 'baseline-reuse.json']]
    paths += [TRAIN / 'provenance/freeze.json', TRAIN / 'config.json', TRAIN / 'qualification/encoder.json']
    paths += [TRAIN / 'results' / n for n in ['heads.json', 'COMPLETE.json', 'verification.json', 'decision.json']]
    files = {str(p.relative_to(PROJECT)): sha(p) for p in paths}; files.update(prep['source_files'])
    for entry in read(ROOT / 'data/monomer-catalog.json').values():
        assert sha(PROJECT / entry['path']) == entry['sha256']
        files[entry['path']] = entry['sha256']
    atomic(ROOT / 'provenance/freeze.json', {'at_utc': now(), 'files': files,
        'image': cfg['image'], 'image_sha256': cfg['image_sha256'],
        'source_retrain_fingerprint': cfg['retrain_fingerprint'], 'heads_sha256': cfg['heads_sha256'],
        'test_authorized': True, 'test_performance_computed': False, 'heads_fitted': False,
        'calibration_performed': False, 'no_shuffle': True})
    print({'frozen': True, 'fingerprint': sha(ROOT / 'provenance/freeze.json'), 'files': len(files)}, flush=True)


if __name__ == '__main__':
    main()
