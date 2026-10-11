"""Freeze the executed protocol after CPU/GPU qualifications, before any head fit."""
import subprocess
import sys
from pathlib import Path
from shared import ROOT, PROJECT, BASE, read, config, sha, now, atomic


def main():
    if (ROOT / 'provenance/freeze.json').exists() or (ROOT / 'results/fit-start.json').exists():
        raise RuntimeError('Cannot replace an existing freeze or fitted experiment')
    cfg = config(); prep = read(ROOT / 'provenance/preparation.json')
    assert prep['complete'] and prep['rows'] == 222452
    q = read(ROOT / 'qualification/encoder.json'); b = read(ROOT / 'qualification/baseline.json')
    assert q['passed'] and b['passed'] and b['rows'] == 59260
    assert b['baseline_sha256'] == sha(ROOT / 'data/baseline.npz') and b['code_sha256'] == sha(ROOT / 'scripts/baseline_missing.py')
    for name, digest in q['code_sha256'].items():
        assert sha(ROOT / name) == digest, name
    amendment = read(ROOT / 'provenance/resource-amendment-1.json')
    original_cfg = read(ROOT / 'provenance/config-at-start.json')
    assert original_cfg['budgets']['batch_wall_hours'] == amendment['old_wall_hours']
    original_cfg['budgets']['batch_wall_hours'] = amendment['new_wall_hours']
    assert cfg == original_cfg, 'Only the pre-fit walltime ceiling may change'
    original_doc = (ROOT / 'provenance/proposal-at-start.md').read_text()
    assert (ROOT / 'README.md').read_text() == original_doc.replace(
        '24 hours, 192 allocated GPU-hours and 13,824 allocated CPU-core-hours',
        '28 hours, 224 allocated GPU-hours and 16,128 allocated CPU-core-hours')
    for name, digest in read(ROOT / 'provenance/preparation-start.json')['code'].items():
        local = str((PROJECT / name).relative_to(ROOT)) if str((PROJECT / name)).startswith(str(ROOT) + '/') else None
        if local in amendment['old_file_sha256']:
            assert digest == amendment['old_file_sha256'][local]
            assert sha(PROJECT / name) == amendment['new_file_sha256'][local]
        else:
            assert sha(PROJECT / name) == digest, name
    # Check the original freeze after relocation, including cached source data.
    original = read(BASE / 'provenance/freeze.json')
    for name, digest in original['files'].items():
        assert sha(PROJECT / name) == digest, name
    log = ROOT / 'qualification/unit.log'
    with log.open('wb') as handle:
        subprocess.run([sys.executable, '-B', '-m', 'unittest', 'discover', '-s', str(ROOT / 'tests'), '-v'],
                       stdout=handle, stderr=subprocess.STDOUT, check=True)
    atomic(ROOT / 'qualification/unit.json', {'at_utc': now(), 'passed': True, 'log_sha256': sha(log),
        'test_sha256': {str(p.relative_to(ROOT)): sha(p) for p in sorted((ROOT / 'tests').glob('*.py'))}})
    image = PROJECT / cfg['image']; assert sha(image) == cfg['image_sha256']
    native = PROJECT / 'images/plm-interact/plm-interact-native-arm64-v1.sif'
    native_sha = 'e064e38053d6dfcacc65a23467d97f75f79ca6095e6f760def4125ccf452ffc2'
    assert sha(native) == native_sha
    paths = [ROOT / 'README.md', ROOT / 'config.json', BASE / 'provenance/freeze.json', BASE / 'config.json']
    paths += list((ROOT / 'scripts').glob('*.py')) + list((ROOT / 'tests').glob('*.py')) + list((ROOT / 'slurm').glob('*.sbatch'))
    paths += [p for p in (ROOT / 'data').iterdir() if p.is_file()]
    paths += [ROOT / 'qualification' / n for n in ['encoder.json', 'baseline.json', 'unit.json']]
    paths += [ROOT / 'provenance' / n for n in ['resource-amendment-1.json', 'config-at-start.json', 'proposal-at-start.md', 'authorization.json', 'preparation-start.json', 'preparation.json']]
    paths += [BASE / 'scripts' / n for n in ['common.py', 'msa.py', 'encoder.py']]
    paths += [BASE / 'data' / n for n in ['sample.json', 'dev-groups.json', 'proteins.json', 'monomer-catalog.json']]
    paths += [PROJECT / 'benchmark-v1/scripts' / n for n in ['common.py', 'infer.py', 'frozen_model.py', 'frozen_data.py']]
    files = {str(p.relative_to(PROJECT)): sha(p) for p in paths}
    for entry in read(ROOT / 'data/monomer-catalog.json').values():
        assert sha(PROJECT / entry['path']) == entry['sha256']
        files[entry['path']] = entry['sha256']
    source_fp = sha(BASE / 'provenance/freeze.json')
    for row in read(BASE / 'data/sample.json'):
        p = BASE / 'features' / (row['uid'] + '.json'); side = read(p.with_suffix('.sha.json'))
        assert sha(p) == side['sha256'] and side['fingerprint'] == source_fp
        files[str(p.relative_to(PROJECT))] = side['sha256']; files[str(p.with_suffix('.sha.json').relative_to(PROJECT))] = sha(p.with_suffix('.sha.json'))
    files.update(prep['source_files'])
    atomic(ROOT / 'provenance/freeze.json', {'at_utc': now(), 'files': files, 'image': cfg['image'],
        'image_sha256': cfg['image_sha256'], 'native_image_sha256': native_sha,
        'original_pilot_freeze_verified': True, 'no_heads_fitted': True, 'no_shuffle': True, 'test_authorized': False})
    print({'frozen': True, 'fingerprint': sha(ROOT / 'provenance/freeze.json'), 'files': len(files)}, flush=True)


if __name__ == '__main__':
    main()
