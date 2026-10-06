"""Freeze a qualified v4 release and stop before production submission."""
import datetime
import json
import shutil
from pathlib import Path
from contracts import (CORE, RUNS, core_hashes, production_config, sha256,
                       verify_evidence, verify_inputs)

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ['model-and-data.json', 'full-length-esm2.json', 'full-length-esmc.json',
           'checkpoint-faults.json', 'wrapper.json', 'production-guards.json',
           'worker-environment.json', 'ddp-resume.json']


def main():
    verify_inputs(ROOT)
    hashes = core_hashes(ROOT / 'scripts')
    evidence = {}
    for filename in REPORTS:
        name = 'qualification/' + filename
        digest = sha256(ROOT / name)
        report = verify_evidence(ROOT, name, digest)
        if 'code' in report: assert report['code'] == hashes, ('Stale qualification', name)
        evidence[name] = digest
    ddp = json.loads((ROOT / 'qualification/ddp-resume.json').read_text())
    assert ddp['world_size'] == 4 and ddp['conditions'] == ['S1', 'C0', 'C1']
    for name in ddp['comparison_reports']:
        comparison = json.loads((ROOT / name).read_text())
        assert comparison['passed'] and comparison['bitwise_identical']
        assert comparison['validation_predictions_bitwise_identical']
    profiles = [json.loads((ROOT / 'qualification' / ('full-length-' + b + '.json')).read_text())
                for b in ['esm2', 'esmc']]
    for run in RUNS:
        cfg = json.loads((ROOT / 'configs' / (run + '.json')).read_text()); production_config(cfg)
        measured = [p for report in profiles for p in report['profiles'] if p['run'] == run]
        assert {p['stage']: p['tokens'] for p in measured} == {'longest-training': 16322, 'longest-validation': 39391}
    wrapper = json.loads((ROOT / 'qualification/wrapper.json').read_text())
    assert len(wrapper['cases']) == 10
    assert not list((ROOT / 'runs').glob('*/contract.json')), 'Preparation must not train production candidates'
    release = ROOT / 'releases/20261001-production'
    assert not release.exists(), 'Never overwrite a frozen release'
    for name in ['code', 'configs', 'slurm', 'protocol']: (release / name).mkdir(parents=True)
    for name in CORE + ['container.sh', 'runtime.py', 'launch.py']:
        shutil.copyfile(ROOT / 'scripts' / name, release / 'code' / name)
    for name in RUNS:
        shutil.copyfile(ROOT / 'configs' / (name + '.json'), release / 'configs' / (name + '.json'))
    shutil.copyfile(ROOT / 'slurm/train.sbatch', release / 'slurm/train.sbatch')
    shutil.copyfile(ROOT / 'PROTOCOL.md', release / 'protocol/PROTOCOL.md')
    shutil.copyfile(ROOT.parent / 'improvment-proposal-v4.md', release / 'protocol/improvment-proposal-v4.md')
    shutil.copyfile(ROOT / 'provenance/reused-control.json', release / 'protocol/reused-control.json')
    inputs = ['data/prepared/manifest.json', 'provenance/downloads.json', 'provenance/images.json',
              'provenance/reused-control.json', 'provenance/v3-source.json']
    manifest = {'stage': 'v4-three-run-training-only', 'ready_for_submission': True,
        'created_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'enabled_runs': RUNS, 'production_submitted_at_preparation': False,
        'automatic_benchmark': False, 'reused_control_retrained': False,
        'qualification_reports': evidence,
        'input_manifests': {name: sha256(ROOT / name) for name in inputs},
        'code': hashes,
        'files': {str(p.relative_to(release)): sha256(p) for p in sorted(release.rglob('*')) if p.is_file()}}
    (release / 'release.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for p in release.rglob('*'):
        if p.is_file(): p.chmod(0o444)
    (ROOT / 'releases/CURRENT').write_text(release.name + '\n')
    print(json.dumps({'release': str(release), 'ready_for_submission': True,
                      'production_jobs_submitted': 0, 'automatic_benchmark': False}, indent=2))


if __name__ == '__main__': main()
