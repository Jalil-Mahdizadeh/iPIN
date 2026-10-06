"""Freeze the user-corrected training-only wrapper with the same resumable core."""
import datetime
import json
import shutil
from pathlib import Path
from launch import ROOT, sha, verify


def main():
    old = ROOT / 'releases/20260930-readout'
    inherited = verify(old)
    name = 'clean-residue-mean-official-seed2'
    run = ROOT / 'runs' / name
    contract = json.loads((run / 'contract.json').read_text())
    cfg_path = ROOT / 'configs' / f'{name}.json'
    assert contract['configuration'] == json.loads(cfg_path.read_text())
    wrapper = ROOT / 'qualification/final-training-only-wrapper.json'
    report = json.loads(wrapper.read_text())
    assert report['passed'] and report['wrapper_sha256'] == sha(ROOT / 'slurm/final-training.sbatch')
    folder = ROOT / 'releases/20261001-final-training'
    assert not folder.exists()
    for d in ['code', 'configs', 'slurm', 'protocol']: (folder / d).mkdir(parents=True)
    core = ['train.py', 'model.py', 'data.py', 'state.py', 'release.py']
    for n in core:
        assert sha(old / 'code' / n) == contract['code'][n]
        shutil.copy2(old / 'code' / n, folder / 'code' / n)
    for n in ['container.sh', 'launch.py', 'final_runtime.py', 'launch_final.py']:
        shutil.copy2(ROOT / 'scripts' / n, folder / 'code' / n)
    shutil.copy2(cfg_path, folder / 'configs' / cfg_path.name)
    shutil.copy2(ROOT / 'slurm/final-training.sbatch', folder / 'slurm/train.sbatch')
    for p in [ROOT / 'FINAL_PROTOCOL.md', ROOT / 'decisions/readout-window5000.json',
              ROOT / 'provenance/final-training-only-amendment.json']:
        shutil.copy2(p, folder / 'protocol' / p.name)
    reports = dict(inherited['qualification_reports'])
    reports[str(wrapper.relative_to(ROOT))] = sha(wrapper)
    manifest = {'ready_for_submission': True, 'production_submitted_at_preparation': False,
        'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'stage': 'final-single-run-training', 'enabled_runs': [name],
        'scope': 'One full-data residue-MLP training run only; benchmarking deferred at user request.',
        'automatic_final_benchmark': False, 'test_inference_enabled': False,
        'resume_training_fingerprint': contract['fingerprint'],
        'supersedes_release': '20261001-final',
        'files': {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
        'qualification_reports': reports,
        'data_manifest_sha256': inherited['data_manifest_sha256'],
        'initialization_manifest_sha256': inherited['initialization_manifest_sha256'],
        'sif_sha256': inherited['sif_sha256'], 'readout_release_sha256': sha(old / 'release.json'),
        'development_gate_passed': False, 'user_authorized_exploratory_override': True,
        'original_decision_sha256': sha(ROOT / 'decisions/readout-window5000.json')}
    (folder / 'release.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for p in folder.rglob('*'):
        if p.is_file(): p.chmod(0o444)
    for p in sorted(folder.rglob('*'), reverse=True):
        if p.is_dir(): p.chmod(0o555)
    folder.chmod(0o555)
    (ROOT / 'releases/CURRENT_FINAL').write_text(folder.name + '\n')
    print(json.dumps({'release': str(folder), 'training_contract_unchanged': True,
        'resume_fingerprint': contract['fingerprint'], 'automatic_benchmark': False}, indent=2))


if __name__ == '__main__': main()
