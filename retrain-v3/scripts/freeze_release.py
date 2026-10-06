"""Freeze only the six qualified v3 adaptation runs. Never calls SLURM."""
import argparse
import datetime
import json
import shutil
from pathlib import Path
from launch import sha

ROOT = Path(__file__).resolve().parents[1]
CORE = ['train.py', 'model.py', 'data.py', 'state.py', 'release.py']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name', default='20260930-adaptation')
    args = parser.parse_args()
    assert Path(args.name).name == args.name
    folder = ROOT / 'releases' / args.name
    assert not folder.exists(), 'Never overwrite a frozen release'
    core = {name: sha(ROOT / 'scripts' / name) for name in CORE}
    reports = ['v3-contract.json', 'checkpoint-faults.json', 'signal-resume.json',
               'signal-comparison.json', 'full-model-profile.json', 'ddp.json',
               'ddp-clean-comparison.json', 'launcher-checks.json', 'decisions.json', 'production-guard.json']
    evidence = {}
    for relative in ['qualification/' + n for n in reports] + ['diagnostics/cheap-baselines.json']:
        path = ROOT / relative
        report = json.loads(path.read_text())
        assert report['passed'], relative
        if 'code' in report:
            assert report['code'] == core, (relative, 'code changed')
        if 'data_manifest_sha256' in report:
            assert report['data_manifest_sha256'] == sha(ROOT / 'data/prepared/manifest.json')
        for source, digest in report.get('source_sha256', {}).items():
            assert sha(ROOT / source) == digest, (relative, source, 'changed after qualification')
        evidence[relative] = sha(path)
    campaign = json.loads((ROOT / 'configs/campaign.json').read_text())
    enabled = campaign['initial_runs']
    assert len(enabled) == 6 and all('cls-linear' in name and '-fold' in name for name in enabled)
    assert not campaign['production_submission_authorized_now']
    assert not list((ROOT / 'runs').iterdir()), 'Initial preparation must leave production unstarted'
    folder.mkdir()
    for directory in ['code', 'configs', 'slurm', 'protocol']:
        (folder / directory).mkdir()
    for name in CORE + ['container.sh', 'launch.py', 'compare_development.py', 'metrics.py']:
        shutil.copy2(ROOT / 'scripts' / name, folder / 'code' / name)
    shutil.copy2(ROOT / 'slurm/train.sbatch', folder / 'slurm/train.sbatch')
    for name in enabled:
        shutil.copy2(ROOT / 'configs' / (name + '.json'), folder / 'configs' / (name + '.json'))
    for name in ['PROTOCOL.md', 'DATA_CARD.md', 'EXPERIMENT_LEDGER.md', 'QUALIFICATION.md']:
        shutil.copy2(ROOT / name, folder / 'protocol' / name)
    shutil.copy2(ROOT / 'configs/campaign.json', folder / 'protocol/campaign.json')
    files = {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
    sif = sha(ROOT.parent / 'images/plm-interact/plm-interact-native-arm64-v1.sif')
    assert sif == 'e064e38053d6dfcacc65a23467d97f75f79ca6095e6f760def4125ccf452ffc2'
    manifest = {
        'ready_for_submission': True, 'production_submitted_at_preparation': False,
        'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'stage': 'adaptation', 'enabled_runs': enabled, 'files': files,
        'qualification_reports': evidence, 'data_manifest_sha256': sha(ROOT / 'data/prepared/manifest.json'),
        'initialization_manifest_sha256': sha(ROOT / 'provenance/downloads.json'), 'sif_sha256': sif,
        'scope': 'Only Stage 1 (six matched clean-BCE learning-rate/fold runs). Stage 2, optional cap, native adaptation and confirmation are disabled.',
        'proposal_sha256': sha(ROOT / 'provenance/improvment-proposal-v3.md'),
        'budget_gpu_hours': campaign['stage1_budget_gpu_hours'],
        'external_confirmation_ready': False}
    (folder / 'release.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for p in folder.rglob('*'):
        if p.is_file():
            p.chmod(0o444)
    for p in sorted(folder.rglob('*'), reverse=True):
        if p.is_dir():
            p.chmod(0o555)
    folder.chmod(0o555)
    (ROOT / 'releases/CURRENT').write_text(args.name + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
