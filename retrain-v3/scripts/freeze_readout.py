"""Freeze the qualified second stage, preserving the original adaptation release."""
import datetime
import json
import shutil
from pathlib import Path
from launch import sha, verify

ROOT = Path(__file__).resolve().parents[1]


def main():
    old = ROOT / 'releases' / (ROOT / 'releases/CURRENT').read_text().strip()
    inherited = verify(old)
    policy = json.loads((ROOT / 'configs/readout-stage.json').read_text())
    assert sha(ROOT / 'decisions/adaptation-early.json') == policy['adaptation_decision_sha256']
    folder = ROOT / 'releases/20260930-readout'
    assert not folder.exists()
    reports = dict(inherited['qualification_reports'])
    core_names = ['train.py', 'model.py', 'data.py', 'state.py', 'release.py']
    core = {n: sha(ROOT / 'readout-code' / n) for n in core_names}
    for n in core_names[1:]:
        assert core[n] == sha(old / 'code' / n), n
    for name in ['readout-wrapper-v2.json', 'readout-aligned-protocol.json', 'readout-aligned-ddp.json',
                 'readout-aligned-cls-mlp-comparison.json', 'readout-aligned-residue-mean-comparison.json',
                 'readout-engine-amendment.json']:
        path = ROOT / 'qualification' / name
        report = json.loads(path.read_text())
        assert report['passed'], name
        if 'code' in report:
            assert report['code'] == core
        if 'data_manifest_sha256' in report:
            assert report['data_manifest_sha256'] == inherited['data_manifest_sha256']
        for source, digest in report.get('source_sha256', {}).items():
            assert sha(ROOT / source) == digest, (name, source)
        reports[str(path.relative_to(ROOT))] = sha(path)
    for name in policy['enabled_runs']:
        assert not (ROOT / 'runs' / name).exists(), 'No readout production before release freezing'
    folder.mkdir()
    for directory in ['code', 'configs', 'slurm', 'protocol']:
        (folder / directory).mkdir()
    for name in core_names:
        shutil.copy2(ROOT / 'readout-code' / name, folder / 'code' / name)
    for name in ['container.sh', 'launch.py', 'launch_readout.py', 'window_stop.py',
                 'selection_window.py', 'compare_readout.py', 'compare_development.py', 'metrics.py']:
        shutil.copy2(ROOT / 'scripts' / name, folder / 'code' / name)
    for name in policy['enabled_runs']:
        shutil.copy2(ROOT / 'configs' / (name + '.json'), folder / 'configs' / (name + '.json'))
    shutil.copy2(ROOT / 'slurm/readout.sbatch', folder / 'slurm/train.sbatch')
    for source in ['READOUT_PROTOCOL.md', 'configs/readout-stage.json', 'decisions/adaptation-early.json']:
        shutil.copy2(ROOT / source, folder / 'protocol' / Path(source).name)
    manifest = {
        'ready_for_submission': True, 'production_submitted_at_preparation': False,
        'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'stage': 'readout', 'enabled_runs': policy['enabled_runs'], 'control_runs': policy['control_runs'],
        'screen_stop_update': 5000, 'optimizer_schedule_updates': 6000,
        'scope': 'Six clean residual-readout models at LR 2e-5; same five-validation selection window as preserved controls.',
        'files': {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
        'qualification_reports': reports,
        'data_manifest_sha256': inherited['data_manifest_sha256'],
        'initialization_manifest_sha256': inherited['initialization_manifest_sha256'],
        'sif_sha256': inherited['sif_sha256'], 'budget_gpu_hours': [125, 225],
        'adaptation_release': str(old.relative_to(ROOT)), 'adaptation_release_sha256': sha(old / 'release.json'),
        'adaptation_decision_sha256': policy['adaptation_decision_sha256'],
        'engine_amendment': policy['engine_amendment'],
        'inherited_qualification_scope': 'Unchanged model, data, checkpoint implementation and scientific tests; prior engine execution evidence is historical. Amended engine has its own full-size four-GPU head resume tests.',
        'external_confirmation_ready': False}
    (folder / 'release.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for p in folder.rglob('*'):
        if p.is_file():
            p.chmod(0o444)
    for p in sorted(folder.rglob('*'), reverse=True):
        if p.is_dir():
            p.chmod(0o555)
    folder.chmod(0o555)
    (ROOT / 'releases/CURRENT_READOUT').write_text(folder.name + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
