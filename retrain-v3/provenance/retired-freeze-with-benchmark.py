"""Freeze one bounded user-authorized final run; retain all earlier decisions."""
import datetime
import json
import shutil
from pathlib import Path
from launch import ROOT, sha, verify


def main():
    old = ROOT / 'releases/20260930-readout'
    inherited = verify(old)
    bench = ROOT.parent / 'benchmark-v3'
    preflight = json.loads((bench / 'provenance/preflight.json').read_text())
    assert preflight['passed']
    finalized = json.loads((bench / 'provenance/finalization-check.json').read_text())
    assert finalized['passed']
    for name, digest in finalized['source_sha256'].items():
        assert sha(bench / 'scripts' / name) == digest
    for name in ['common.py', 'qualify.py', 'frozen_data.py', 'training_data.py', 'frozen_model.py']:
        assert sha(bench / 'scripts' / name) == preflight['source_sha256']['scripts/' + name]
    wrapper = ROOT / 'qualification/final-wrapper.json'
    checked = json.loads(wrapper.read_text())
    assert checked['passed'] and checked['wrapper_sha256'] == sha(ROOT / 'slurm/final.sbatch')
    name = 'clean-residue-mean-official-seed2'
    assert not (ROOT / 'runs' / name).exists()
    folder = ROOT / 'releases/20261001-final'
    assert not folder.exists()
    for d in ['code', 'configs', 'slurm', 'protocol']: (folder / d).mkdir(parents=True)
    core = ['train.py', 'model.py', 'data.py', 'state.py', 'release.py']
    for f in core:
        assert sha(ROOT / 'readout-code' / f) == sha(old / 'code' / f)
        shutil.copy2(old / 'code' / f, folder / 'code' / f)
    for f in ['container.sh', 'launch.py', 'final_runtime.py', 'launch_final.py']:
        shutil.copy2(ROOT / 'scripts' / f, folder / 'code' / f)
    shutil.copy2(ROOT / 'configs' / f'{name}.json', folder / 'configs' / f'{name}.json')
    shutil.copy2(ROOT / 'slurm/final.sbatch', folder / 'slurm/train.sbatch')
    for path in [ROOT / 'FINAL_PROTOCOL.md', ROOT / 'decisions/readout-window5000.json']:
        shutil.copy2(path, folder / 'protocol' / path.name)
    evidence = {'passed': True, 'benchmark_preflight_sha256': sha(bench / 'provenance/preflight.json'),
        'benchmark_finalization_check_sha256': sha(bench / 'provenance/finalization-check.json'),
        'same_qualified_core': {n: sha(folder / 'code' / n) for n in core},
        'source_release_sha256': sha(old / 'release.json'),
        'scope': 'Identical core; full-size four-GPU exact residue resume qualification inherited; final adapter forward checked separately.'}
    (ROOT / 'qualification/final-preflight.json').write_text(json.dumps(evidence, indent=2) + '\n')
    reports = dict(inherited['qualification_reports'])
    for f in ['final-wrapper.json', 'final-preflight.json']:
        reports['qualification/' + f] = sha(ROOT / 'qualification' / f)
    inputs = [bench / 'config.json', bench / 'PROTOCOL.md']
    inputs += [p for d in ['scripts', 'data', 'reused', 'provenance'] for p in (bench / d).iterdir() if p.is_file()]
    pinned = {str(p.relative_to(bench)): sha(p) for p in sorted(inputs)}
    manifest = {'ready_for_submission': True, 'production_submitted_at_preparation': False,
        'frozen_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'stage': 'final-single-run-closeout', 'enabled_runs': [name],
        'scope': 'One exploratory full official-data residue-MLP run; automatic final benchmark; end v3 regardless of result.',
        'files': {str(p.relative_to(folder)): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()},
        'benchmark_input_sha256': pinned, 'qualification_reports': reports,
        'data_manifest_sha256': inherited['data_manifest_sha256'],
        'initialization_manifest_sha256': inherited['initialization_manifest_sha256'],
        'sif_sha256': inherited['sif_sha256'], 'readout_release_sha256': sha(old / 'release.json'),
        'development_gate_passed': False, 'user_authorized_exploratory_override': True,
        'original_decision_sha256': sha(ROOT / 'decisions/readout-window5000.json'),
        'further_v3_experiments': False}
    (folder / 'release.json').write_text(json.dumps(manifest, indent=2) + '\n')
    for p in folder.rglob('*'):
        if p.is_file(): p.chmod(0o444)
    for p in sorted(folder.rglob('*'), reverse=True):
        if p.is_dir(): p.chmod(0o555)
    folder.chmod(0o555)
    for p in inputs: p.chmod(0o444)
    (bench / 'scripts').chmod(0o555)
    (ROOT / 'releases/CURRENT_FINAL').write_text(folder.name + '\n')
    print(json.dumps({'release': str(folder), 'runs': [name], 'benchmark_inputs_pinned': len(pinned)}, indent=2))


if __name__ == '__main__': main()
