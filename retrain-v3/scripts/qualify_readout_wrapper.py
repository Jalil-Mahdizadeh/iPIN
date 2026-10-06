"""Exercise the actual readout shell/monitor with local scheduler and worker doubles."""
import contextlib
import hashlib
import importlib.util
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def main():
    fixture = ROOT / 'qualification/readout-wrapper-fixture'
    assert not fixture.exists()
    release = fixture / 'release'
    for path in [fixture / 'bin', fixture / 'runs', fixture / 'provenance', release / 'code', release / 'configs']:
        path.mkdir(parents=True, exist_ok=True)
    names = ['budget', 'failure', 'requeue', 'manual-stop', 'missed-window', 'monitor-error']
    (release / 'release.json').write_text(json.dumps({'enabled_runs': names, 'screen_stop_update': 2}))
    for name in names:
        (release / 'configs' / (name + '.json')).write_text(json.dumps({
            'root': str(fixture), 'total_updates': 6, 'validate_every_updates': 2}))
    shutil.copy2(ROOT / 'scripts/window_stop.py', release / 'code/window_stop.py')
    shell = fixture / 'train.sbatch'
    shell.write_text((ROOT / 'slurm/readout.sbatch').read_text().replace(str(ROOT), str(fixture)))
    subprocess.run(['bash', '-n', str(shell)], check=True)
    (fixture / 'bin/srun').write_text('#!/usr/bin/env bash\nset -euo pipefail\nwhile test "$1" != bash; do shift; done\nexec "$@"\n')
    (fixture / 'bin/scontrol').write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$QUAL_CONTROL_LOG"\n')
    for p in (fixture / 'bin').iterdir():
        p.chmod(0o755)
    (release / 'code/container.sh').write_text('''#!/usr/bin/env bash
set -euo pipefail
exec python - "$@" <<'PY'
import hashlib,json,os,signal,sys,time
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--output')+1]);kind=os.environ['QUAL_CASE']
if kind=='failure':sys.exit(42)
if kind=='missed-window':
    (p/'completed.json').write_text('{}');sys.exit(0)
if kind=='requeue':
    os.kill(os.getppid(),signal.SIGUSR1)
    for _ in range(200):
        if (p/'REQUEST_REQUEUE').exists():break
        time.sleep(.02)
    assert (p/'REQUEST_REQUEUE').exists()
    state={'update':1,'last_validation_update':0,'pending_validation':False,'total_steps':6}
    (p/'stopped.json').write_text(json.dumps({'checkpoint_committed':True,'state':state}));sys.exit(0)
(p/'validation').mkdir();(p/'checkpoints').mkdir()
(p/'contract.json').write_text(json.dumps({'fingerprint':'fixture'}))
file=p/'checkpoints/update-000000002.pt';file.write_bytes(b'local synthetic checkpoint')
meta={'file':file.name,'bytes':file.stat().st_size,'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),
      'fingerprint':'fixture','update':2,'best_update':2,'best_ap':.6}
for target in [p/'latest.json',p/'best.json',file.with_suffix('.pt.json')]:target.write_text(json.dumps(meta))
pred=p/'validation/update-000000002.npz';pred.write_bytes(b'synthetic prediction artifact')
(pred.with_suffix('.json')).write_text(json.dumps({'update':2,'fingerprint':'wrong' if kind=='monitor-error' else 'fixture',
    'sha256':hashlib.sha256(pred.read_bytes()).hexdigest()}))
for _ in range(500):
    if (p/'REQUEST_STOP').exists():break
    time.sleep(.02)
assert (p/'REQUEST_STOP').exists(),'Monitor failed to stop worker'
state={'update':2,'last_validation_update':2,'pending_validation':False,'best_update':2,'total_steps':6}
(p/'stopped.json').write_text(json.dumps({'checkpoint_committed':True,'state':state}))
PY
''')
    outcomes = {}
    for index, case in enumerate(names):
        out = fixture / 'runs' / case
        if case == 'manual-stop':
            out.mkdir(); (out / 'REQUEST_STOP').touch()
        env = {**os.environ, 'PATH': str(fixture / 'bin') + ':' + os.environ['PATH'],
               'QUAL_CASE': case, 'QUAL_CONTROL_LOG': str(fixture / 'scheduler-calls.txt'),
               'SLURM_JOB_ID': '12340', 'SLURM_ARRAY_JOB_ID': '12340', 'SLURM_ARRAY_TASK_ID': str(index),
               'SLURM_RESTART_COUNT': '0'}
        result = subprocess.run(['bash', str(shell), str(release)], env=env, text=True, capture_output=True, timeout=30)
        (fixture / (case + '.log')).write_text(result.stdout + result.stderr)
        expected = {'failure': 42, 'missed-window': 80, 'monitor-error': 1}.get(case, 0)
        assert result.returncode == expected, (case, result.returncode, result.stdout, result.stderr)
        if case == 'budget':
            report = json.loads((out / 'SCREEN_COMPLETE.json').read_text())
            assert report['comparison_window_end'] == 2 and report['actual_stopped_update'] == 2
            assert not report['completed_full_schedule']
        if case == 'requeue':
            assert (fixture / 'scheduler-calls.txt').read_text() == 'requeue 12340_2\n'
        if case == 'monitor-error':
            assert (out / 'WINDOW_MONITOR_ERROR.json').exists() and (out / 'REQUEST_STOP').exists()
        if case == 'manual-stop':
            assert not (out / 'contract.json').exists()
        outcomes[case] = result.returncode
    spec = importlib.util.spec_from_file_location('readout_launch_qualified', ROOT / 'scripts/launch_readout.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    dry_names = [f'unused-{i}' for i in range(6)]
    verified = {'enabled_runs': dry_names, 'stage': 'readout', 'screen_stop_update': 5000}
    with patch.object(module, 'ROOT', fixture), patch.object(module, 'verify', return_value=verified), \
         patch.object(module.subprocess, 'run', side_effect=AssertionError('Dry run invoked a subprocess')), \
         patch.object(sys, 'argv', ['launch_readout.py', '--release', str(release)]):
        module.main()
    plan = json.loads((fixture / 'provenance/readout-launch-plan.json').read_text())
    assert plan['dry_run'] and '--array=0-5%6' in plan['command']
    assert all(not (fixture / 'runs' / n).exists() for n in dry_names)
    with patch.object(module, 'ROOT', fixture), patch.object(module, 'verify', return_value=verified), \
         patch.object(module.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, 'plmi-v3-readout\n', '')) as process, \
         patch.object(sys, 'argv', ['launch_readout.py', '--release', str(release), '--submit']):
        try:
            module.main()
        except AssertionError as error:
            assert 'already active' in str(error)
        else:
            raise AssertionError('Duplicate campaign accepted')
        assert process.call_count == 1
    report = {'passed': True, 'scope': 'Real shell, OS signal, and window monitor; mocked local worker/scheduler, no real scheduler calls.',
              'scenarios': outcomes, 'default_six_way_dry_run': True, 'duplicate_campaign_rejected': True,
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in
                               [ROOT / 'scripts/window_stop.py', ROOT / 'scripts/launch_readout.py',
                                ROOT / 'slurm/readout.sbatch', Path(__file__)]}}
    (ROOT / 'qualification/readout-wrapper.json').write_text(json.dumps(report, indent=2) + '\n')
    print('Readout wrapper qualification passed.')


if __name__ == '__main__':
    main()
