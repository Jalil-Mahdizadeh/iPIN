"""Model-free tests of default dry-run behavior and batch failure/signal handling."""
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch
import hashlib

root=Path(__file__).resolve().parents[1]
fixture=root/'qualification/launcher-fixture'
assert not fixture.exists(),'Keep prior evidence'
fixture.mkdir();(fixture/'bin').mkdir();(fixture/'releases/mock/code').mkdir(parents=True)
release=fixture/'releases/mock';(release/'configs').mkdir();(fixture/'provenance').mkdir()
names=json.loads((root/'configs/campaign.json').read_text())['initial_runs']
(release/'release.json').write_text(json.dumps({'enabled_runs':names}))
for name in names:(release/'configs'/f'{name}.json').write_text('{}')
(fixture/'releases/CURRENT').write_text('mock\n')
script=root/'slurm/train.sbatch'
subprocess.run(['bash','-n',str(script)],check=True)
# Only the root path is substituted. Actual batch control flow is exercised with
# local SLURM doubles; no scheduler command reaches the real cluster here.
copy=fixture/'train.sbatch'
copy.write_text(script.read_text().replace(str(root),str(fixture)))
srun=fixture/'bin/srun'
srun.write_text('#!/usr/bin/env bash\nset -euo pipefail\nwhile test "$1" != bash; do shift; done\nexec "$@"\n')
scontrol=fixture/'bin/scontrol'
scontrol.write_text('#!/usr/bin/env bash\nset -euo pipefail\nprintf "%s\\n" "$*" >> "$QUAL_CONTROL_LOG"\n')
for path in [srun,scontrol]:path.chmod(0o755)
worker=release/'code/container.sh'
worker.write_text('''#!/usr/bin/env bash
set -euo pipefail
exec python - "$@" <<'PY'
import json,os,signal,sys,time
from pathlib import Path
p=Path(sys.argv[sys.argv.index('--output')+1]);kind=os.environ['QUAL_CASE']
if kind=='failure':sys.exit(42)
if kind=='requeue':
    os.kill(os.getppid(),signal.SIGUSR1)
    for _ in range(100):
        if (p/'REQUEST_REQUEUE').exists():break
        time.sleep(.02)
    assert (p/'REQUEST_REQUEUE').exists()
    (p/'stopped.json').write_text(json.dumps({'checkpoint_committed':True}))
else:(p/'completed.json').write_text('{}')
PY
''')
checks=[]
for index,case in enumerate(['success','failure','requeue','manual-stop','success','success']):
    name=names[index];out=fixture/'runs'/name
    if case=='manual-stop':out.mkdir(parents=True);(out/'REQUEST_STOP').touch()
    env={**os.environ,'PATH':str(fixture/'bin')+':'+os.environ['PATH'],'QUAL_CASE':case,
         'QUAL_CONTROL_LOG':str(fixture/'scheduler-calls.txt'),'SLURM_JOB_ID':'9876',
         'SLURM_ARRAY_JOB_ID':'9870','SLURM_ARRAY_TASK_ID':str(index),'SLURM_RESTART_COUNT':'0'}
    result=subprocess.run(['bash',str(copy),str(release)],env=env,text=True,capture_output=True,timeout=30)
    assert result.returncode==(42 if case=='failure' else 0),(case,result.stdout,result.stderr)
    if case=='success':assert (out/'completed.json').exists()
    if case=='requeue':assert (fixture/'scheduler-calls.txt').read_text()=='requeue 9870_2\n'
    if case=='manual-stop':assert not (out/'completed.json').exists()
    checks.append(case)
spec=importlib.util.spec_from_file_location('launch_under_test',root/'scripts/launch.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
# Run the real CLI entry point with verification substituted by an already-
# verified empty fixture; assert it cannot call sbatch or any external process.
dry_names=[f'unused-{i}' for i in range(6)]
with patch.object(module,'ROOT',fixture),patch.object(module,'verify',return_value={'enabled_runs':dry_names}),\
     patch.object(sys,'argv',['launch.py']),patch.object(module.subprocess,'run',side_effect=AssertionError('Dry run invoked subprocess')):
    module.main()
plan=json.loads((fixture/'provenance/production-launch-plan.json').read_text())
assert plan['dry_run'] and plan['production_jobs_submitted_by_this_call']==0
assert all(not (fixture/'runs'/name).exists() for name in dry_names)
assert '--array=0-5%2' in plan['command']
report={'passed':True,'default_dry_run_submits_nothing':True,'dry_run_creates_no_run_directory':True,
    'batch_scenarios':checks,'array_task_specific_requeue':True,
    'scope':'Real shell logic, mocked SLURM/worker commands; no v3 scheduler requeue was performed.',
    'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in [script,root/'scripts/launch.py']}}
(root/'qualification/launcher-checks.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
