"""Exercise final pipeline phase/requeue guards using mocked scheduler commands."""
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = 'clean-residue-mean-official-seed2'


def main():
    source = ROOT / 'slurm/final.sbatch'
    results = []
    with tempfile.TemporaryDirectory(prefix='final-wrapper-', dir=ROOT / 'qualification') as temporary:
        base = Path(temporary)
        for scenario, expected in [('normal', 0), ('resume_benchmark', 0), ('manual_stop', 0),
            ('train_failure', 31), ('incomplete_train', 79), ('training_requeue', 0),
            ('benchmark_requeue', 0), ('benchmark_failure', 32), ('requeue_limit', 78)]:
            folder = base / scenario
            r, b, release, binaries = [folder / n for n in ['retrain', 'benchmark', 'release', 'bin']]
            run = r / 'runs' / RUN
            for p in [run, b, release / 'code', release / 'configs', binaries]: p.mkdir(parents=True)
            (release / 'code/final_runtime.py').write_text('print("mock release verification; production verifier tested separately")\n')
            if scenario == 'resume_benchmark': (run / 'completed.json').write_text('{}')
            if scenario == 'manual_stop': (run / 'REQUEST_STOP').touch()
            script = folder / 'run.sh'
            script.write_text(source.read_text().replace(f'R={ROOT}', f'R={r}').replace(f'B={ROOT.parent / "benchmark-v3"}', f'B={b}'))
            fake = '''#!/usr/bin/env python3
import os,sys
from pathlib import Path
r=Path(os.environ['FIXTURE_R']);b=Path(os.environ['FIXTURE_B']);run=r/'runs/clean-residue-mean-official-seed2'
case=os.environ['FIXTURE_CASE'];training='--production' in sys.argv
with (r/'calls.txt').open('a') as f:f.write(('train' if training else 'benchmark')+'\\n')
if training:
 if case=='train_failure':sys.exit(31)
 if case=='incomplete_train':sys.exit(0)
 if case in ['training_requeue','requeue_limit']:
  (run/'REQUEST_REQUEUE').touch();(run/'stopped.json').write_text('{}');sys.exit(0)
 (run/'completed.json').write_text('{}')
else:
 if case=='benchmark_failure':sys.exit(32)
 if case=='benchmark_requeue':
  (run/'REQUEST_REQUEUE').touch();sys.exit(75)
 (b/'completed.json').write_text('{}');(r/'FINALIZED.json').write_text('{}')
'''
            (binaries / 'srun').write_text(fake); (binaries / 'srun').chmod(0o755)
            (binaries / 'scontrol').write_text('#!/usr/bin/env bash\nprintf "%s\\n" "$*" >> "$FIXTURE_R/requeues.txt"\n')
            (binaries / 'scontrol').chmod(0o755)
            env = dict(os.environ, PATH=str(binaries) + ':' + os.environ['PATH'], FIXTURE_R=str(r),
                FIXTURE_B=str(b), FIXTURE_CASE=scenario, SLURM_JOB_ID='fixture-42',
                SLURM_RESTART_COUNT='8' if scenario == 'requeue_limit' else '0')
            result = subprocess.run(['bash', str(script), str(release)], env=env, capture_output=True, text=True, timeout=30)
            assert result.returncode == expected, (scenario, result.returncode, result.stdout, result.stderr)
            calls = (r / 'calls.txt').read_text().splitlines() if (r / 'calls.txt').exists() else []
            wanted = (['train', 'benchmark'] if scenario in ['normal', 'benchmark_requeue', 'benchmark_failure']
                      else ['benchmark'] if scenario == 'resume_benchmark' else [] if scenario == 'manual_stop' else ['train'])
            assert calls == wanted, (scenario, calls)
            assert (r / 'requeues.txt').exists() == (scenario in ['training_requeue', 'benchmark_requeue'])
            if (r / 'requeues.txt').exists(): assert (r / 'requeues.txt').read_text().strip() == 'requeue fixture-42'
            assert (r / 'FINALIZED.json').exists() == (scenario in ['normal', 'resume_benchmark'])
            results.append({'scenario': scenario, 'exit_code': result.returncode, 'phases': calls, 'passed': True})
    report = {'passed': True, 'wrapper_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'scheduler_commands_mocked': True, 'production_training_executed': False, 'cases': results}
    (ROOT / 'qualification/final-wrapper.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
