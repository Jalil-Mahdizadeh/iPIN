"""Exercise the production shell's stop/requeue/failure paths with fake workers."""
import json
import os
import subprocess
from pathlib import Path
from contracts import sha256

ROOT = Path(__file__).resolve().parents[1]
RUN = 'esmc-standard-official-seed2'


def main():
    fixture = ROOT / 'qualification/wrapper-fixture'
    assert not fixture.exists()
    fixture.mkdir()
    source = (ROOT / 'slurm/train.sbatch').read_text()
    original_root_line = 'V4_ROOT=' + str(ROOT)
    assert source.count(original_root_line) == 1
    assert 'sbatch ' not in source and 'benchmark-v' not in source
    results = []
    for case, code, calls, requeue in [
        ('complete', 0, 1, False), ('failure', 42, 1, False),
        ('signal', 0, 1, True), ('signal-and-failure', 42, 1, False),
        ('requeue-limit', 78, 1, False), ('manual-stop', 0, 0, False),
        ('stop-during-worker', 0, 1, False), ('already-completed', 0, 0, False),
        ('exit-without-completion', 79, 1, False), ('duplicate-lock', 73, 0, False)]:
        base = fixture / case
        out = base / 'runs' / RUN
        out.mkdir(parents=True)
        release = base / 'release'; (release / 'code').mkdir(parents=True)
        (release / 'configs').mkdir()
        (release / 'code/runtime.py').write_text('pass\n')
        fakebin = base / 'bin'; fakebin.mkdir()
        wrapper = base / 'train.sbatch'
        wrapper.write_text(source.replace(original_root_line, 'V4_ROOT=' + str(base)))
        worker = fakebin / 'srun'
        worker.write_text('''#!/usr/bin/env python3
import json,os,signal,time,sys
from pathlib import Path
base=Path(os.environ['V4_FIXTURE']);out=base/'runs'/'esmc-standard-official-seed2'
(base/'worker-called').write_text('called')
case=os.environ['V4_CASE']
assert '--production' in sys.argv
assert 'torch.distributed.run' in sys.argv and '-m' in sys.argv
if case in ['signal','signal-and-failure','requeue-limit']:
    os.kill(os.getppid(),signal.SIGUSR1);time.sleep(.2)
if case=='stop-during-worker':(out/'REQUEST_STOP').touch()
if case=='complete':(out/'completed.json').write_text('{}')
raise SystemExit(42 if case in ['failure','signal-and-failure'] else 0)
''')
        worker.chmod(0o755)
        requeuer = fakebin / 'scontrol'
        requeuer.write_text('''#!/usr/bin/env python3
import os,sys
from pathlib import Path
assert sys.argv[1:]==['requeue','fixture-job']
(Path(os.environ['V4_FIXTURE'])/'requeued').write_text('recorded; no scheduler call made')
''')
        requeuer.chmod(0o755)
        if case == 'manual-stop': (out / 'REQUEST_STOP').touch()
        if case == 'already-completed': (out / 'completed.json').write_text('{}')
        env = {**os.environ, 'PATH': str(fakebin) + ':' + os.environ['PATH'],
               'V4_FIXTURE': str(base), 'V4_CASE': case, 'SLURM_JOB_ID': 'fixture-job',
               'SLURM_RESTART_COUNT': '8' if case == 'requeue-limit' else '0'}
        lock = None
        if case == 'duplicate-lock':
            import fcntl
            lock = (out / '.pipeline.lock').open('a+')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = subprocess.run(['bash', str(wrapper), str(release), RUN],
                                env=env, capture_output=True, text=True, timeout=15)
        if lock is not None: lock.close()
        (base / 'output.log').write_text(result.stdout + result.stderr)
        assert result.returncode == code, (case, result.returncode, result.stdout, result.stderr)
        assert int((base / 'worker-called').exists()) == calls, case
        assert (base / 'requeued').exists() == requeue, case
        results.append({'case': case, 'exit_code': result.returncode,
                        'worker_called': bool(calls), 'scheduler_requeue_requested': requeue})
    report = {'passed': True, 'cases': results,
              'scope': 'Actual wrapper control flow with only its root path relocated; srun/scontrol and release verification replaced by fixtures. No production job or real scheduler requeue was invoked.',
              'source_sha256': {'slurm/train.sbatch': sha256(ROOT / 'slurm/train.sbatch'),
                                'scripts/qualify_wrapper.py': sha256(Path(__file__))}}
    (ROOT / 'qualification/wrapper.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
