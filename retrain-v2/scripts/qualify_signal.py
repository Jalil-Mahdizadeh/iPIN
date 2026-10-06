"""Deliver a real SIGUSR1 to a bounded disposable trainer and compare its restart."""
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from state import atomic_json, sha256

root=Path(__file__).resolve().parents[1];qual=root/'qualification'
cfg=json.loads((root/'configs/qualification.json').read_text())
cfg.update(global_pairs_per_update=7,train_max_pairs=1)
config=qual/'signal-config.json';atomic_json(config,cfg)
a,b=qual/'signal-full',qual/'signal-resumed'
assert not a.exists() and not b.exists(),'Preserve qualification evidence on repeat'
def command(out):
    return [sys.executable,str(root/'scripts/train.py'),'--config',str(config),'--output',str(out),
            '--qualification','--tiny','--max-updates','3','--train-limit','11','--val-limit','8',
            '--qualification-max-tokens','384']
subprocess.run(command(a),check=True)
worker=subprocess.Popen(command(b)+['--sleep-after-update','2'])
try:
    deadline=time.monotonic()+120
    while time.monotonic()<deadline:
        assert worker.poll() is None,'Trainer exited before signal'
        status=b/'status.json'
        if status.exists() and json.loads(status.read_text())['state']['update']>=1:
            os.kill(worker.pid,signal.SIGUSR1);break
        time.sleep(.05)
    else:raise RuntimeError('No optimizer update before signal deadline')
    assert worker.wait(timeout=120)==0
finally:
    if worker.poll() is None:worker.terminate();worker.wait(timeout=120)
stopped=json.loads((b/'stopped.json').read_text())
assert stopped['checkpoint_committed'] and stopped['reason']=='signal_or_REQUEST_STOP'
assert 1<=stopped['state']['update']<3
subprocess.run(command(b),check=True)
subprocess.run([sys.executable,str(root/'scripts/compare_resume.py'),str(a),str(b),
               '--report',str(qual/'signal-comparison.json')],check=True)
atomic_json(qual/'signal-resume.json',{'passed':True,'actual_signal':'SIGUSR1',
    'stopped_at_update':stopped['state']['update'],'bitwise_restart_equivalence':True,
    'scope':'Single GPU, tiny model, real OS signal and fresh training process; scheduler wrapper checked separately.',
    'code':{n:sha256(root/'scripts'/n) for n in ['train.py','model.py','data.py','state.py','release.py']}})
