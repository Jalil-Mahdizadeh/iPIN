"""Verify that an unfrozen or oversized invocation cannot begin training."""
import json
import os
import subprocess
import sys
from pathlib import Path
from state import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
cfg = ROOT / 'configs/clean-cls-linear-lr2e-5-fold0-seed2.json'
out = ROOT / 'runs/qualification-must-not-create-this'
assert not out.exists()
base = [sys.executable, str(ROOT / 'scripts/train.py'), '--config', str(cfg), '--output', str(out)]
run = subprocess.run(base + ['--production'], env={**os.environ, 'WORLD_SIZE': '4'},
                     text=True, capture_output=True, timeout=60)
assert run.returncode != 0 and 'Production requires a frozen, qualified release' in run.stderr
assert not out.exists()
qout = ROOT / 'qualification/oversized-must-not-create-this'
run2 = subprocess.run([sys.executable, str(ROOT / 'scripts/train.py'), '--config', str(cfg),
                       '--output', str(qout), '--qualification', '--max-updates', '6000',
                       '--train-limit', '67', '--val-limit', '8'], text=True, capture_output=True, timeout=60)
assert run2.returncode != 0 and not qout.exists()
assert not list((ROOT / 'runs').iterdir())
atomic_json(ROOT / 'qualification/production-guard.json', {
    'passed': True, 'unfrozen_production_rejected_before_creating_output': True,
    'oversized_qualification_rejected': True, 'production_directories_created': 0,
    'source_sha256': {f'scripts/{n}': sha256(ROOT / 'scripts' / n) for n in
                      ['train.py', 'release.py', 'qualify_production_guard.py']}})
print('Production guard passed; no production output directories created.')
