"""Check the Python environment actually inherited by distributed workers."""
import importlib.metadata as md
import json
import os
import sys
from pathlib import Path
from model_esmc import ESMCPairModel

root = Path(__file__).resolve().parents[1]
assert sys.prefix == '/opt/esmc/venv', sys.executable
assert md.version('esm') == '3.2.3'
assert md.version('transformers') == '4.48.1'
assert md.version('torch') == '2.8.0a0+34c6371d24.nv25.8'
assert os.environ['HF_HUB_OFFLINE'] == '1'
value = {'passed': True, 'rank': int(os.environ['RANK']),
         'world_size': int(os.environ['WORLD_SIZE']), 'python': sys.executable,
         'prefix': sys.prefix, 'esm': md.version('esm'),
         'model_adapter_imported': True, 'gpu_training_performed': False}
(root / 'qualification' / ('worker-environment-rank' + os.environ['RANK'] + '.json')).write_text(json.dumps(value, indent=2) + '\n')
print(json.dumps(value), flush=True)
