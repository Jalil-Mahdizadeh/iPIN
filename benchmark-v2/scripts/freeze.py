"""Freeze qualified inference and prespecified analysis before job submission."""
import ast
import hashlib
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
WORK = ROOT.parent
def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(16*1024**2), b''): h.update(block)
    return h.hexdigest()
assert not (ROOT / 'provenance/frozen-files.sha256').exists(), 'Already frozen'
qualification = json.loads((ROOT / 'provenance/qualification.json').read_text())
assert qualification['passed']
for p in (ROOT / 'scripts').glob('*.py'): ast.parse(p.read_text(), filename=str(p))
core = ['config.json', 'provenance/selection.json'] + [f'scripts/{n}' for n in
    ['common.py', 'infer.py', 'frozen_data.py', 'frozen_model.py', 'container.sh']]
fingerprint = hashlib.sha256(json.dumps({n:sha(ROOT/n) for n in core},sort_keys=True).encode()).hexdigest()
assert qualification['prediction_fingerprint'] == fingerprint
files = [ROOT/p for p in core]
files += [ROOT/p for p in ['PROTOCOL.md', 'scripts/qualify.py', 'scripts/frozen_v2_model.py',
    'scripts/frozen_v2_data.py', 'scripts/analyze.py', 'slurm/benchmark.sbatch',
    'provenance/qualification.json', 'provenance/validation-before-test.json', 'provenance/reuse-verification.json']]
selection = json.loads((ROOT/'provenance/selection.json').read_text())
files += [ROOT/'data'/n for n in selection['data']['sha256']]
files += [ROOT/selection['models'][n]['path'] for n in selection['fresh_inference_models']]
files += [Path(selection['container']['path'])]
lines = [f'{sha(p)}  {p.relative_to(WORK)}' for p in files]
(ROOT/'provenance/frozen-files.sha256').write_text('\n'.join(lines)+'\n')
print(json.dumps({'frozen_files':len(files),'prediction_fingerprint':fingerprint}))
