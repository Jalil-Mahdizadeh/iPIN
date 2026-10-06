"""Freeze the two preregistered arms only after their engineering gates pass."""
import hashlib
import json
import shutil
from pathlib import Path

root=Path(__file__).resolve().parents[1]
required=['attention.json','single-v3-resume-comparison.json',
          'ddp-v3-resume-comparison.json','checkpoint-faults.json','data-invariants.json']
for name in required:
    report=json.loads((root/'qualification'/name).read_text())
    assert report['passed'],name
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()
for qualification in ['single-v3-uninterrupted','ddp-v3-uninterrupted']:
    contract=json.loads((root/'qualification'/qualification/'contract.json').read_text())
    for name,expected in contract['code'].items():
        assert sha(root/'scripts'/name)==expected,f'Code changed since qualification: {name}'
    assert sha(root/'data/prepared/manifest.json')==contract['data_manifest_sha256']
    assert sha(root/'provenance/downloads.json')==contract['initialization_manifest_sha256']
image=root.parent/'images/plm-interact/plm-interact-native-arm64-v1.sif'
image_hash=sha(image)
assert image_hash=='e064e38053d6dfcacc65a23467d97f75f79ca6095e6f760def4125ccf452ffc2'
for arm in ['reference','symmetric']:
    cfg=root/'configs'/f'{arm}.json';config=json.loads(cfg.read_text())
    out=root/'runs'/config['name']
    assert not out.exists(),f'Refusing to overwrite {out}'
    (out/'code').mkdir(parents=True)
    shutil.copy2(cfg,out/'config.json')
    for name in ['train.py','model.py','data.py','state.py','container.sh']:
        shutil.copy2(root/'scripts'/name,out/'code'/name)
    files=sorted((out/'code').iterdir())+[out/'config.json',image]
    records={str(p):(image_hash if p==image else sha(p)) for p in files}
    (out/'frozen-files.sha256').write_text(''.join(f'{h}  {p}\n' for p,h in records.items()))
    (out/'qualification-references.json').write_text(json.dumps({
        name:sha(root/'qualification'/name) for name in required},indent=2)+'\n')
    for p in files:
        if p!=image:p.chmod(0o444)
    print(out)
