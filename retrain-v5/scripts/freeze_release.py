"""Freeze the two qualified, explicitly authorized production runs."""
import datetime
import json
import shutil
from pathlib import Path
from contracts import CORE,RUNS,core_hashes,sha256,production_config,verify_evidence,verify_inputs

ROOT=Path(__file__).resolve().parents[1]
REPORTS=['model-esm2.json','model-esmc.json','full-length-esm2.json','full-length-esmc.json',
         'checkpoint-faults.json','wrapper.json','production-guards.json','ddp-resume.json']

def main():
    verify_inputs(ROOT);code=core_hashes(ROOT/'scripts');evidence={}
    for name in REPORTS:
        relative='qualification/'+name;digest=sha256(ROOT/relative)
        r=verify_evidence(ROOT,relative,digest)
        if 'code' in r:assert r['code']==code,('Stale qualification',relative)
        evidence[relative]=digest
    q=json.loads((ROOT/'qualification/ddp-resume.json').read_text())
    assert q['world_size']==4 and q['conditions']==['esm2','esmc']
    for name in q['comparison_reports']:
        r=json.loads((ROOT/name).read_text())
        assert r['passed'] and r['bitwise_identical'] and r['validation_predictions_bitwise_identical']
    for b in ['esm2','esmc']:
        r=json.loads((ROOT/'qualification'/f'full-length-{b}.json').read_text())
        assert {x['stage']:x['tokens'] for x in r['profiles']}=={'longest-training':17652,'longest-validation':42878}
        assert r['decoder_gradients_verified'] and r['no_sequence_truncation']
    auth=json.loads((ROOT/'provenance/authorization.json').read_text())
    assert auth['production_training_authorized'] and auth['production_runs']==RUNS
    for name in RUNS:production_config(json.loads((ROOT/'configs'/f'{name}.json').read_text()))
    assert not list((ROOT/'runs').glob('*/contract.json'))
    release=ROOT/'releases/20261002-native-ilp'
    assert not release.exists(),'Never overwrite a frozen production release'
    for folder in ['code','configs','slurm','protocol']:(release/folder).mkdir(parents=True)
    for name in CORE+['container.sh','runtime.py','launch.py']:shutil.copyfile(ROOT/'scripts'/name,release/'code'/name)
    for name in RUNS:shutil.copyfile(ROOT/'configs'/f'{name}.json',release/'configs'/f'{name}.json')
    shutil.copyfile(ROOT/'slurm/train.sbatch',release/'slurm/train.sbatch')
    shutil.copyfile(ROOT/'PROTOCOL.md',release/'protocol/PROTOCOL.md')
    shutil.copyfile(ROOT/'provenance/improvment-proposal-v5.md',release/'protocol/improvment-proposal-v5.md')
    inputs=['data/prepared/manifest.json','provenance/downloads.json','provenance/images.json',
            'provenance/production-configurations.json','provenance/authorization.json',
            'provenance/data-completed.json','provenance/inherited-code.json']
    manifest=dict(stage='v5-two-native-models-training-only',ready_for_submission=True,
                  created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),enabled_runs=RUNS,
                  production_training_authorized=True,automatic_benchmark=False,
                  qualification_reports=evidence,input_manifests={n:sha256(ROOT/n) for n in inputs},
                  code=code,files={str(p.relative_to(release)):sha256(p) for p in sorted(release.rglob('*')) if p.is_file()})
    (release/'release.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for p in release.rglob('*'):
        if p.is_file():p.chmod(0o444)
    (ROOT/'releases/CURRENT').write_text(release.name+'\n')
    print(json.dumps({'release':str(release),'ready_for_submission':True,'runs':RUNS},indent=2))

if __name__=='__main__':main()
