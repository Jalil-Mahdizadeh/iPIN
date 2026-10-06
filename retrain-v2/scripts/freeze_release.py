"""Freeze only qualified initial-screen runs; this script never submits jobs."""
import argparse
import datetime
import hashlib
import json
import os
import shutil
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
    return h.hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--name',default='20260929-initial-screen');args=p.parse_args()
    assert Path(args.name).name==args.name
    folder=ROOT/'releases'/args.name
    assert not folder.exists(),'Never overwrite an existing release'
    core=['train.py','model.py','data.py','state.py','release.py']
    code={n:sha(ROOT/'scripts'/n) for n in core}
    reports=['science.json','checkpoint-faults.json','full-model-profile.json','ddp.json',
             'ddp-reference-comparison.json','ddp-experimental-comparison.json','signal-resume.json','launcher-checks.json',
             'scheduler-requeue.json','ddp-experimental-requeue-comparison.json','production-guard.json','configurations.json']
    evidence={}
    for name in reports:
        path=ROOT/'qualification'/name;report=json.loads(path.read_text())
        assert report['passed'],name
        if 'code' in report:assert report['code']==code,(name,'code changed after qualification')
        if 'data_manifest_sha256' in report:assert report['data_manifest_sha256']==sha(ROOT/'data/prepared/manifest.json')
        for source,digest in report.get('source_sha256',{}).items():assert sha(ROOT/source)==digest,(name,source)
        evidence[str(path.relative_to(ROOT))]=sha(path)
    campaign=json.loads((ROOT/'configs/campaign.json').read_text())
    enabled=campaign['groups']['initial-screen']
    assert enabled==['reference-official-seed2','capped-official-seed2','positive10-official-seed2','clean-bce-official-seed2']
    folder.mkdir();(folder/'code').mkdir();(folder/'configs').mkdir();(folder/'slurm').mkdir()
    for name in core+['container.sh','launch.py']:
        shutil.copy2(ROOT/'scripts'/name,folder/'code'/name)
    shutil.copy2(ROOT/'slurm/train.sbatch',folder/'slurm/train.sbatch')
    for name in enabled:shutil.copy2(ROOT/'configs'/f'{name}.json',folder/'configs'/f'{name}.json')
    files={str(path.relative_to(folder)):sha(path) for path in sorted(folder.rglob('*')) if path.is_file()}
    sif_digest=sha(ROOT.parent/'images/plm-interact/plm-interact-native-arm64-v1.sif')
    assert sif_digest=='e064e38053d6dfcacc65a23467d97f75f79ca6095e6f760def4125ccf452ffc2','Pinned SIF changed'
    manifest={'ready_for_submission':True,'production_submitted_at_preparation':False,
        'frozen_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'enabled_runs':enabled,
        'scope':'Initial four-arm controlled screen only. Later proposal stages remain conditional.',
        'files':files,'qualification_reports':evidence,
        'data_manifest_sha256':sha(ROOT/'data/prepared/manifest.json'),
        'initialization_manifest_sha256':sha(ROOT/'provenance/downloads.json'),
        'sif_sha256':sif_digest}
    (folder/'release.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for path in folder.rglob('*'):
        if path.is_file():path.chmod(0o444)
    for path in sorted(folder.rglob('*'),reverse=True):
        if path.is_dir():path.chmod(0o555)
    folder.chmod(0o555)
    (ROOT/'releases/CURRENT').write_text(args.name+'\n')
    print(json.dumps(manifest,indent=2))
if __name__=='__main__':main()
