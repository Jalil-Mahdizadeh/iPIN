"""Require the exact qualified release before production or resumption."""
import json
from pathlib import Path
from contracts import RUNS,core_hashes,production_config,sha256,verify_evidence,verify_inputs

def verify_release(root,config_path,code_dir,verify_assets=True):
    root,config_path,code_dir=map(Path,(root,config_path,code_dir))
    release=code_dir.parent;assert release.is_relative_to(root/'releases')
    m=json.loads((release/'release.json').read_text())
    assert m['ready_for_submission'] and m['enabled_runs']==RUNS and not m['automatic_benchmark']
    assert m['production_training_authorized']
    for name,digest in m['files'].items():assert sha256(release/name)==digest,('Release changed',name)
    cfg=json.loads(config_path.read_text());production_config(cfg)
    assert cfg['root']==str(root.resolve()) and config_path.name==cfg['name']+'.json'
    assert config_path.is_relative_to(release/'configs')
    assert sha256(config_path)==m['files'][str(config_path.relative_to(release))]
    for name,digest in m['input_manifests'].items():assert sha256(root/name)==digest,name
    for name,digest in m['qualification_reports'].items():
        report=verify_evidence(root,name,digest)
        if 'code' in report:assert report['code']==core_hashes(code_dir),('Stale qualification',name)
    if verify_assets:verify_inputs(root,backbone=cfg['backbone'])
    return m
