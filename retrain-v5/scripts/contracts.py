"""Immutable inputs and two-run native-architecture production contract."""
import hashlib
import json
from pathlib import Path

CORE = ['train.py','model.py','native_head.py','model_esm2.py','model_esmc.py',
        'data.py','state.py','contracts.py','release.py']
RUNS = ['esm2-native-ilp-seed2','esmc-native-ilp-seed2']

def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return h.hexdigest()

def core_hashes(code):
    return {name:sha256(Path(code)/name) for name in CORE}

def production_config(cfg):
    root=Path(cfg['root'])
    protocol=json.loads((root/'provenance/production-configurations.json').read_text())
    assert cfg['name'] in RUNS and cfg == protocol[cfg['name']], 'Production settings differ from frozen protocol'
    assert cfg['readout']=='cls_linear' and cfg['attention_mode']=='standard'
    assert cfg['classification_corruption'] and cfg['mlm_weight']==1. and cfg['classification_weight']==10.
    assert cfg['positive_weight']==1. and cfg['world_size']==4 and cfg['global_pairs_per_update']==64
    assert cfg['initialization']==cfg['backbone']+'_pretrained'
    assert cfg['train_cap_residues'] is None and cfg['partition']=='v5_ilp'

def verify_inputs(root,backbone=None,images=True):
    root=Path(root);manifest=json.loads((root/'data/prepared/manifest.json').read_text())
    assert not manifest['test_imported'] and not manifest['partition_changed']
    assert manifest['counts']=={'train':700764,'val':165742}
    for name,digest in manifest['files'].items():assert sha256(root/'data/prepared'/name)==digest,('Data changed',name)
    for item in json.loads((root/'provenance/downloads.json').read_text()):
        if backbone is None or item['local_path'].startswith('assets/'+backbone+'/'):
            assert sha256(root/item['local_path'])==item['sha256'],('Pretrained initialization changed',item['local_path'])
    if images:
        for name,item in json.loads((root/'provenance/images.json').read_text()).items():
            if backbone is None or backbone==name:
                assert sha256(root.parent/item['relative_path'])==item['sha256'],('SIF changed',name)

def verify_evidence(root,name,digest):
    path=Path(root)/name;assert sha256(path)==digest,('Qualification changed',name)
    report=json.loads(path.read_text());assert report['passed'],name
    for source,expected in report.get('source_sha256',{}).items():
        assert sha256(Path(root)/source)==expected,('Qualification dependency changed',source)
    return report
