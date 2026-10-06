"""Audit permitted scientific differences independently of training execution."""
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
base=json.loads((root/'configs/reference-official-seed2.json').read_text())
assert base['root']==str(root) and base['initialization']=='esm2_pretrained'
expected={'seed':2,'world_size':4,'total_updates':12745,'global_pairs_per_update':64,
    'learning_rate':2e-5,'warmup_updates':2000,'classification_weight':10.,'positive_weight':1.,
    'classification_corruption':True,'mlm_weight':1.,'readout':'cls_linear','readout_width':128,
    'attention_mode':'standard','attention_backend':'efficient','train_cap_residues':None,
    'validate_every_updates':1000,'selection_metric':'pooled_ap'}
for key,value in expected.items():assert base[key]==value,key
differences={
    'reference':{},'capped':{'train_cap_residues':2193},'positive10':{'positive_weight':10.},
    'capped-positive10':{'train_cap_residues':2193,'positive_weight':10.},
    'masked-bce':{'mlm_weight':0.},'clean-bce':{'classification_corruption':False,'mlm_weight':0.},
    'clean-plus-mlm':{'classification_corruption':False},'lower-lr':{'learning_rate':5e-6},
    'cls-mlp':{'readout':'cls_mlp'},'residue-readout':{'readout':'residue_mean'},
    'chain-aware':{'attention_mode':'chain_aware'},
    'chain-residue':{'attention_mode':'chain_aware','readout':'residue_mean'}}
for arm,change in differences.items():
    name=f'{arm}-official-seed2'
    configuration=json.loads((root/'configs'/f'{name}.json').read_text())
    assert configuration=={**base,**change,'arm':arm,'name':name},arm
    for fold in range(3):
        name=f'{arm}-fold{fold}-seed2'
        observed=json.loads((root/'configs'/f'{name}.json').read_text())
        assert observed=={**configuration,'name':name,'partition':f'fold-{fold}','total_updates':6000,'warmup_updates':1000},name
campaign=json.loads((root/'configs/campaign.json').read_text())
assert campaign['groups']['initial-screen']==['reference-official-seed2','capped-official-seed2','positive10-official-seed2','clean-bce-official-seed2']
report={'passed':True,'official_configurations':12,'development_configurations':36,
    'controlled_differences_verified':True,'initial_runs':campaign['groups']['initial-screen'],
    'full_pair_exposure':12745*64,'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((root/'configs').glob('*.json'))}}
(root/'qualification/configurations.json').write_text(json.dumps(report,indent=2)+'\n')
print('All 48 experiment configurations match their permitted differences; initial exposure = 815680 pairs per arm.')
