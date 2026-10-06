"""Prepare preregistered staged ablations; creating configs never launches work."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=dict(root=str(ROOT),seed=2,partition='official',initialization='esm2_pretrained',
    world_size=4,total_updates=12745,global_pairs_per_update=64,learning_rate=2e-5,
    weight_decay=.01,warmup_updates=2000,clip_grad_norm=1.,classification_weight=10.,
    positive_weight=1.,classification_corruption=True,mlm_weight=1.,
    readout='cls_linear',readout_width=128,attention_mode='standard',attention_backend='efficient',
    train_cap_residues=None,train_token_budget=8192,train_max_pairs=4,
    eval_token_budget=16384,eval_max_pairs=8,checkpoint_every_updates=100,
    checkpoint_minutes=15,validate_every_updates=1000,log_every_updates=10,
    selection_metric='pooled_ap')
arms={
 'reference':{},
 'capped':{'train_cap_residues':2193},
 'positive10':{'positive_weight':10.},
 'capped-positive10':{'train_cap_residues':2193,'positive_weight':10.},
 'masked-bce':{'mlm_weight':0.},
 'clean-bce':{'classification_corruption':False,'mlm_weight':0.},
 'clean-plus-mlm':{'classification_corruption':False},
 'lower-lr':{'learning_rate':5e-6},
 'cls-mlp':{'readout':'cls_mlp'},
 'residue-readout':{'readout':'residue_mean'},
 'chain-aware':{'attention_mode':'chain_aware'},
 'chain-residue':{'attention_mode':'chain_aware','readout':'residue_mean'},
}
def write(path,value):path.write_text(json.dumps(value,indent=2)+'\n')
for arm,overrides in arms.items():
 cfg={**base,**overrides,'name':f'{arm}-official-seed2','arm':arm}
 write(ROOT/'configs'/f'{arm}-official-seed2.json',cfg)
 for fold in range(3):
  value={**cfg,'name':f'{arm}-fold{fold}-seed2','partition':f'fold-{fold}',
          'total_updates':6000,'warmup_updates':1000}
  write(ROOT/'configs'/f'{arm}-fold{fold}-seed2.json',value)
qualification={**base,'name':'qualification','arm':'qualification','global_pairs_per_update':3,
 'total_updates':5,'warmup_updates':1,'validate_every_updates':2,
 'checkpoint_every_updates':1,'train_token_budget':1600,'train_max_pairs':2,'log_every_updates':1}
write(ROOT/'configs/qualification.json',qualification)
groups={
 'initial-screen':['reference-official-seed2','capped-official-seed2','positive10-official-seed2','clean-bce-official-seed2'],
 'objective-completion':['masked-bce-official-seed2','clean-plus-mlm-official-seed2'],
 'weight-coverage-interaction':['capped-positive10-official-seed2'],
 'readout-screen':['cls-mlp-official-seed2','residue-readout-official-seed2'],
 'attention-screen':['chain-aware-official-seed2'],
 'attention-readout-combination':['chain-residue-official-seed2'],
 'adaptation-screen':['lower-lr-official-seed2'],
 'development-confirmation':{arm:[f'{arm}-fold{fold}-seed2' for fold in range(3)] for arm in arms}
}
write(ROOT/'configs/campaign.json',{'initial_group':'initial-screen','groups':groups,
 'production_submission_authorized_now':False,
 'staging':'Initial four-run screen only; follow-up groups require reviewing development results.',
 'development_promotion':{'minimum_mean_ap_delta':.005,'minimum_positive_folds':2,
    'maximum_single_fold_ap_drop':.01,'maximum_mean_auroc_drop':.005,
    'interpretation':'Screening thresholds, not a statistical or external-generalization claim.'},
 'external_confirmation':{'minimum_ap_delta':.01,'paired_ci_lower_above_zero':True,'training_seeds':[2,17,42],
    'status':'Requires independently curated exposure-audited holdout and seed confirmation; not replaced by Bernett test.'}})
print('Wrote',len(list((ROOT/'configs').glob('*.json'))),'configs/campaign records; initial screen:',groups['initial-screen'])
