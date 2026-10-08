"""Post-pilot diagnostics of saved features and frozen heads; no refitting."""
import json
import sys
import time
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent/'scripts'))
from common import ROOT, atomic, now, sha, verify_freeze
from features import load_record


def main():
    started = time.monotonic(); fingerprint = verify_freeze()
    rows = json.loads((ROOT/'data/sample.json').read_text())
    data = []
    for i, r in enumerate(rows):
        x = load_record(ROOT/'features', r['uid'], fingerprint)
        assert x is not None and x['uid'] == r['uid']
        assert sorted([x['a'],x['b']]) == sorted([r['a'],r['b']])
        data.append(x)
        if (i+1)%2000 == 0:print(json.dumps({'verified':i+1}),flush=True)
    heads = json.loads((ROOT/'results/heads.json').read_text())
    baseline = json.loads((ROOT/'data/baseline.json').read_text())
    x = np.array([r['true'] for r in data]); null = np.array([r['shuffled'] for r in data])
    gate = np.array([r['gate'] for r in data]); eligible = gate > 0
    train = np.array([r['split']=='train' for r in rows]); labels=np.array([r['label'] for r in rows])
    report = {'at_utc':now(), 'fingerprint':fingerprint, 'script_sha256':sha(__file__),
              'verified_records':len(data), 'heads_refitted':False,
              'posthoc_diagnostics_only':True, 'frozen_results_changed':False}
    h = heads['true']; scale=np.array(h['standard_scale']); mean=np.array(h['standard_mean'])
    pca=np.array(h['pca_components']); pm=np.array(h['pca_mean'])
    standardized=(x-mean)/scale; standardized_null=(null-mean)/scale
    delta=(x-null)/scale
    retained=delta@pca.T
    report['pca_train_total_variance_retained']=float(np.var((standardized[train&eligible]-pm)@pca.T,axis=0).sum()/np.var(standardized[train&eligible],axis=0).sum())
    report['pairing_change_energy_retained_by_true_head_pca']={name:float(np.square(retained[mask]).sum()/np.square(delta[mask]).sum()) for name,mask in [('train',train&eligible),('dev',~train&eligible)]}
    report['exactly_identical_true_shuffled_features_eligible']=int(np.all(x[eligible]==null[eligible],axis=1).sum())
    def logits(z):return (((z-pm)@pca.T)@np.array(h['coef']).T+np.array(h['intercept'])).ravel()
    ev=logits(standardized); en=logits(standardized_null)
    report['true_head_score_correlation_true_vs_shuffled_eligible_dev']=float(np.corrcoef(ev[~train&eligible],en[~train&eligible])[0,1])
    report['feature_standardized_rms_pairing_change_eligible_dev']=float(np.sqrt(np.square(delta[~train&eligible]).mean()))
    report['same_true_head_with_shuffled_input']={}
    base=np.array([baseline[r['uid']]['score'] if r['split']=='val' else 0 for r in rows])
    true=base.copy();shuffled=base.copy()
    true[eligible]+=h['alpha']*gate[eligible]*ev[eligible]
    shuffled[eligible]+=h['alpha']*gate[eligible]*en[eligible]
    for role in ['assessment','calibration','all_dev']:
        mask=np.array([r['split']=='val' if role=='all_dev' else r['role']==role for r in rows])
        report['same_true_head_with_shuffled_input'][role]={k:{'ap':float(average_precision_score(labels[mask],s[mask])), 'auroc':float(roc_auc_score(labels[mask],s[mask]))} for k,s in [('true',true),('shuffled_input',shuffled)]}
    # A shuffled index does not necessarily change an amino-acid sequence. This is
    # checked on real tensors separately in gpu_checks.py, not inferred here.
    report['null_changed_row_fraction']={name:float(np.mean([r['null']['changed_fraction'] for r,keep in zip(data,mask) if keep])) for name,mask in [('train',train&eligible),('dev',~train&eligible)]}
    report['fusion_alpha_at_upper_grid_boundary']={name:h['alpha']==max(x['alpha'] for x in h['calibration_grid']) for name,h in heads.items()}
    report['covered_dev_score_scale']={'baseline_std':float(base[~train&eligible].std()),
                                     'gated_true_evidence_std':float((gate*ev)[~train&eligible].std())}
    report['seconds']=time.monotonic()-started
    atomic(HERE/'cpu.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
