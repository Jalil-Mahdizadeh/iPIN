"""Compare validation-selected v2 models on matched development data only."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from state import sha256, atomic_json
from data import PairData

ROOT=Path(__file__).resolve().parents[1]
def load(run):
    run=Path(run).resolve()
    assert run.is_relative_to(ROOT/'runs'),'Use production development runs only'
    assert (run/'completed.json').exists(),'Wait for the prespecified horizon before comparing arms'
    contract=json.loads((run/'contract.json').read_text())
    assert not contract['qualification'] and contract['configuration']['selection_metric']=='pooled_ap'
    best=json.loads((run/'best.json').read_text())
    path=run/'validation'/f'update-{best["update"]:09d}.npz'
    metadata=json.loads(path.with_suffix('.json').read_text())
    assert sha256(path)==metadata['sha256'] and metadata['fingerprint']==contract['fingerprint']
    with np.load(path) as f:predictions=f['predictions']
    data=PairData(ROOT/'data/prepared',contract['configuration']['partition'],'val')
    assert len(predictions)==len(data)
    assert np.array_equal(predictions[:,0],np.arange(len(data))) and np.array_equal(predictions[:,1],data.rows[:,2])
    return contract,predictions,data,best
def metrics(y,s,weight=None):
    return {'ap':float(average_precision_score(y,s,sample_weight=weight)),
            'auroc':float(roc_auc_score(y,s,sample_weight=weight)),
            'brier':float(np.average((expit(s)-y)**2,weights=weight))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--control',required=True);p.add_argument('--candidate',required=True)
    p.add_argument('--output',required=True);p.add_argument('--bootstrap',type=int,default=1000);args=p.parse_args()
    assert 200<=args.bootstrap<=10000
    ca,pa,data,ba=load(args.control);cb,pb,db,bb=load(args.candidate)
    assert np.array_equal(data.rows,db.rows)
    for key in ['partition','seed','total_updates','global_pairs_per_update','warmup_updates','initialization',
                'classification_weight','validate_every_updates','selection_metric']:
        assert ca['configuration'][key]==cb['configuration'][key],('Unmatched control',key)
    assert ca['data_manifest_sha256']==cb['data_manifest_sha256']==sha256(ROOT/'data/prepared/manifest.json')
    y=pa[:,1];a=pa[:,2:4].mean(1);b=pb[:,2:4].mean(1)
    ma,mb=metrics(y,a),metrics(y,b)
    strata={}
    for name,mask in [('up-to-2193',data.lengths<=2196),('over-2193',data.lengths>2196)]:
        if len(np.unique(y[mask]))==2:
            strata[name]={'rows':int(mask.sum()),'prevalence':float(y[mask].mean()),
                          'control':metrics(y[mask],a[mask]),'candidate':metrics(y[mask],b[mask])}
    # Poisson node resampling: endpoint-count products preserve the pairing of
    # models while acknowledging repeated proteins. Conditional graph estimate.
    proteins,pair_indices=np.unique(data.rows[:,:2],return_inverse=True)
    pair_indices=pair_indices.reshape(-1,2)
    rng=np.random.default_rng(20260929);deltas=[]
    for _ in range(args.bootstrap):
        multiplicity=rng.poisson(1.,len(proteins))
        weight=multiplicity[pair_indices[:,0]]*multiplicity[pair_indices[:,1]]
        if np.unique(y[weight>0]).size<2:continue
        deltas.append(average_precision_score(y,b,sample_weight=weight)-average_precision_score(y,a,sample_weight=weight))
    assert len(deltas)>=.95*args.bootstrap
    retrieval=[]
    for protein in proteins:
        mask=(data.rows[:,:2]==protein).any(1)
        if np.unique(y[mask]).size==2:
            retrieval.append([average_precision_score(y[mask],a[mask]),average_precision_score(y[mask],b[mask])])
    report={'control':args.control,'candidate':args.candidate,'partition':ca['configuration']['partition'],
        'control_best_update':ba['update'],'candidate_best_update':bb['update'],
        'rows':len(y),'prevalence':float(y.mean()),'control_metrics':ma,'candidate_metrics':mb,
        'deltas':{k:mb[k]-ma[k] for k in ma},'length_strata':strata,
        'protein_bootstrap_ap_delta_95_ci':np.quantile(deltas,[.025,.975]).tolist(),
        'bootstrap_replicates':len(deltas),'macro_partner_retrieval_ap':{
            'proteins_with_both_labels':len(retrieval),'control':float(np.mean(retrieval,axis=0)[0]),
            'candidate':float(np.mean(retrieval,axis=0)[1])},
        'inference':'BF16, mean of both orientation logits',
        'interpretation':'Development-only selected-checkpoint comparison; uncertainty conditions on this graph and excludes training-seed and model-selection uncertainty.',
        'test_evaluated':False}
    atomic_json(Path(args.output),report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
