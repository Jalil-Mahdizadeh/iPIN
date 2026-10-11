"""Prespecified AP contrasts and matched-protein bootstrap; no model selection."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score

CONTRASTS = {
    'combined_true_minus_vx':{'combined_true':1,'vx':-1},
    'combined_true_minus_combined_quality':{'combined_true':1,'combined_quality':-1},
    'combined_true_minus_native':{'combined_true':1,'native':-1},
    'contact_true_minus_vx':{'contact_true':1,'vx':-1},
    'contact_true_minus_native':{'contact_true':1,'native':-1},
    'contact_true_minus_quality':{'contact_true':1,'quality_length':-1},
    'combined_true_minus_shuffled':{'combined_true':1,'combined_shuffled':-1},
    'combined_same_head_pairing_gap':{'combined_true':1,'combined_true_on_shuffled':-1},
    'combined_true_minus_coordinate':{'combined_true':1,'combined_coordinate_true':-1},
    'combined_same_head_coordinate_gap':{'combined_true':1,'combined_true_on_coordinate':-1},
    'combined_boundary_minus_vx':{'combined_boundary_true':1,'vx':-1},
    'combined_fixed_head_boundary_minus_vx':{'combined_true_on_boundary':1,'vx':-1}
}


def contrasts(aps):
    return {k:float(sum(w*aps[n] for n,w in terms.items())) for k,terms in CONTRASTS.items()}


def ap_setup(y,z):
    order=np.argsort(-z,kind='stable'); ends=np.r_[np.flatnonzero(np.diff(z[order])),len(z)-1]
    return order,ends,y[order]


def weighted_ap(setup,weights):
    order,ends,y=setup; w=weights[order]
    tp=np.cumsum(w*y)[ends]; total=np.cumsum(w)[ends]
    if not tp[-1] or total[-1]==tp[-1]:
        return np.nan
    precision=np.divide(tp,total,out=np.zeros_like(tp,dtype=float),where=total>0)
    return float(np.sum(np.diff(np.r_[0,tp])*precision)/tp[-1])


def summarize(mask,y,scores,available):
    result={'rows':int(mask.sum()),'positive':int(y[mask].sum()),'eligible':int((mask&available).sum())}
    if len(np.unique(y[mask]))!=2:
        return {**result,'status':'insufficient_classes'}
    yy=y[mask]; counts=np.bincount(yy,minlength=2); weights=.5/counts[yy]
    result['metrics']={n:{'ap':float(average_precision_score(yy,z[mask])),
        'auroc':float(roc_auc_score(yy,z[mask])),
        'ap_at_50_percent_prevalence':float(average_precision_score(yy,z[mask],sample_weight=weights))} for n,z in scores.items()}
    result['contrasts']=contrasts({n:v['ap'] for n,v in result['metrics'].items()})
    return result


def bootstrap(rows,y,scores,masks,cfg,seed):
    ids=sorted({r[k] for r in rows for k in ['a','b']}); lookup={p:i for i,p in enumerate(ids)}
    ia=np.array([lookup[r['a']] for r in rows]); ib=np.array([lookup[r['b']] for r in rows])
    accepted={k:m for k,m in masks.items() if min(np.sum(m&(y==c)) for c in [0,1])>=cfg['minimum_subgroup_per_class']}
    setups={k:{n:ap_setup(y[m],z[m]) for n,z in scores.items()} for k,m in accepted.items()}
    samples={k:{n:[] for n in CONTRASTS} for k in accepted}; rng=np.random.default_rng(seed)
    for _ in range(cfg['bootstrap_replicates']):
        count=rng.poisson(1,len(ids)); weights=count[ia]*count[ib]
        for group,m in accepted.items():
            aps={n:weighted_ap(s,weights[m]) for n,s in setups[group].items()}
            for name,value in contrasts(aps).items():
                if np.isfinite(value):samples[group][name].append(value)
    out={}
    for group,cs in samples.items():
        out[group]={}
        for name,values in cs.items():
            rec={'replicates':len(values)}
            if len(values)>=.95*cfg['bootstrap_replicates']:
                rec.update({'low':float(np.quantile(values,.025)),'high':float(np.quantile(values,.975))})
            else:rec['status']='insufficient_valid_replicates'
            out[group][name]=rec
    return out
