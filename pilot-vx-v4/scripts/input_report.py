"""Input summaries only; no diagnostic is used to select or fit the model."""
import numpy as np


def flatten(value,prefix=''):
    out={}
    for key,v in value.items():
        name=prefix+key
        if isinstance(v,dict):out.update(flatten(v,name+'/'))
        elif isinstance(v,(int,float)) and not isinstance(v,bool):out[name]=v
        elif v is None:out[name]=None
    return out


def aggregate(audits,sample,pairs,masks):
    eligible={p['uid']:p for p in pairs if p['available']}
    if set(audits)!=set(eligible):raise ValueError('Incomplete input diagnostics')
    for uid,a in audits.items():
        if not a['exact_query_PAD_depth'] or not a['intact_source_rows'] or a['labels_used'] or a['depth_including_query']!=eligible[uid]['paired_depth']:
            raise ValueError('Input invariants failed')
    flat={k:flatten(v) for k,v in audits.items()};groups={}
    for group,mask in masks.items():
        chosen=[flat[r['uid']] for r,keep in zip(sample,mask) if keep and r['uid'] in flat]
        metrics={}
        for key in sorted({k for v in chosen for k in v}):
            a=np.array([v[key] for v in chosen if v.get(key) is not None],dtype=float)
            if not np.isfinite(a).all():raise ValueError('Nonfinite diagnostic')
            metrics[key]={'pairs':len(a),**({'mean':float(a.mean()),'min':float(a.min()),'q25':float(np.quantile(a,.25)),'median':float(np.median(a)),'q75':float(np.quantile(a,.75)),'max':float(a.max())} if len(a) else {})}
        groups[group]={'eligible_pairs':len(chosen),'metrics':metrics}
    return {'groups':groups,'pair_weighting':'equal weight per eligible pair','labels_used_for_sampling':False,
            'used_for_model_selection':False,'R_evaluated':False,'independent_sampling_changes_composition_and_diversity':True}
