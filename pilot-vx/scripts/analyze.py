"""One frozen TRAIN fit and DEV calibration/assessment; no test inputs."""
import json
import warnings
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from common import ROOT, atomic, config, now, sha, verify_freeze
from features import load_record


def fuse(base,evidence,gate,alpha):
    base=np.asarray(base,float);out=base.copy();active=np.asarray(gate)>0
    # Explicit assignment, not multiplication by zero: missing evidence cannot propagate NaN.
    if alpha:
        if not np.isfinite(np.asarray(evidence)[active]).all():raise ValueError('Nonfinite available evidence')
        out[active]+=alpha*np.asarray(gate)[active]*np.asarray(evidence)[active]
    return out


def metrics(y,z,weights=None):
    return {'ap':float(average_precision_score(y,z,sample_weight=weights)),
            'auroc':float(roc_auc_score(y,z,sample_weight=weights))}


def intervals(rows,y,scores,seed,repeats):
    ids=sorted({r[k] for r in rows for k in ['a','b']});lookup={x:i for i,x in enumerate(ids)}
    ia=np.array([lookup[r['a']] for r in rows]);ib=np.array([lookup[r['b']] for r in rows])
    rng=np.random.default_rng(seed);pairs=[('true','baseline'),('true','shuffled'),('true','quality')]
    values={a+'-minus-'+b:[] for a,b in pairs}
    for _ in range(repeats):
        counts=rng.poisson(1,len(ids));w=counts[ia]*counts[ib]
        if not all(w[y==c].sum()>0 for c in [0,1]):continue
        ap={k:float(average_precision_score(y,z,sample_weight=w)) for k,z in scores.items()}
        for a,b in pairs:values[a+'-minus-'+b].append(ap[a]-ap[b])
    return {k:{'low':float(np.quantile(v,.025)),'high':float(np.quantile(v,.975)),'replicates':len(v)} for k,v in values.items()}


def main():
    cfg=config();fingerprint=verify_freeze();sample=json.loads((ROOT/'data/sample.json').read_text())
    data=[];missing=[]
    for r in sample:
        x=load_record(ROOT/'features',r['uid'],fingerprint)
        if x is None:missing.append(r['uid'])
        data.append(x)
    if missing:
        atomic(ROOT/'results/decision.json',{'at_utc':now(),'status':'inconclusive_incomplete','expected_pairs':len(sample),
               'completed_pairs':len(sample)-len(missing),'missing_pairs':missing,'fitting_performed':False})
        (ROOT/'results/REPORT.md').write_text(f'# Bernett Vx pilot\n\nInconclusive: {len(missing)} of {len(sample)} computational outputs are missing. No head was fitted and no incomplete subset was substituted for the fixed sample. See decision.json.\n')
        print(json.dumps({'status':'inconclusive_incomplete','missing':len(missing)}),flush=True);return
    labels=np.array([r['label'] for r in sample]);gate=np.array([r['gate'] for r in data])
    train=np.array([r['split']=='train' for r in sample]);dev=~train
    cal=np.array([r['role']=='calibration' for r in sample]);ass=np.array([r['role']=='assessment' for r in sample])
    fit=train&(gate>0)
    requirements={name:[int(np.sum(mask&(labels==c))) for c in [0,1]] for name,mask in [('fit',fit),('calibration',cal),('assessment',ass)]}
    if min(requirements['fit'])<100 or min(requirements['calibration'])<50 or min(requirements['assessment'])<50:
        atomic(ROOT/'results/decision.json',{'at_utc':now(),'status':'inconclusive_insufficient_evidence','counts':requirements,'fitting_performed':False})
        (ROOT/'results/REPORT.md').write_text('# Bernett Vx pilot\n\nInconclusive: too few eligible TRAIN examples or too few examples in an internal DEV group. No head was fitted. See decision.json for exact class counts.\n');return
    baseline=json.loads((ROOT/'data/baseline.json').read_text());base=np.zeros(len(sample))
    for i,r in enumerate(sample):
        if dev[i]:base[i]=baseline[r['uid']]['score']
    predictions={'baseline':base};alphas={};fits={};warnings.simplefilter('error',ConvergenceWarning)
    for kind in ['true','shuffled','quality']:
        x=np.asarray([r[kind] for r in data],float)
        pipe=make_pipeline(StandardScaler(),PCA(n_components=cfg['fusion']['pca_components'],svd_solver='full'),
                           LogisticRegression(C=cfg['fusion']['C'],max_iter=1000,solver='lbfgs',random_state=cfg['seed']))
        pipe.fit(x[fit],labels[fit]);evidence=pipe.decision_function(x)
        choices=[]
        for alpha in cfg['fusion']['alpha_grid']:
            z=fuse(base,evidence,gate,alpha);choices.append((float(average_precision_score(labels[cal],z[cal])),alpha))
        best=max(v for v,a in choices);alpha=min(a for v,a in choices if v==best)
        z=fuse(base,evidence,gate,alpha);assert np.array_equal(z[dev&(gate==0)],base[dev&(gate==0)])
        predictions[kind]=z;alphas[kind]=alpha
        sc,pc,cl=pipe.steps[0][1],pipe.steps[1][1],pipe.steps[2][1]
        fits[kind]={'standard_mean':sc.mean_.tolist(),'standard_scale':sc.scale_.tolist(),
                    'pca_mean':pc.mean_.tolist(),'pca_components':pc.components_.tolist(),
                    'coef':cl.coef_.tolist(),'intercept':cl.intercept_.tolist(),'classes':cl.classes_.tolist(),
                    'calibration_grid':[{'alpha':a,'ap':v} for v,a in choices],'alpha':alpha}
    atomic(ROOT/'results/heads.json',fits)
    results={}
    for name,mask in [('assessment',ass),('fixed_dev_sample',dev),('calibration',cal)]:
        subset=[r for r,keep in zip(sample,mask) if keep];yy=labels[mask];ss={k:z[mask] for k,z in predictions.items()}
        results[name]={'rows':int(mask.sum()),'positive':int(yy.sum()),'covered':int((mask&(gate>0)).sum()),
                       'metrics':{k:metrics(yy,z) for k,z in ss.items()}}
        if name!='calibration':results[name]['ap_intervals']=intervals(subset,yy,ss,cfg['seed'],cfg['fusion']['bootstrap_replicates'])
    strata={}
    for lo,hi in [(0,512),(512,1024),(1024,1536),(1536,2048),(2048,10**9)]:
        mask=dev&np.array([lo<r['length']<=hi for r in sample]);yy=labels[mask]
        strata[f'{lo+1}-{hi}']={'rows':int(mask.sum()),'covered':int((mask&(gate>0)).sum())}
        if len(np.unique(yy))==2:strata[f'{lo+1}-{hi}']['metrics']={k:metrics(yy,z[mask]) for k,z in predictions.items()}
    assessment=results['assessment'];m=assessment['metrics'];ci=assessment['ap_intervals'];delta=m['true']['ap']-m['baseline']['ap']
    changed=[x['null']['changed_fraction'] for x,keep in zip(data,ass) if keep and x['available']]
    null_levels={level:sum(x['null'].get('level_counts',{}).get(level,0) for x,keep in zip(data,ass) if keep and x['available']) for level in ['family','order','class','unchanged']}
    pairing_null_adequate=bool(changed) and float(np.mean(np.array(changed)>=.5))>=.8
    passes=delta>=cfg['fusion']['continuation_delta_ap'] and ci['true-minus-baseline']['low']>0 and \
           ci['true-minus-shuffled']['low']>0 and ci['true-minus-quality']['low']>0 and \
           m['true']['auroc']>=m['baseline']['auroc']-.005 and pairing_null_adequate
    atomic(ROOT/'results/metrics.json',{'at_utc':now(),'fingerprint':fingerprint,'results':results,'alphas':alphas,'length_strata':strata,
        'fit_counts':requirements,'assessment_null_row_counts_by_taxonomic_level':null_levels,
        'bootstrap':'matched protein Poisson-multiplier weights, edge weight product',
        'structural_heads_used':False,'test_used':False})
    atomic(ROOT/'results/decision.json',{'at_utc':now(),'status':'statistical_criteria_met_review_family_breadth_and_cost' if passes else 'no_go_or_inconclusive',
        'assessment_delta_ap':delta,'pairing_null_adequate':pairing_null_adequate,
        'mean_null_changed_fraction':float(np.mean(changed)) if changed else None,'production_authorized':False,
        'metrics_sha256':sha(ROOT/'results/metrics.json')})
    lines=['# Bernett Vx pilot results','',f"Assessment AP change: {delta:+.5f}. Status: {'statistical criteria met; review breadth/cost' if passes else 'no-go or inconclusive'}.",'',
           'No TEST labels, TEST predictions, or released structural contact heads were used.','',
           '| Population | Model | AP | AUROC | Covered / rows |','|---|---|---:|---:|---:|']
    for name,item in results.items():
        for model,v in item['metrics'].items():lines.append(f"| {name} | {model} | {v['ap']:.5f} | {v['auroc']:.5f} | {item['covered']} / {item['rows']} |")
    lines+=['','The fixed DEV sample has 4,000 rows; it is not the complete 59,260-row DEV dataset. Calibration results are selection results. All unsupported examples retain the native baseline.','',
            'See metrics.json for matched protein-bootstrap intervals, strata and fusion coefficients. Family breadth and production cost require review before any continuation.']
    (ROOT/'results/REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'assessment_delta_ap':delta,'passes_statistical_checks':passes}),flush=True)


if __name__=='__main__':main()
