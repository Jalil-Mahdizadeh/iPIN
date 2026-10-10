"""CPU source audit and independent numerical reconstruction; classifiers never refit."""
import time
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
from study import ROOT,config,read,atomic,now,sha,verify_freeze,DIMENSIONS
from analyze import load_complete
from inputs import verify_input_population
from statistics_v5 import CONTRASTS
from decision import decide


def manual(h,x):
    standardized=(np.asarray(x)-np.asarray(h['standard_mean']))/np.asarray(h['standard_scale'])
    projected=(standardized-np.asarray(h['pca_mean'])).dot(np.asarray(h['pca_components']).T)
    return projected.dot(np.asarray(h['coef']).ravel())+h['intercept'][0]


def main():
    started=time.monotonic();cfg=config();fp=verify_freeze();sample,pairs,x=load_complete(fp)
    y=np.array([r['label'] for r in sample]);gate=np.array([p['gate'] for p in pairs]);available=gate>0
    train=np.array([r['split']=='train' for r in sample]);dev=~train;fit=train&available
    cal=np.array([r['role']=='calibration' for r in sample]);ass=np.array([r['role']=='assessment' for r in sample])
    model=read(ROOT/'results/heads.json');metrics=read(ROOT/'results/metrics.json');decision=read(ROOT/'results/decision.json')
    if any(v['fingerprint']!=fp for v in (model,metrics,decision)):raise ValueError('Result fingerprint differs')
    if model['fit_pair_uids']!=[r['uid'] for r,k in zip(sample,fit) if k] or model['TRAIN_baseline_used']:raise ValueError('Wrong TRAIN population')
    for arm,dim in DIMENSIONS.items():
        h=model[arm];sc=StandardScaler().fit(x[arm][fit]);sx=sc.transform(x[arm][fit]);pc=np.asarray(h['pca_components'])
        np.testing.assert_allclose(sc.mean_,h['standard_mean'],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(sc.scale_,h['standard_scale'],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(sx.mean(0),h['pca_mean'],atol=1e-12,rtol=1e-12)
        np.testing.assert_allclose(pc@pc.T,np.eye(len(pc)),atol=1e-10,rtol=1e-10)
        ratio=((sx-sx.mean(0))@pc.T).var(0,ddof=1)/sx.var(0,ddof=1).sum()
        np.testing.assert_allclose(ratio,h['pca_explained_variance_ratio'],atol=1e-10,rtol=1e-10)
        if pc.shape!=(cfg['fusion']['pca_components'],dim) or h['fit_rows']!=int(fit.sum()) or h['classes']!=[0,1]:raise ValueError('Wrong head dimensions/classes')
    baseline=read(ROOT/'data/baseline.json');base=np.array([baseline[r['uid']]['score'] if r['split']=='val' else 0. for r in sample])
    with np.load(ROOT/'data/source.npz',allow_pickle=False) as f:old={k:f[k].copy() for k in f.files}
    if old['uids'].tolist()!=[r['uid'] for r in sample]:raise ValueError('Source order changed')
    np.testing.assert_array_equal(old['gate'],gate)
    oldmeta=read(ROOT/'data/source_metadata.json')
    for j,p in enumerate(pairs):
        if p['available']:
            expected=old['quality'][j].copy();expected[-3]=np.log1p(p['msa256']['neff'])
            np.testing.assert_array_equal(expected,x['P256'][j])
            if p['msa256']['gate']!=oldmeta[j]['msa']['gate']:raise ValueError('Natural gate changed')
    heads={**read(ROOT/'data/source_heads.json'),**{a:model[a] for a in DIMENSIONS}}
    scores={'baseline':base};evidence={};alphas={}
    for arm,h in heads.items():
        ev=manual(h,x[arm] if arm in DIMENSIONS else old[arm]);evidence[arm]=ev
        if not np.isfinite(ev[available]).all():raise ValueError('Nonfinite evidence')
        z=base.copy();z[available]+=h['alpha']*gate[available]*ev[available];scores[arm]=z
        np.testing.assert_array_equal(z[~available],base[~available])
        if arm in DIMENSIONS:
            choices=[]
            for alpha in cfg['fusion']['alpha_grid']:
                zz=base.copy();zz[available]+=alpha*gate[available]*ev[available]
                choices.append((alpha,float(average_precision_score(y[cal],zz[cal]))))
            best=max(ap for _,ap in choices);selected=min(a for a,ap in choices if ap==best)
            if selected!=h['alpha']:raise ValueError('Incorrect calibration choice')
            np.testing.assert_allclose([ap for _,ap in choices],[v['ap'] for v in h['calibration_grid']],atol=1e-12,rtol=1e-12)
            alphas[arm]=selected
    with np.load(ROOT/'results/dev_predictions.npz',allow_pickle=False) as f:saved={k:f[k].copy() for k in f.files}
    if saved['uids'].tolist()!=[r['uid'] for r,k in zip(sample,dev) if k]:raise ValueError('Saved order changed')
    np.testing.assert_array_equal(saved['labels'],y[dev]);np.testing.assert_array_equal(saved['gate'],gate[dev]);errors={}
    for arm,z in scores.items():
        errors[arm]=float(np.max(np.abs(z[dev]-saved[arm])));np.testing.assert_allclose(z[dev],saved[arm],atol=1e-10,rtol=1e-10)
    with np.load(ROOT/'results/dev_evidence.npz',allow_pickle=False) as f:
        for arm,ev in evidence.items():np.testing.assert_allclose(f[arm],ev[dev],atol=1e-10,rtol=1e-10)
    with np.load(ROOT/'data/vy_dev_predictions.npz',allow_pickle=False) as f:
        if f['uids'].tolist()!=saved['uids'].tolist():raise ValueError('Historical reference order differs')
        np.testing.assert_array_equal(f['labels'],y[dev])
        for arm in ('baseline','true','shuffled','quality'):
            np.testing.assert_allclose(scores[arm][dev],f['baseline' if arm=='baseline' else 'vx_'+arm],atol=1e-10,rtol=1e-10)
    scopes={'assessment':ass,'assessment/eligible':ass&available,'assessment/fallback':ass&~available,
            'calibration':cal,'fixed_dev_sample':dev,'crossing_descriptive':np.array([r['role']=='crossing' for r in sample])}
    for pop,m in scopes.items():
        group=metrics['results'][pop]
        if group['rows']!=int(m.sum()) or group['positive']!=int(y[m].sum()):raise ValueError('Reported population differs')
        if len(np.unique(y[m]))<2:continue
        for arm,z in scores.items():
            for name,fn in [('ap',average_precision_score),('auroc',roc_auc_score)]:
                if abs(fn(y[m],z[m])-group['metrics'][arm][name])>1e-12:raise ValueError('Reported metric differs')
    rows=[r for r,k in zip(sample,ass) if k];yy=y[ass];ids=sorted({r[k] for r in rows for k in ('a','b')});lookup={p:i for i,p in enumerate(ids)}
    ia=np.array([lookup[r['a']] for r in rows]);ib=np.array([lookup[r['b']] for r in rows]);rng=np.random.default_rng(cfg['seed'])
    subsets={'assessment':np.ones(len(rows),dtype=bool),'assessment/eligible':available[ass]};values={pop:{name:[] for name in CONTRASTS} for pop in subsets}
    for _ in range(cfg['fusion']['bootstrap_replicates']):
        counts=rng.poisson(1,len(ids));w=counts[ia]*counts[ib]
        for pop,m in subsets.items():
            if any(w[m&(yy==c)].sum()==0 for c in (0,1)):continue
            aps={arm:average_precision_score(yy[m],z[ass][m],sample_weight=w[m]) for arm,z in scores.items()}
            for name,terms in CONTRASTS.items():values[pop][name].append(float(sum(c*aps[k] for k,c in terms.items())))
    intervals={}
    for pop,results in values.items():
        intervals[pop]={}
        for name,v in results.items():
            low,high=np.quantile(v,[.025,.975]);ref=metrics['results'][pop]['ap_intervals'][name]
            np.testing.assert_allclose([low,high],[ref['low'],ref['high']],atol=1e-10,rtol=1e-10)
            if len(v)!=ref['replicates']:raise ValueError('Bootstrap replicate count differs')
            intervals[pop][name]={'low':float(low),'high':float(high),'replicates':len(v)}
    previous=read(ROOT/'data/source_metrics.json')
    for name,oldname in [('true_minus_baseline','true-minus-baseline'),('true_minus_shuffled','true-minus-shuffled')]:
        c=intervals['assessment'][name];ref=previous['results']['assessment']['ap_intervals'][oldname]
        np.testing.assert_allclose([c['low'],c['high']],[ref['low'],ref['high']],atol=1e-10,rtol=1e-10)
    audit=verify_input_population(pairs)
    if audit['eligible_inputs_verified']!=int(available.sum()):raise ValueError('Incomplete input audit')
    if metrics['input_diagnostics_sha256']!=sha(ROOT/'results/input-diagnostics.json') or metrics['R_evaluated']:raise ValueError('Diagnostic report changed')
    if any(decision[k]!=v for k,v in decide(metrics['results'],cfg).items()) or decision['metrics_sha256']!=sha(ROOT/'results/metrics.json'):raise ValueError('Decision differs from frozen criteria')
    atomic(ROOT/'results/verification.json',{'at_utc':now(),'passed':True,'fingerprint':fp,'pairs_verified':len(sample),
        'required_pairs_verified':int(available.sum()),'complete_coverage':True,'alphas_independently_reproduced':alphas,
        'maximum_prediction_reconstruction_error':max(errors.values()),'prediction_errors':errors,
        'independent_sklearn_intervals':intervals,'TRAIN_only_transforms_verified':True,'exact_baseline_fallback':True,
        'original_predictions_and_intervals_reproduced':True,'depth_input_audit':audit,'heads_refitted':False,'test_accessed':False,'R_evaluated':False,
        'scientific_result_hashes':{name:sha(ROOT/'results'/name) for name in ['heads.json','metrics.json','decision.json','dev_predictions.npz','dev_evidence.npz','input-diagnostics.json']},
        'script_sha256':sha(ROOT/'scripts/verify.py'),'seconds':time.monotonic()-started})
    print({'independent_verification_passed':True,'seconds':time.monotonic()-started},flush=True)


if __name__=='__main__':main()
