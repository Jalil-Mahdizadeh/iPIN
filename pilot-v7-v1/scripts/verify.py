"""Independently reconstruct every VAL score and every cached contact summary."""
import time
import numpy as np
from sklearn.metrics import average_precision_score
from study import ROOT, read, config, verify_freeze, load_record, evidence, fuse, sha, atomic, now
from contact_features import all_views
from heads import quality_features


def main():
    started=time.monotonic(); fp=verify_freeze(); cfg=config(); sample=read(ROOT/'data/sample.json')
    meta=read(ROOT/'data/source_metadata.json'); diag=read(ROOT/'data/diagnostics.json')
    records=[]; maximum=0.
    for r,m in zip(sample,meta):
        v=load_record(r['uid'],fp,maps=True)
        if v is None or any(v[k]!=m[k] for k in ['uid','a','b','gate','available','reason']):raise ValueError('Incomplete/changed record')
        if v['available']:
            with np.load(ROOT/v['map_path'],allow_pickle=False) as f:
                for kind in ['true','shuffled']:
                    views,_=all_views(f[kind],f['valid_a'],f['valid_b'],r['uid'],v['a'],v['b'])
                    for name,arm in [('standard',kind),('coordinate','coordinate_'+kind),('boundary','boundary_'+kind)]:
                        error=float(np.max(np.abs(views[name]-v['features'][arm]))); maximum=max(maximum,error)
                        if error>1e-12:raise ValueError('Cached map does not reproduce features')
        records.append(v)
    dev=np.array([r['split']=='val' for r in sample]); fit=np.array([r['split']=='train' and m['available'] for r,m in zip(sample,meta)])
    gate=np.array([m['gate'] for m in meta])[dev]; rows=[r for r,m in zip(sample,dev) if m]
    baseline=read(ROOT/'data/baseline.json'); base=np.array([baseline[r['uid']]['score'] for r in rows])
    old=read(ROOT/'data/source_heads.json'); states=read(ROOT/'results/heads.json')
    with np.load(ROOT/'data/source.npz',allow_pickle=False) as f:source={k:f[k].copy() for k in f.files}
    expected={'native':base}
    for kind,name in [('true','vx'),('shuffled','vx_shuffled'),('quality','vx_quality')]:
        expected[name]=fuse(base,evidence(old[kind],source[kind][dev]),gate,old[kind]['alpha'])
    q=quality_features(source['quality'],sample,meta,diag)[dev]; h=states['quality_length']
    z=evidence(h['model']['full'],q)
    expected['quality_length']=fuse(base,z,gate,h['native_fusion']['alpha'])
    expected['combined_quality']=fuse(expected['vx'],z,gate,h['vx_fusion']['alpha'])
    vectors={};full_vectors={'quality_length':quality_features(source['quality'],sample,meta,diag),'vx_oof':source['true']}
    for arm in cfg['arms']:
        full_vectors[arm]=np.array([v['features'][arm] for v in records]);vectors[arm]=full_vectors[arm][dev]; h=states[arm]
        z=evidence(h['model']['full'],vectors[arm])
        expected['contact_'+arm]=fuse(base,z,gate,h['native_fusion']['alpha'])
        expected['combined_'+arm]=fuse(expected['vx'],z,gate,h['vx_fusion']['alpha'])
    h=states['true']
    for arm,label in [('shuffled','shuffled'),('coordinate_true','coordinate'),('boundary_true','boundary')]:
        expected['combined_true_on_'+label]=fuse(expected['vx'],evidence(h['model']['full'],vectors[arm]),gate,h['vx_fusion']['alpha'])
    folds=read(ROOT/'data/folds.json'); fold=np.array(folds['folds']); pair_groups=np.array(folds['groups'])
    all_y=np.array([r['label'] for r in sample]);all_gate=np.array([m['gate'] for m in meta]);native_train=np.zeros(len(sample))
    with np.load(ROOT/'cache/native_train.npz',allow_pickle=False) as f:native_train[f['indices']]=f['scores']
    oofs={}
    for name,state in states.items():
        model=state if name=='vx_oof' else state['model']
        if model['full']['fit_indices']!=np.flatnonzero(fit).tolist():raise ValueError('Full head used non-TRAIN rows')
        np.testing.assert_allclose(model['full']['standard_mean'],full_vectors[name][fit].mean(0),atol=1e-12,rtol=1e-12)
        oofs[name]=np.full(len(sample),np.nan)
        for s in model['folds']:
            fi=np.asarray(s['head']['fit_indices']); hi=np.asarray(s['held_indices'])
            if not fit[fi].all() or not fit[hi].all() or set(pair_groups[fi])&set(pair_groups[hi]):raise ValueError('OOF leakage')
            if not np.all(fold[hi]==s['fold']) or np.any(fold[fi]==s['fold']):raise ValueError('OOF fold mismatch')
            np.testing.assert_allclose(s['head']['standard_mean'],full_vectors[name][fi].mean(0),atol=1e-12,rtol=1e-12)
            oofs[name][hi]=evidence(s['head'],full_vectors[name][hi])
        if name!='vx_oof':
            for f in ['native_fusion','vx_fusion']:
                if state[f]['calibration_indices']!=np.flatnonzero(fit).tolist():raise ValueError('Calibration used non-TRAIN rows')
                grid=state[f]['eligible_train_oof_grid']; best=max(v['ap'] for v in grid)
                if state[f]['alpha']!=min(v['alpha'] for v in grid if v['ap']==best):raise ValueError('Fusion choice mismatch')
    for name,state in states.items():
        if name=='vx_oof':continue
        for fusion in ['native_fusion','vx_fusion']:
            b=native_train[fit].copy()
            if fusion=='vx_fusion':b+=old['true']['alpha']*all_gate[fit]*oofs['vx_oof'][fit]
            for item in state[fusion]['eligible_train_oof_grid']:
                z=b+item['alpha']*all_gate[fit]*oofs[name][fit]
                actual=average_precision_score(all_y[fit],z)
                if abs(actual-item['ap'])>1e-12:raise ValueError('Recorded TRAIN calibration AP does not reconstruct')
    with np.load(ROOT/'results/val_predictions.npz',allow_pickle=False) as f:
        if f['uids'].tolist()!=[r['uid'] for r in rows]:raise ValueError('VAL prediction order mismatch')
        np.testing.assert_array_equal(f['labels'],[r['label'] for r in rows])
        if set(f.files)!=(set(expected)|{'uids','labels'}):raise ValueError('Unexpected output models')
        for name,z in expected.items():
            np.testing.assert_allclose(z,f[name],atol=1e-10,rtol=1e-10)
            np.testing.assert_array_equal(f[name][gate==0],base[gate==0])
    ass=np.array([r['role']=='assessment' for r in rows]); y=np.array([r['label'] for r in rows]); metrics=read(ROOT/'results/metrics.json')
    aps={n:float(average_precision_score(y[ass],z[ass])) for n,z in expected.items()}
    for n,ap in aps.items():
        if abs(ap-metrics['results']['assessment']['metrics'][n]['ap'])>1e-12:raise ValueError('Reported AP mismatch')
    point=aps['combined_true']-aps['vx']
    if abs(point-read(ROOT/'results/decision.json')['primary_point'])>1e-12:raise ValueError('Decision contrast mismatch')
    # Independently reproduce the primary interval through sklearn's weighted AP.
    ids=sorted({r[k] for r in rows for k in ['a','b']});lookup={p:i for i,p in enumerate(ids)}
    ia=np.array([lookup[r['a']] for r in rows]);ib=np.array([lookup[r['b']] for r in rows]);values=[]
    rng=np.random.default_rng(cfg['seed'])
    for _ in range(cfg['evaluation']['bootstrap_replicates']):
        counts=rng.poisson(1,len(ids));w=(counts[ia]*counts[ib])[ass]
        if not w[y[ass]==0].sum() or not w[y[ass]==1].sum():continue
        values.append(average_precision_score(y[ass],expected['combined_true'][ass],sample_weight=w)-average_precision_score(y[ass],expected['vx'][ass],sample_weight=w))
    interval=metrics['results']['assessment']['ap_intervals']['combined_true_minus_vx']
    if len(values)!=interval['replicates']:raise ValueError('Bootstrap replicate count mismatch')
    if len(values)>=.95*cfg['evaluation']['bootstrap_replicates']:
        np.testing.assert_allclose(np.quantile(values,[.025,.975]),[interval['low'],interval['high']],atol=1e-12,rtol=1e-12)
    result={'at_utc':now(),'fingerprint':fp,'passed':True,'records':len(records),'maps_checked':sum(v['available'] for v in records)*2,
        'maximum_feature_reconstruction_error':maximum,'all_scores_reconstructed':True,'exact_fallback':True,
        'train_only_fit_and_calibration':True,'all_oof_calibration_grids_reconstructed':True,'primary_interval_independently_verified':True,
        'test_accessed':False,'seconds':time.monotonic()-started,
        'metrics_sha256':sha(ROOT/'results/metrics.json'),'heads_sha256':sha(ROOT/'results/heads.json')}
    atomic(ROOT/'results/verification.json',result); print(result,flush=True)


if __name__=='__main__':main()
