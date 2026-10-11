"""TRAIN-only fitting/fusion followed by one fixed exploratory VAL assessment."""
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.metrics import average_precision_score
from study import ROOT, config, read, atomic, save_npz, sha, now, verify_freeze, load_record, evidence, fuse, remaining
from heads import crossfit, choose_alpha, quality_features
from statistics_v7 import CONTRASTS, summarize, bootstrap


def main():
    started=time.monotonic(); cfg=config(); fp=verify_freeze(); remaining(300)
    sample=read(ROOT/'data/sample.json'); previous=read(ROOT/'data/source_metadata.json'); diagnostics=read(ROOT/'data/diagnostics.json')
    with ThreadPoolExecutor(max_workers=8) as ex:
        records=list(ex.map(lambda r:load_record(r['uid'],fp,maps=True),sample))
    if any(v is None for v in records):raise ValueError('Incomplete extraction; no partial-population analysis')
    for r,v,m in zip(sample,records,previous):
        if any(v[k]!=m[k] for k in ['uid','a','b','gate','available','reason']):
            raise ValueError('Changed source population')
    y=np.array([r['label'] for r in sample]); gate=np.array([m['gate'] for m in previous]); available=gate>0
    train=np.array([r['split']=='train' for r in sample]); dev=~train; fit=train&available
    ass=np.array([r['role']=='assessment' for r in sample]); cal=np.array([r['role']=='calibration' for r in sample])
    if (fit.sum(),ass.sum(),cal.sum(),(dev&available).sum())!=(1825,958,1068,2628):raise ValueError('Original governance changed')
    baseline=read(ROOT/'data/baseline.json'); base=np.zeros(len(sample))
    for i,r in enumerate(sample):
        if r['split']=='val':
            b=baseline[r['uid']]
            if any(b[k]!=r[k] for k in ['a','b','label']):raise ValueError('Baseline identity mismatch')
            base[i]=b['score']
    native_meta=read(ROOT/'cache/native_train.json')
    if native_meta['fingerprint']!=fp or sha(ROOT/'cache/native_train.npz')!=native_meta['sha256']:raise ValueError('Native TRAIN provenance mismatch')
    with np.load(ROOT/'cache/native_train.npz',allow_pickle=False) as f:
        np.testing.assert_array_equal(f['indices'],np.flatnonzero(fit))
        if f['uids'].tolist()!=[sample[i]['uid'] for i in f['indices']]:raise ValueError('Native TRAIN score order mismatch')
        np.testing.assert_array_equal(f['scores'],f['orientations'].mean(1)); base[f['indices']]=f['scores']
    with np.load(ROOT/'data/source.npz',allow_pickle=False) as f: source={k:f[k].copy() for k in f.files}
    old=read(ROOT/'data/source_heads.json'); old_metrics=read(ROOT/'data/source_metrics.json')
    scores={'native':base}
    for kind,name in [('true','vx'),('shuffled','vx_shuffled'),('quality','vx_quality')]:
        scores[name]=fuse(base,evidence(old[kind],source[kind]),gate,old[kind]['alpha'])
        expected=old_metrics['results']['assessment']['metrics'][kind]['ap']
        if abs(average_precision_score(y[ass],scores[name][ass])-expected)>1e-12:raise ValueError('Legacy Vx scores not reproduced')
    if abs(average_precision_score(y[ass],base[ass])-old_metrics['results']['assessment']['metrics']['baseline']['ap'])>1e-12:
        raise ValueError('Native assessment not reproduced')
    folds=np.asarray(read(ROOT/'data/folds.json')['folds'])
    vx_fit,vx_full,vx_oof=crossfit(source['true'],y,fit,folds,pca=32,seed=cfg['source_seed'])
    np.testing.assert_allclose(vx_full,evidence(old['true'],source['true']),atol=1e-8,rtol=1e-8)
    # Only the calibration copies use OOF Vx evidence; all VAL scores use the immutable incumbent.
    vx_cal=base.copy(); vx_cal[fit]=fuse(base[fit],vx_oof[fit],gate[fit],old['true']['alpha'])
    states={'vx_oof':vx_fit}; vectors={}; x_quality=quality_features(source['quality'],sample,previous,diagnostics)
    state,z,oof=crossfit(x_quality,y,fit,folds,pca=cfg['fitting']['quality_pca'])
    states['quality_length']={'model':state,'native_fusion':choose_alpha(base,oof,gate,y,fit),
        'vx_fusion':choose_alpha(vx_cal,oof,gate,y,fit)}
    scores['quality_length']=fuse(base,z,gate,states['quality_length']['native_fusion']['alpha'])
    scores['combined_quality']=fuse(scores['vx'],z,gate,states['quality_length']['vx_fusion']['alpha'])
    for arm in cfg['arms']:
        x=np.asarray([r['features'][arm] for r in records],dtype=float)
        state,z,oof=crossfit(x,y,fit,folds)
        native_fusion=choose_alpha(base,oof,gate,y,fit); vx_fusion=choose_alpha(vx_cal,oof,gate,y,fit)
        states[arm]={'model':state,'native_fusion':native_fusion,'vx_fusion':vx_fusion}
        scores['contact_'+arm]=fuse(base,z,gate,native_fusion['alpha'])
        scores['combined_'+arm]=fuse(scores['vx'],z,gate,vx_fusion['alpha']); vectors[arm]=x
    true=states['true']; alpha=true['vx_fusion']['alpha']
    for arm,label in [('shuffled','shuffled'),('coordinate_true','coordinate'),('boundary_true','boundary')]:
        scores['combined_true_on_'+label]=fuse(scores['vx'],evidence(true['model']['full'],vectors[arm]),gate,alpha)
    for name,z in scores.items():
        if not np.isfinite(z).all() or not np.array_equal(z[dev&~available],base[dev&~available]):raise ValueError('Fallback or prediction failure: '+name)
    atomic(ROOT/'results/heads.json',states)
    masks={'assessment':ass,'fixed_val_sample':dev,'legacy_calibration_descriptive':cal,
        'crossing_descriptive':np.array([r['role']=='crossing' for r in sample])}
    length=np.array([r['length'] for r in sample])
    for pop,popmask in [('assessment',ass),('fixed_val_sample',dev)]:
        masks[pop+'/eligible']=popmask&available
        for label,low,high in [('le512',0,512),('513_1024',512,1024),('1025_1536',1024,1536),('over1536',1536,np.inf)]:
            masks[pop+'/length_'+label]=popmask&(length>low)&(length<=high)
    fields={
        'paired_depth':lambda d:d['retained_depth'],
        'neff80':lambda d:d['paired_diversity']['neff80'],
        'joint_support':lambda d:d['true_support']['fraction_cells_at_least_half_rows'],
        'valid_map_size':lambda d:d['valid_interchain_cells'],
        'minimum_quality':lambda d:d['minimum_quality'],
        'length_asymmetry':lambda d:d['length_asymmetry'],
        'null_token_change':lambda d:d['actual_token_changed_fraction']}
    cuts={}
    for name,get in fields.items():
        v=np.array([get(d['diagnostics']) if m['available'] else np.nan for d,m in zip(diagnostics,previous)])
        q=np.unique(np.quantile(v[fit],[.25,.5,.75])); cuts[name]=q.tolist(); bins=np.searchsorted(q,v,side='right')
        for k in range(len(q)+1):masks[f'assessment/{name}/train_bin_{k}']=ass&available&(bins==k)
    ann=read(ROOT/'data/dev_family_witnesses.json'); families=[]; family_count=np.full(len(sample),-1)
    for i,r in enumerate(sample):
        aa,bb=ann.get(str(r['a'])),ann.get(str(r['b']))
        if aa is not None and bb is not None:family_count[i]=int(aa['shared_family']=='True')+int(bb['shared_family']=='True')
        families.append({a['family'] for a in [aa,bb] if a is not None and a['family']})
    for k in [-1,0,1,2]:masks[f'assessment/shared_family_witness_count_{k}']=ass&available&(family_count==k)
    groups={n:summarize(m,y,scores,available) for n,m in masks.items()}
    # Bootstrap only scores needed by declared contrasts; every row gets the same protein weights.
    required={n for terms in CONTRASTS.values() for n in terms}
    cis=bootstrap([r for r,m in zip(sample,dev) if m],y[dev],{n:scores[n][dev] for n in required},
        {n:m[dev] for n,m in masks.items() if n not in ['legacy_calibration_descriptive','crossing_descriptive']},cfg['evaluation'],cfg['seed'])
    for name,ci in cis.items():groups[name]['ap_intervals']=ci
    influence={}
    for kind in ['protein','family_witness']:
        count=Counter(v for i in np.flatnonzero(ass) for v in ([sample[i]['a'],sample[i]['b']] if kind=='protein' else families[i]))
        vals=[]
        for member,_ in count.most_common(10):
            belongs=np.array([member in ([r['a'],r['b']] if kind=='protein' else families[i]) for i,r in enumerate(sample)])
            m=ass&~belongs; result=summarize(m,y,scores,available)
            vals.append({'group':str(member),'removed':int((ass&belongs).sum()),'contrasts':result.get('contrasts')})
        influence[kind]=vals
    mm=groups['assessment']; primary=cfg['evaluation']['primary']; ci=mm['ap_intervals'][primary]
    checks={'meaningful_gain':mm['contrasts'][primary]>=cfg['evaluation']['meaningful_ap_gain'],
        'positive_primary_interval':ci.get('low',-np.inf)>0,
        'separation_from_quality':mm['ap_intervals']['combined_true_minus_combined_quality'].get('low',-np.inf)>0,
        'auroc_noninferiority':mm['metrics']['combined_true']['auroc']>=mm['metrics']['vx']['auroc']-cfg['evaluation']['maximum_auroc_decline'],
        'boundary_exclusion_retains_positive_point':mm['contrasts']['combined_boundary_minus_vx']>0,
        'top_group_removals_retain_positive_point':all(v['contrasts'] is not None and v['contrasts'][primary]>0 for values in influence.values() for v in values)}
    exposure=read(ROOT/'provenance/exposure-audit.json')
    report={'at_utc':now(),'fingerprint':fp,'primary_contrast':primary,'contrast_definitions':CONTRASTS,
        'results':groups,'diagnostic_cuts_from_eligible_train':cuts,'group_influence':influence,
        'calibration_population':'eligible TRAIN OOF only; pair-grouped, not protein/family-disjoint',
        'legacy_vx_alpha':old['true']['alpha'],'legacy_vx_scores_unchanged':True,
        'native_train_logits_in_sample_for_native_model':True,'assessment_previously_inspected':True,
        'contact_head_training_membership_resolved':exposure['membership_resolved'],
        'test_accessed':False,'seconds':time.monotonic()-started}
    atomic(ROOT/'results/metrics.json',report)
    decision={'at_utc':now(),'fingerprint':fp,'status':'exploratory_contact_gain_exposure_unresolved' if all(checks.values()) else 'no_go_or_inconclusive',
        'checks':{k:bool(v) for k,v in checks.items()},'primary_point':mm['contrasts'][primary],'primary_interval':ci,
        'contact_head_training_membership_resolved':exposure['membership_resolved'],'production_authorized':False,'test_accessed':False}
    atomic(ROOT/'results/decision.json',decision)
    save_npz(ROOT/'results/val_predictions.npz',uids=np.array([r['uid'] for r,m in zip(sample,dev) if m]),labels=y[dev],**{n:z[dev] for n,z in scores.items()})
    lines=['# V7 v1 frozen contact-head pilot','',f"Decision: **{decision['status']}**.",'',
        'Primary: native + frozen Vx + true contact evidence versus native + frozen Vx. All new fitting and fusion selection used eligible TRAIN only. VAL assessment remains the original 958 rows.','',
        '| Assessment score | AP | AUROC |','|---|---:|---:|']
    for name,v in mm['metrics'].items():lines.append(f"| {name} | {v['ap']:.6f} | {v['auroc']:.6f} |")
    lines+=['','| Assessment AP contrast | Estimate | 95% matched protein interval |','|---|---:|---|']
    for name,point in mm['contrasts'].items():
        cc=mm['ap_intervals'][name]; lines.append(f"| {name} | {point:+.6f} | [{cc.get('low',float('nan')):+.6f}, {cc.get('high',float('nan')):+.6f}] |")
    lines+=['','The quality comparison adds the quality/length branch to the same frozen Vx incumbent. Coordinate and boundary variants are prespecified diagnostics, not alternate models selected on VAL.',
        '', 'The source contact-head training membership remains unresolved; no exposure-free or independently blind generalization claim is supported. Family witnesses are incomplete. Contact predictions are not calibrated human PPI probabilities.',
        '', 'Unavailable pairs retain exact native scores. Computational failures invalidate the run instead of becoming biological fallback. No TEST access or automatic continuation.']
    (ROOT/'results/REPORT.md').write_text('\n'.join(lines)+'\n')
    print(decision,flush=True)


if __name__ == '__main__':main()
