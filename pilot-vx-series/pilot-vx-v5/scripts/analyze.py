"""Three fixed heads; same TRAIN, calibration, assessment, and depth-128 references."""
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from study import ROOT,config,read,atomic,atomic_npz,now,sha,verify_freeze,load_record,fuse,check_budget,DIMENSIONS
from inputs import vector_hash
from heads import fit_head,references
from strata import strata
from statistics_v5 import CONTRASTS,bootstrap,summarize
from decision import decide
from input_report import aggregate


def load_complete(fp):
    sample=read(ROOT/'data/sample.json');pairs=read(ROOT/'data/pairs.json')
    if len(sample)!=config()['expected']['pairs'] or len(pairs)!=len(sample):raise ValueError('Fixed cohort changed')
    with np.load(ROOT/'data/source.npz',allow_pickle=False) as f:old={k:f[k].copy() for k in f.files}
    if old['uids'].tolist()!=[r['uid'] for r in sample]:raise ValueError('Original feature order differs')
    with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(lambda p:load_record(p['uid'],fp,p),pairs))
    if any(r is None for r in records):raise RuntimeError('Incomplete computational coverage; no fit allowed')
    for j,(s,p,r) in enumerate(zip(sample,pairs,records)):
        if s['uid']!=p['uid'] or sorted([s['a'],s['b']])!=[p['a'],p['b']] or p['available']!=(p['gate']>0) or p['gate']!=old['gate'][j]:
            raise ValueError('Original identity or gate changed')
        timing=r['timing']
        if not p['available']:
            if timing is not None:raise ValueError('Fallback encoded')
        else:
            if vector_hash(r['features']['P256'])!=p['P256_sha256']:raise ValueError('Quality features differ from prepared input')
            if p['depth_increased']:
                if timing is None or timing.get('kind')!='encoded':raise ValueError('Missing GPU timing')
                for arm in ('T256','S256'):
                    t=timing[arm]
                    if t['orientation_depths']!=[p['depth_256']]*2 or t['orientation_layers']!=list(range(16))*2 or t['length']!=p['length'] or t['breakpoint']!=p['breakpoint']:
                        raise ValueError('Wrong depth/layer/coordinate execution')
            else:
                if timing!={'kind':'reused_unchanged_depth'}:raise ValueError('Incorrect feature reuse')
                for arm,name in [('T256','true'),('S256','shuffled'),('P256','quality')]:np.testing.assert_array_equal(r['features'][arm],old[name][j])
    return sample,pairs,{arm:np.asarray([r['features'][arm] for r in records]) for arm in DIMENSIONS}


def main():
    started=time.monotonic();cfg=config();fp=verify_freeze();check_budget()
    if (ROOT/'results/fit-start.json').exists():raise RuntimeError('Fit already started; no automatic refitting')
    sample,pairs,x=load_complete(fp);y=np.array([r['label'] for r in sample]);gate=np.array([p['gate'] for p in pairs]);available=gate>0
    train=np.array([r['split']=='train' for r in sample]);dev=~train;fit=train&available
    cal=np.array([r['role']=='calibration' for r in sample]);ass=np.array([r['role']=='assessment' for r in sample])
    counts={name:[int(np.sum(m&(y==c))) for c in (0,1)] for name,m in [('fit',fit),('calibration',cal),('assessment',ass)]}
    if counts!=read(ROOT/'data/source_metrics.json')['fit_counts']:raise ValueError('Original class/split counts changed')
    if min(counts['fit'])<cfg['fusion']['minimum_fit_per_class'] or min(counts['calibration']+counts['assessment'])<cfg['fusion']['minimum_calibration_assessment_per_class']:raise ValueError('Insufficient support')
    base,scores,ev,oldgate=references(ROOT,sample);np.testing.assert_array_equal(gate,oldgate)
    masks,cuts,families=strata(sample,pairs,cfg)
    for pop,m in [('assessment',ass),('fixed_dev_sample',dev)]:
        for name,select in [('depth_unchanged',lambda p:p['available'] and not p['depth_increased']),
                            ('depth_extended_partial',lambda p:p.get('depth_increased',False) and p['depth_256']<256),
                            ('depth_full256',lambda p:p.get('depth_256')==256)]:
            masks[pop+'/'+name]=m&np.array([select(p) for p in pairs])
    atomic(ROOT/'results/input-diagnostics.json',aggregate(read(ROOT/'data/input-audits.json'),sample,pairs,{'TRAIN':train,**masks}))
    fit_uids=[r['uid'] for r,keep in zip(sample,fit) if keep]
    atomic(ROOT/'results/fit-start.json',{'at_utc':now(),'fingerprint':fp,'fit_pair_uids':fit_uids,'TRAIN_baseline_used':False,'arms':list(DIMENSIONS)})
    models={}
    for arm in DIMENSIONS:models[arm],ev[arm],scores[arm]=fit_head(x[arm],y,fit,cal,base,gate,cfg)
    atomic(ROOT/'results/heads.json',{'at_utc':now(),'fingerprint':fp,'fit_pair_uids':fit_uids,'TRAIN_baseline_used':False,**models})
    for z in scores.values():np.testing.assert_array_equal(z[dev&~available],base[dev&~available])
    groups={name:summarize(mask,y,scores,available) for name,mask in masks.items()}
    for pop,m in [('assessment',ass),('fixed_dev_sample',dev)]:
        local={name:mask[m] for name,mask in masks.items() if name==pop or name.startswith(pop+'/')}
        ci=bootstrap([r for r,keep in zip(sample,m) if keep],y[m],{k:z[m] for k,z in scores.items()},local,cfg['fusion'],cfg['seed'])
        for name,value in ci.items():groups[name]['ap_intervals']=value
    previous=read(ROOT/'data/source_metrics.json')
    for pop in ('assessment','fixed_dev_sample'):
        for name,old in [('true_minus_baseline','true-minus-baseline'),('true_minus_shuffled','true-minus-shuffled')]:
            c,ref=groups[pop]['ap_intervals'][name],previous['results'][pop]['ap_intervals'][old]
            np.testing.assert_allclose([c['low'],c['high']],[ref['low'],ref['high']],atol=1e-12,rtol=1e-12)
    common={};unfused={}
    for pop,m in [('assessment',ass),('assessment/eligible',ass&available),('calibration',cal)]:
        common[pop]={str(a):{arm:{'ap':float(average_precision_score(y[m],fuse(base,v,gate,a)[m])),
            'auroc':float(roc_auc_score(y[m],fuse(base,v,gate,a)[m]))} for arm,v in ev.items()} for a in cfg['fusion']['alpha_grid']}
        mm=m&available;unfused[pop]={arm:{'ap':float(average_precision_score(y[mm],v[mm])),
            'auroc':float(roc_auc_score(y[mm],v[mm]))} for arm,v in ev.items()}
    metrics={'at_utc':now(),'fingerprint':fp,'primary_contrast':'T256_minus_true','contrast_definitions':CONTRASTS,
        'results':groups,'fit_counts':counts,'alphas':{**{k:h['alpha'] for k,h in read(ROOT/'data/source_heads.json').items()},**{k:h['alpha'] for k,h in models.items()}},
        'common_alpha_descriptive':common,'unfused_eligible_head_metrics':unfused,'diagnostic_stratum_cuts_from_eligible_train':cuts,
        'bootstrap':'Original assessment-specific matched protein Poisson, 1000 draws, seed 20261007, conditional intervals',
        'subgroup_intervals':'Exploratory and unadjusted; not used for selection','assessment_previously_inspected':True,
        'original_predictions_and_intervals_reproduced':True,'exact_baseline_fallback':True,'test_accessed':False,
        'R_evaluated':False,'forbidden_heads_evaluated':False,'input_diagnostics_sha256':sha(ROOT/'results/input-diagnostics.json'),
        'seconds':time.monotonic()-started}
    atomic(ROOT/'results/metrics.json',metrics)
    decision={**decide(groups,cfg),'at_utc':now(),'fingerprint':fp,'metrics_sha256':sha(ROOT/'results/metrics.json')}
    atomic(ROOT/'results/decision.json',decision)
    atomic_npz(ROOT/'results/dev_predictions.npz',uids=np.array([r['uid'] for r,keep in zip(sample,dev) if keep]),labels=y[dev],gate=gate[dev],**{k:z[dev] for k,z in scores.items()})
    atomic_npz(ROOT/'results/dev_evidence.npz',**{k:z[dev] for k,z in ev.items()})
    m=groups['assessment'];lines=['# Vx v5: matched MSA depth 256 versus 128','',f"Decision: **{decision['status']}**.",'',
        'Same original TRAIN/DEV cohort, masks, gates and readout. Depth includes the query. Historical depth-128 predictions reused unchanged. Exploratory assessment; no TEST access.','',
        '| Assessment arm | AP | AUROC |','|---|---:|---:|']
    for arm in ['baseline','true','T256','shuffled','S256','quality','P256']:
        v=m['metrics'][arm];lines.append(f"| {arm} | {v['ap']:.6f} | {v['auroc']:.6f} |")
    lines+=['','| AP contrast | Estimate | 95% matched protein interval |','|---|---:|---|']
    for name,value in m['contrasts'].items():
        c=m['ap_intervals'][name];lines.append(f"| {name} | {value:+.6f} | [{c['low']:+.6f}, {c['high']:+.6f}] |")
    lines+=['',f"New fusion alphas: { {k:h['alpha'] for k,h in models.items()} }. TRAIN-only fits on {fit.sum()} pairs; assessment {ass.sum()} rows, {(ass&available).sum()} eligible.",'',
        'Primary material benefit requires at least +0.010 AP with positive lower interval and AUROC loss no greater than 0.005 versus true128. An interval upper bound below +0.010 argues against that material benefit, not against every smaller gain. Non-significance is not equivalence.','',
        'True256 extends the exact true128 prefix using the unchanged accession matcher. Shuffled256 applies the original shuffler at its larger depth, so its old-prefix permutation can change. Naturally shallow inputs reuse verified original features; all three new heads are fitted once.','',
        'See depth-coverage.json for actual depth counts, input-diagnostics.json for diversity/coverage, qualification/real.json for same-pair timing at both depths, and metrics.json for length/depth/family strata and common-alpha diagnostics. No automatic full benchmark or TEST evaluation.']
    (ROOT/'results/REPORT.md').write_text('\n'.join(lines)+'\n');print({'decision':decision['status'],'alphas':metrics['alphas']},flush=True)


if __name__=='__main__':main()
