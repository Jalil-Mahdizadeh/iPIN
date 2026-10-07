"""Fixed-prediction evaluation; paired uncertainty and class-specific coverage."""
import csv, gzip
from sklearn.metrics import average_precision_score, roc_auc_score
from nonhuman_common import *
from evaluate import metrics, write_csv

def intervals_for(dataset,pairs,y,scores,estimates):
    names=list(scores); ranks=[Ranking(y,scores[n]) for n in names]
    check=np.random.default_rng(177).integers(0,4,len(y)).astype(float)
    for n,rank in zip(names,ranks):
        assert np.allclose(rank.compute(check),[average_precision_score(y,scores[n],sample_weight=check),
            roc_auc_score(y,scores[n],sample_weight=check)],atol=1e-12,rtol=1e-12)
    unique,inverse=np.unique(pairs,return_inverse=True); ends=inverse.reshape(-1,2)
    selfpair=ends[:,0]==ends[:,1]; n=len(unique)
    rng=np.random.default_rng(CONFIG['bootstrap']['seed']); samples=np.empty((500,len(names),2))
    for rep in range(500):
        counts=np.bincount(rng.integers(0,n,n),minlength=n)
        weights=counts[ends[:,0]].astype(float)*counts[ends[:,1]]
        weights[selfpair]=counts[ends[selfpair,0]]
        for j,rank in enumerate(ranks):samples[rep,j]=rank.compute(weights)
        if rep%250==0:print(dataset,'bootstrap',rep,flush=True)
    assert np.isfinite(samples).all()
    npz(ROOT/'results'/(dataset+'-bootstrap.npz'),samples=samples,names=np.asarray(names),metrics=np.asarray(['ap','auroc']))
    comparisons=[]
    for method in BASELINES:
        comparisons.append(('bernett:'+method,'v5:'+method))
        comparisons += [('v5:'+method,n) for n in ['ipin-esm2','ipin-esmc']]
        comparisons += [('bernett:'+method,n) for n in ['v2-capped','v2-clean-bce','native-plm']]
    ci=[];diff=[]
    for j,name in enumerate(names):
        for k,metric in enumerate(['ap','auroc']):
            lo,hi=np.quantile(samples[:,j,k],[.025,.975])
            ci.append({'dataset':dataset,'model':name,'metric':metric,'estimate':estimates[name][metric],'low':float(lo),'high':float(hi)})
    for name,reference in comparisons:
        a,b=names.index(name),names.index(reference)
        for k,metric in enumerate(['ap','auroc']):
            lo,hi=np.quantile(samples[:,a,k]-samples[:,b,k],[.025,.975])
            diff.append({'dataset':dataset,'model':name,'reference':reference,'metric':metric,
                         'difference':estimates[name][metric]-estimates[reference][metric],'low':float(lo),'high':float(hi)})
    return ci,diff

def main():
    start=time.monotonic(); freeze_extension(); verify_legacy_models()
    marker=ROOT/'provenance/nonhuman/evaluation.json'
    frozen=read(ROOT/'provenance/nonhuman/predictions-frozen.json')
    identity={'predictions_sha256':sha(ROOT/'provenance/nonhuman/predictions-frozen.json'),'script_sha256':sha(__file__)}
    if marker.exists():
        old=read(marker);assert old['identity']==identity
        for item in old['artifacts']:verify(item)
        print('Nonhuman evaluation verified; reused.',flush=True);return
    for item in frozen['artifacts']:verify(item)
    meta=read(NDATA/'sequences.json'); count=len(meta['sequence']); proteins={r:np.load(NMODELS/(r+'-protein-values.npz')) for r in TRAINING}
    expected={(r['test'],r['model']):r for r in csv.DictReader((NONHUMAN/'results/combined-figure-metrics.csv').open()) if r['subset']=='full'}
    summary={}; rows=[]; intervals=[]; differences=[]; coverage=[]; subsets=[]; subset_counts=[]; artifacts=[]
    union_mask=proteins['v5']['train_mask']|proteins['bernett']['train_mask']
    for species in SPECIES:
        z=np.load(NDATA/(species+'-rows.npz'));p,y=z['pairs'],z['labels'];scores={}; supports={}
        for ref in TRAINING:
            baseline=np.load(ROOT/'results'/(species+'-'+ref+'-baseline-scores.npz'))
            for name in BASELINES:scores[ref+':'+name]=baseline[name]
            supports[ref]=baseline['interolog_support_count']
        neural=np.load(NDATA/(species+'-neural.npz'))
        for name in NEURAL:scores[name]=neural[name]
        assert list(scores)==ALL_MODELS
        estimates={n:metrics(y,s) for n,s in scores.items()}
        for name in NEURAL:
            for metric in ['ap','auroc']:assert abs(estimates[name][metric]-float(expected[species,name][metric]))<1e-12
        counts={'rows':len(y),'positives':int(y.sum()),'negatives':int((1-y).sum())}
        summary[species]={**counts,'prevalence':float(y.mean()),'models':estimates}
        for name,m in estimates.items():rows.append({'dataset':species,'model':name,**counts,**m})
        for ref in TRAINING:
            values=proteins[ref];has=values['homolog_indices'][:,0]>=0;exact=values['train_mask']; active=np.unique(p)
            for label in [1,0]:
                mask=y==label;both=has[p].all(1); inter=scores[ref+':interolog']>0
                coverage.append({'dataset':species,'training_reference':ref,'label':label,'rows':int(mask.sum()),
                    'pairs_with_exact_endpoint':int((exact[p].any(1)&mask).sum()),
                    'pairs_with_both_homologs':int((both&mask).sum()),
                    'pairs_with_interolog_support':int((inter&mask).sum()),
                    'fraction_with_both_homologs':float(both[mask].mean()),'fraction_with_interolog_support':float(inter[mask].mean()),
                    'cohort_distinct_proteins':len(active),'cohort_proteins_with_homolog':int(has[active].sum()),
                    'cohort_proteins_exact_in_train':int(exact[active].sum())})
        codes=np.minimum(p[:,0],p[:,1])*count+np.maximum(p[:,0],p[:,1])
        _,first,inverse=np.unique(codes,return_index=True,return_inverse=True)
        pos=np.bincount(inverse,weights=y,minlength=len(first));neg=np.bincount(inverse,weights=1-y,minlength=len(first))
        conflict=(pos>0)&(neg>0); clean_unique=np.zeros(len(y),bool);clean_unique[first[~conflict]]=True
        masks={'common_train_endpoint_unexposed':~union_mask[p].any(1),'unique_nonconflicting_pairs':clean_unique}
        for subset,mask in masks.items():
            assert np.unique(y[mask]).size==2
            subcount={'rows':int(mask.sum()),'positives':int(y[mask].sum()),'negatives':int((1-y[mask]).sum())}
            subset_counts.append({'dataset':species,'subset':subset,**subcount,'prevalence':float(y[mask].mean()),
                                 'conflicting_pair_groups_in_full_set':int(conflict.sum())})
            for name,score in scores.items():subsets.append({'dataset':species,'subset':subset,'model':name,**subcount,**metrics(y[mask],score[mask])})
        # Predictions of each baseline must be identical for an unordered sequence pair, even with conflicting labels.
        for name in ALL_MODELS[:8]:assert np.array_equal(scores[name],scores[name][first[inverse]])
        path=ROOT/'results'/(species+'-predictions.csv.gz')
        with gzip.open(path,'wt',newline='') as f:
            writer=csv.writer(f);writer.writerow(['source_row_1based','protein_a_sha256','protein_b_sha256','label',*scores,
              'v5_a_hits','v5_b_hits','v5_interolog_support_count','bernett_a_hits','bernett_b_hits','bernett_interolog_support_count'])
            hits={r:(proteins[r]['homolog_indices']>=0).sum(1) for r in TRAINING}
            for i,((a,b),label) in enumerate(zip(p,y)):
                writer.writerow([i+1,meta['sha256'][a],meta['sha256'][b],int(label),*[repr(float(scores[n][i])) for n in scores],
                    int(hits['v5'][a]),int(hits['v5'][b]),int(supports['v5'][i]),int(hits['bernett'][a]),int(hits['bernett'][b]),int(supports['bernett'][i])])
        artifacts.append(record(path))
        print(species,{n:{k:m[k] for k in ['ap','auroc']} for n,m in estimates.items()},flush=True)
        ci,di=intervals_for(species,p,y,scores,estimates);intervals+=ci;differences+=di
        artifacts.append(record(ROOT/'results'/(species+'-bootstrap.npz')))
    for filename,items in [('nonhuman-metrics.csv',rows),('nonhuman-confidence-intervals.csv',intervals),
        ('nonhuman-paired-differences.csv',differences),('nonhuman-coverage.csv',coverage),('nonhuman-subsets.csv',subsets),('nonhuman-subset-counts.csv',subset_counts)]:
        path=ROOT/'results'/filename;write_csv(path,items);artifacts.append(record(path))
    result={'complete':True,'at_utc':now(),'datasets':summary,'training':frozen['references'],'models':{n:NLABELS[n] for n in ALL_MODELS},
        'config':CONFIG,'checks':{'all_neural_primary_metrics_reproduced':True,'independent_weighted_metrics_agree':True,
        'new_baseline_same_unordered_pair_scores_identical':True,'shared_bootstrap_weights_all_models':True,'no_target_selection_or_inversion':True},
        'resource_use':stage_stat(start)}
    path=ROOT/'results/nonhuman-summary.json';atomic(path,result);artifacts.append(record(path))
    atomic(marker,{'complete':True,'at_utc':now(),'identity':identity,'artifacts':artifacts,'resource_use':stage_stat(start)})
    print('Five-species evaluation complete',flush=True)

if __name__=='__main__':main()
