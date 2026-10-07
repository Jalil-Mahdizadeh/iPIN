"""Independent source/export, fitted-model and paired-bootstrap verification."""
import csv,gzip,joblib
from sklearn.metrics import average_precision_score,roc_auc_score
from nonhuman_common import *
from fit_predict import features

def main():
    start=time.monotonic();freeze_extension();verify_legacy_models();checks=[]
    def check(name,condition):
        assert condition,name
        checks.append(name)
    legacy_complete=read(ARCHIVE/'results/COMPLETE.json')
    for item in legacy_complete['artifacts']:
        archived=ARCHIVE/item['path'];p=archived if archived.exists() else ROOT/item['path']
        assert sha(p)==item['sha256'],p
    check('all 83 original-study artifact hashes preserved',len(legacy_complete['artifacts'])==83)
    prepared=read(ROOT/'provenance/nonhuman/prepared.json')
    for item in prepared['artifacts']+prepared['sources']:verify(item)
    check('original released species rows and every reused source hash verified',prepared['source_rows_verified']==242000)
    frozen=read(ROOT/'provenance/nonhuman/predictions-frozen.json')
    for item in frozen['artifacts']:verify(item)
    evaluation=read(ROOT/'provenance/nonhuman/evaluation.json')
    for item in evaluation['artifacts']:verify(item)
    summary=read(ROOT/'results/nonhuman-summary.json');meta=read(NDATA/'sequences.json')
    check('distinct complete sequence hashes verified',len(set(meta['sequence']))==len(meta['sequence']) and
          all(hashlib.sha256(s.encode()).hexdigest()==h for s,h in zip(meta['sequence'],meta['sha256'])))
    train_masks={};references={}
    rng=np.random.default_rng(812);sample=rng.choice(meta['query_indices'],500,replace=False)
    for ref in TRAINING:
        v=np.load(NMODELS/(ref+'-protein-values.npz'));references[ref]=v;train_masks[ref]=v['train_mask']
        path=verify(frozen['references'][ref]['fitted_regressor']);model=joblib.load(path)
        prediction=model.predict(features([meta['sequence'][i] for i in sample]))
        check(ref+': regressor reload reproduces species protein scores exactly',np.array_equal(prediction,v['sequence_propensity'][sample]))
        t=set(meta['train_indices'][ref]);h=v['homolog_indices'];w=v['homolog_weights']
        check(ref+': every retained target belongs exclusively to its own TRAIN',set(h[h>=0].tolist())<=t)
        check(ref+': retained weights finite and bounded',np.isfinite(w).all() and ((w>=0)&(w<=1)).all())
        for i in sample:
            if i in t:expected=v['exact_degree'][i]
            else:
                included=[(int(j),float(weight)) for j,weight in zip(h[i],w[i]) if j>=0]
                expected=sum(float(v['exact_degree'][j])*weight for j,weight in included)/sum(weight for _,weight in included) if included else frozen['references'][ref]['no_hit_propensity']
            assert abs(expected-v['homology_degree'][i])<1e-13
        checks.append(ref+': scalar transferred propensity checked on 500 proteins')
        precise=read(ROOT/'provenance/nonhuman'/('alignment-precision-'+ref+'.json'))
        check(ref+': adding backtraces left every primary alignment field unchanged',precise['primary_first_nine_fields_identical_for_every_alignment'])
    count=0;intervals=list(csv.DictReader((ROOT/'results/nonhuman-confidence-intervals.csv').open()))
    differences=list(csv.DictReader((ROOT/'results/nonhuman-paired-differences.csv').open()))
    for species in SPECIES:
        data=np.load(NDATA/(species+'-rows.npz'));p,y=data['pairs'],data['labels'];scores={}
        for ref in TRAINING:
            baseline=np.load(ROOT/'results'/(species+'-'+ref+'-baseline-scores.npz'))
            for method in BASELINES:scores[ref+':'+method]=baseline[method]
        neural=np.load(NDATA/(species+'-neural.npz'))
        scores.update({name:neural[name] for name in NEURAL})
        matrix=np.column_stack([scores[n] for n in ALL_MODELS])
        with gzip.open(ROOT/'results'/(species+'-predictions.csv.gz'),'rt') as f:
            n=0
            for n,row in enumerate(csv.DictReader(f),1):
                a,b=p[n-1]
                assert int(row['source_row_1based'])==n and int(row['label'])==int(y[n-1])
                assert row['protein_a_sha256']==meta['sha256'][a] and row['protein_b_sha256']==meta['sha256'][b]
                assert np.array_equal(np.array([float(row[k]) for k in ALL_MODELS]),matrix[n-1])
        check(species+': every exported row/label/sequence/score matches',n==len(y));count+=n
        for name,v in scores.items():
            expected=summary['datasets'][species]['models'][name]
            assert abs(average_precision_score(y,v)-expected['ap'])<1e-12
            assert abs(roc_auc_score(y,v)-expected['auroc'])<1e-12
        checks.append(species+': all AP/AUROC point estimates reconstructed')
        boot=np.load(ROOT/'results'/(species+'-bootstrap.npz'));names=boot['names'].tolist();samples=boot['samples']
        check(species+': complete finite paired bootstrap',samples.shape==(500,13,2) and names==ALL_MODELS and np.isfinite(samples).all())
        unique,inverse=np.unique(p,return_inverse=True);endpoints=inverse.reshape(-1,2)
        multiplicity=np.bincount(np.random.default_rng(20261006).integers(0,len(unique),len(unique)),minlength=len(unique))
        weights=multiplicity[endpoints[:,0]].astype(float)*multiplicity[endpoints[:,1]]
        same=endpoints[:,0]==endpoints[:,1];weights[same]=multiplicity[endpoints[same,0]]
        for j,name in enumerate(names):
            expected=[average_precision_score(y,scores[name],sample_weight=weights),roc_auc_score(y,scores[name],sample_weight=weights)]
            assert np.allclose(samples[0,j],expected,atol=1e-12,rtol=1e-12)
        checks.append(species+': first shared bootstrap replicate independently reconstructed with sklearn')
        for row in [r for r in intervals if r['dataset']==species]:
            j=names.index(row['model']);k=['ap','auroc'].index(row['metric'])
            assert np.allclose(np.quantile(samples[:,j,k],[.025,.975]),[float(row['low']),float(row['high'])],atol=1e-14)
        for row in [r for r in differences if r['dataset']==species]:
            j=names.index(row['model']);r=names.index(row['reference']);k=['ap','auroc'].index(row['metric'])
            assert np.allclose(np.quantile(samples[:,j,k]-samples[:,r,k],[.025,.975]),[float(row['low']),float(row['high'])],atol=1e-14)
        checks.append(species+': confidence intervals and paired differences verified')
    check('all 242000 species source observations evaluated',count==242000)
    # The numerical formula/functions and all original v5 model bytes are unchanged.
    original_science=read(ROOT/'provenance/nonhuman/protocol-freeze.json')['identity']['original_scientific_scripts']
    check('inherited scientific scripts unchanged',all(sha(ROOT/'scripts'/s)==h for s,h in original_science.items()))
    atomic(ROOT/'results/nonhuman-verification.json',{'status':'passed','at_utc':now(),'checks':checks,
        'exported_rows_checked':count,'model_scores_per_row':13,'script':record(Path(__file__)),'resource_use':stage_stat(start)})
    print('Independent nonhuman verification passed:',len(checks),'checks;',count,'rows',flush=True)

if __name__=='__main__':main()
