"""Six-model comparison, reusing frozen native/v5 predictions without inference."""
import csv, gzip, hashlib, json, os
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from bench_utils import ROOT, PROJECT, atomic, load_npz, now, read, record, save_npz, sha
from collect import collect, verify
from analyze import Ranking, csvfile, metrics
from historical_infer import EXT, signature, verify_inputs

TESTS=['mouse','fly','worm','yeast','ecoli']
NEW=['v2-capped','v2-clean-bce']
LABELS={'native-human':'Native PLM-interact humanV11','native-plm':'Native PLM-interact Bernett',
        'v2-capped':'V2 length-capped, update 8000','v2-clean-bce':'V2 clean BCE, update 4000',
        'ipin-esm2':'iPIN v5 ESM2','ipin-esmc':'iPIN v5 ESMC'}
NAMES=list(LABELS)


def merge(name):
    folder=EXT/'predictions'/name
    n=len(np.load(ROOT/'data/union.npy'));seen=np.zeros(n,dtype=np.int8);logits=np.full((n,2),np.nan)
    q=read(EXT/'qualification'/f'{name}.json');assert q['passed'] and q['fingerprint']==signature(name)
    manifests=[];gpus=[]
    for rank in range(4):
        path=folder/f'rank-{rank:02d}.done.json';m=read(path)
        assert m['rank']==rank and m['world']==4 and m['fingerprint']==q['fingerprint']
        count=0;gpus.append(m['gpu_uuid'])
        for entry in m['chunks']:
            side=read(verify(entry));assert side['fingerprint']==q['fingerprint']
            z=load_npz(verify(side['file']));ids=z['indices'];values=z['logits']
            assert values.shape==(len(ids),2) and len(np.unique(ids))==len(ids)
            assert (ids>=0).all() and (ids<n).all() and not seen[ids].any()
            seen[ids]+=1;logits[ids]=values;count+=len(ids)
        assert count==m['rows'];manifests.append(record(path))
    assert len(set(gpus))==4 and (seen==1).all() and np.isfinite(logits).all()
    values={'logits':logits,'scores':logits.mean(1),'probabilities':expit(logits.mean(1))}
    target=EXT/'results'/f'{name}-union.npz';save_npz(target,**values)
    return values,{'file':record(target),'shards':manifests,'qualification':record(EXT/'qualification'/f'{name}.json')}


def bootstrap(test,rows,values):
    y=rows[:,2];rankings=[Ranking(y,values[n]['scores']) for n in NAMES]
    check=np.random.default_rng(71).integers(0,4,len(rows)).astype(float)
    for name,ranking in zip(NAMES,rankings):
        truth=[average_precision_score(y,values[name]['scores'],sample_weight=check),roc_auc_score(y,values[name]['scores'],sample_weight=check)]
        assert np.allclose(ranking.compute(check),truth,atol=1e-12,rtol=1e-12)
    _,inverse=np.unique(rows[:,:2],return_inverse=True);ends=inverse.reshape(-1,2);count=int(ends.max()+1);selfpair=ends[:,0]==ends[:,1]
    rng=np.random.default_rng(20261005);samples=np.empty((1000,len(NAMES),2))
    for i in range(1000):
        freq=np.bincount(rng.integers(0,count,count),minlength=count)
        w=freq[ends[:,0]].astype(float)*freq[ends[:,1]];w[selfpair]=freq[ends[selfpair,0]]
        for j,ranking in enumerate(rankings):samples[i,j]=ranking.compute(w)
    save_npz(EXT/'results'/f'{test}-bootstrap.npz',samples=samples,names=np.array(NAMES),metrics=np.array(['ap','auroc']))
    estimates=np.array([r.compute(np.ones(len(rows))) for r in rankings]);cis=[];diffs=[]
    for j,name in enumerate(NAMES):
        for k,metric in enumerate(['ap','auroc']):
            low,high=np.quantile(samples[:,j,k],[.025,.975]);cis.append({'test':test,'model':name,'metric':metric,'estimate':estimates[j,k],'low':low,'high':high})
    comparisons=[(n,ref) for n in NEW for ref in ['native-human','native-plm','ipin-esm2','ipin-esmc']]+[('v2-clean-bce','v2-capped')]
    for name,ref in comparisons:
        j=NAMES.index(name);r=NAMES.index(ref)
        for k,metric in enumerate(['ap','auroc']):
            low,high=np.quantile(samples[:,j,k]-samples[:,r,k],[.025,.975])
            diffs.append({'test':test,'model':name,'reference':ref,'metric':metric,'difference':estimates[j,k]-estimates[r,k],'low':low,'high':high})
    print({'event':'bootstrap_complete','test':test,'replicates':1000},flush=True)
    return cis,diffs


def plots(cis):
    os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors=['#171717','#777777','#b17712','#26864d','#c83d43','#2879b9']
    for metric in ['ap','auroc']:
        fig,axes=plt.subplots(1,5,figsize=(16,4.8),sharey=True)
        for ax,test in zip(axes,TESTS):
            for j,name in enumerate(NAMES):
                c=next(r for r in cis if r['model']==name and r['test']==test and r['metric']==metric)
                ax.plot([c['low'],c['high']],[j,j],c=colors[j],lw=2)
                ax.scatter(c['estimate'],j,c=colors[j],s=32)
                ax.annotate(f"{c['estimate']:.3f}",(c['estimate'],j),xytext=(0,-12),textcoords='offset points',ha='center',fontsize=8)
            ax.set(title='E. coli' if test=='ecoli' else test.title(),xlabel='Average precision' if metric=='ap' else 'AUROC',xlim=(0,1.02))
            ax.axvline(1/11 if metric=='ap' else .5,color='gray',ls=':',lw=1);ax.grid(axis='x',alpha=.2)
        axes[0].set_yticks(range(len(NAMES)),[LABELS[n] for n in NAMES]);axes[0].invert_yaxis()
        fig.suptitle('Frozen earlier controls versus native and v5 models\nFull released rows; 95% paired protein-bootstrap intervals',fontsize=13)
        fig.tight_layout(rect=(0,0,1,.92))
        for extension in ['png','pdf','svg']:fig.savefig(EXT/'results'/f'{metric}-comparison.{extension}',dpi=180,bbox_inches='tight')
        plt.close(fig)


def table(rows,metric):
    out=['| Model | Mouse | Fly | Worm | Yeast | E. coli |','|---|---:|---:|---:|---:|---:|']
    for name in NAMES:
        out.append('| '+LABELS[name]+' | '+' | '.join(f"{next(r[metric] for r in rows if r['test']==test and r['model']==name):.4f}" for test in TESTS)+' |')
    return '\n'.join(out)


def main():
    threadpool_limits(1);sel=verify_inputs();provenance={};data={}
    for name in NAMES:
        if name in NEW:
            cp=sel['models'][name]['checkpoint'];assert sha(cp['path'])==cp['sha256']
            data[name],provenance[name]=merge(name)
        else:
            # Re-merge only existing completed inference shards, never infer.
            item=collect(name);assert item is not None,('reference_inference_incomplete',name)
            data[name]=load_npz(verify(item['file']));provenance[name]=item
    meta=read(ROOT/'data/sequences.json');mapping=load_npz(ROOT/'data/pair-mapping.npz');union=np.load(ROOT/'data/union.npy')
    prepared=read(ROOT/'provenance/prepared.json');exposure=load_npz(ROOT/'provenance/exposure-flags.npz')
    human_seen=np.zeros(len(meta['sequence']),dtype=bool)
    for key in read(ROOT/'provenance/exposure.json')['audits']:
        if not key.startswith('xpair-default'):human_seen|=exposure[key+'__endpoints']>0
    allmetrics=[];orientations=[];subsets=[];intervals=[];differences=[];counts={}
    for test in TESTS:
        rows=np.load(ROOT/'data'/f'{test}.npy');ids=mapping[test];y=rows[:,2];reversed_=mapping[test+'_reversed']
        assert np.array_equal(rows[:,3],np.arange(len(rows)))
        assert np.array_equal(np.sort(rows[:,:2],axis=1),union[ids,:2])
        source=prepared['source_files'][test];assert sha(source['path'])==source['sha256']
        with open(source['path']) as f:
            source_rows=list(csv.DictReader(f))
        assert len(source_rows)==len(rows)
        for i,r in enumerate(source_rows):
            a,b,label,_=map(int,rows[i]);assert r['query']==meta['sequence'][a] and r['text']==meta['sequence'][b] and int(r['label'])==label
        values={n:{k:v[ids].copy() for k,v in d.items()} for n,d in data.items()}
        for n in NAMES:
            values[n]['logits'][reversed_]=values[n]['logits'][reversed_,::-1]
            assert np.array_equal(values[n]['scores'],values[n]['logits'].mean(1))
        counts[test]={'rows':len(rows),'positives':int(y.sum()),'negatives':int((1-y).sum())}
        for n in NAMES:
            d=values[n];allmetrics.append({'test':test,'model':n,**counts[test],**metrics(y,d['scores'],d['probabilities'])})
            orientations.append({'test':test,'model':n,'original_order_ap':average_precision_score(y,d['logits'][:,0]),
                'original_order_auroc':roc_auc_score(y,d['logits'][:,0]),'reverse_order_ap':average_precision_score(y,d['logits'][:,1]),'reverse_order_auroc':roc_auc_score(y,d['logits'][:,1])})
        first={};conflicts=set()
        for i,uid in enumerate(ids):
            if int(uid) in first:
                if y[first[int(uid)]]!=y[i]:conflicts.add(int(uid))
            else:first[int(uid)]=i
        unique=np.zeros(len(rows),dtype=bool)
        for uid,i in first.items():
            if uid not in conflicts:unique[i]=True
        clean=~(human_seen[rows[:,0]]|human_seen[rows[:,1]])
        for group,mask in [('unique_pairs_without_conflicts',unique),('common_human_sources_endpoint_unexposed',clean)]:
            for n in NAMES:
                d=values[n];subsets.append({'test':test,'subset':group,'model':n,'rows':int(mask.sum()),'positives':int(y[mask].sum()),
                    'prevalence':float(y[mask].mean()),**metrics(y[mask],d['scores'][mask],d['probabilities'][mask])})
        with gzip.open(EXT/'results'/f'{test}-predictions.csv.gz','wt',newline='') as f:
            w=csv.writer(f);w.writerow(['source_row','label','sequence_a_sha256','sequence_b_sha256']+[f'{n}__{field}' for n in NAMES for field in ['score','probability','AB_logit','BA_logit']])
            for i,(a,b,label,row) in enumerate(rows):
                w.writerow([int(row),int(label),meta['sha256'][a],meta['sha256'][b]]+[v for n in NAMES for v in [values[n]['scores'][i],values[n]['probabilities'][i],*values[n]['logits'][i]]])
        ci,di=bootstrap(test,rows,values);intervals+=ci;differences+=di
    assert sum(c['rows'] for c in counts.values())==242000
    for name,records in [('metrics',allmetrics),('orientation-metrics',orientations),('subsets',subsets),('confidence-intervals',intervals),('paired-differences',differences)]:
        csvfile(EXT/'results'/f'{name}.csv',records)
    plots(intervals)
    summary={'at_utc':now(),'complete':True,'models':LABELS,'counts':counts,'metrics':allmetrics,
        'selection':record(EXT/'provenance/selection.json'),'prediction_provenance':provenance,
        'bootstrap':{'replicates':1000,'seed':20261005,'paired_protein_endpoints':True,'self_pairs_counted_once':True,'sklearn_weighted_metrics_verified':True},
        'raw_source_rows_verified':242000,'no_training_or_target_selection':True}
    atomic(EXT/'results/summary.json',summary)
    wins=[]
    for n in NEW:
        for ref in ['native-human','native-plm','ipin-esm2','ipin-esmc']:
            count=sum(next(r['ap'] for r in allmetrics if r['test']==t and r['model']==n)>next(r['ap'] for r in allmetrics if r['test']==t and r['model']==ref) for t in TESTS)
            wins.append(f"{LABELS[n]} has higher AP than {LABELS[ref]} in {count}/5 species.")
    clean=[r for r in subsets if r['subset']=='common_human_sources_endpoint_unexposed']
    text=['# Earlier native-like models on the five-species benchmark',
        f'Completed {now()}. Two frozen v2 checkpoints, each scored on all 242,000 released observations. Native humanV11, native Bernett and both v5 predictions were reused from the ongoing nonhuman benchmark. No retraining.',
        'The user requested both meanings of closest: **v2 length-capped, update 8,000** is the available control most similar to the native Bernett architecture/objective/length policy; **v2 clean BCE, update 4,000** is closest in the earlier Bernett AP/AUROC comparison. Both retain their existing DEV-selected weights. Historical Bernett performance informed the requested choice of model family, not checkpoint reselection on nonhuman data.',
        '## Average precision',table(allmetrics,'ap'),'![AP comparison](results/ap-comparison.png)',
        '## AUROC',table(allmetrics,'auroc'),'![AUROC comparison](results/auroc-comparison.png)',
        '## Comparison',' '.join(wins),
        'These are fixed-checkpoint transfer results. HumanV11 uses different human interaction evidence from the native Bernett and v2 models. The v2 models share the native ESM2-650M, standard attention and ReLU(CLS)-linear prediction head. Capped retains masked-input BCE plus MLM and restricts training to combined length 2,193; clean BCE removes training masking and MLM and uses all retained training lengths. Neither is an exact recreation of the authors’ unreported training trajectory. Differences do not isolate negative sampling, architecture or any single training choice.',
        'Every primary score pools AB/BA raw logits. Native humanV11 uses its qualified FP32 computation; both v2 models, native Bernett and v5 use FP32 weights with qualified BF16 autocast and TF32 disabled. The released strings are preserved, all are at most 800 residues, and the v2 training cap causes no test exclusion. The separate original-order metrics remain in [orientation-metrics.csv](results/orientation-metrics.csv); the earlier Figure 2 convention is documented in [the parent audit](../FIGURE2-AUDIT.md).',
        '## Uncertainty and exposure',
        'The figures use 1,000 paired protein-endpoint bootstrap replicates, seed 20261005. Non-self pairs receive the product of endpoint multiplicities; self-pairs receive one multiplicity. Each model uses the same resamples within a species. [Paired differences](results/paired-differences.csv) compare both v2 models with each native and v5 model. Intervals are descriptive, conditional on these trained checkpoints and historical datasets, without training-seed uncertainty or multiple-comparison adjustment.',
        'The conservative human-source exposure mask includes the original Bernett TRAIN/DEV files, a superset of the cleaned/capped v2 training and validation evidence. Its Bernett source component alone contains 94 exact shared test sequences and 18 mouse positive pairs. The shared mask also excludes the known human sources of v5, humanV11, TUnA, D-SCRIPT and SPRINT, so every displayed model uses identical remaining rows. It does not establish absence of homologs or PLM pretraining exposure.',
        table(clean,'ap'),
        'This mask retains 52,065 mouse pairs (3,944 positives), 54,858 fly pairs (4,997 positives), and all worm, yeast and E. coli rows. Subset metrics are descriptive point estimates; their changed prevalence prevents direct AP comparisons with the full tests. [Subsets](results/subsets.csv) also report one observation per unordered sequence pair after removing conflicting labels.',
        '## Validation and files',
        'Each checkpoint was hash-verified, loaded strictly and checked against the saved selected DEV predictions on short, middle and long batches. Fresh raw-string tokenization and unpatched original ESM2 forwards were independently checked on fixed examples from all five species, including each longest pair. All four GPU shards and their atomic chunks were verified, complete coverage restored, and all source strings and labels independently checked against the released CSVs. No production test score was used to choose a checkpoint, threshold, score direction or averaging rule.',
        '[Frozen selection](provenance/selection.json) · [Metrics](results/metrics.csv) · [Intervals](results/confidence-intervals.csv) · [Paired differences](results/paired-differences.csv) · [Summary and provenance](results/summary.json). Per-row scores are in `results/{species}-predictions.csv.gz`. PNG/PDF/SVG figures are saved alongside their data.']
    (EXT/'REPORT.md').write_text('\n\n'.join(text)+'\n')
    atomic(EXT/'results/COMPLETE.json',{'complete':True,'at_utc':now(),'models':NAMES,'rows_per_model':242000,
        'report':record(EXT/'REPORT.md'),'summary':record(EXT/'results/summary.json'),
        'artifacts':{str(p.relative_to(EXT)):record(p) for p in sorted((EXT/'results').iterdir()) if p.is_file() and p.name!='COMPLETE.json'},
        'analysis_code':record(__file__)})
    print('Historical comparison complete: '+str(EXT/'REPORT.md'),flush=True)


if __name__=='__main__':main()
