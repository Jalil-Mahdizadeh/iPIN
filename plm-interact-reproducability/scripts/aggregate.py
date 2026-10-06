"""Verify exact coverage, recompute metrics, and compare fresh vs deposited scores."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score,roc_auc_score
from datasets_eval import ROOT,SPECIES,task_specs,read_task
from metrics_eval import metrics

p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');args=p.parse_args()
OUT=ROOT/'results';PUBLISHED=ROOT/'data/published'
out=[];comparisons=[];coverage=[];complete={};missing=[]

def load_task(spec):
    paths=sorted((OUT/'predictions').glob(spec['name']+'.rank*.json'))
    if not paths:return None
    manifests=[json.loads(x.read_text()) for x in paths]
    world=manifests[0]['config']['world_size']
    if len(paths)!=world:return None
    assert sorted(m['config']['rank'] for m in manifests)==list(range(world))
    assert len(set(m['config']['code_sha256'] for m in manifests))==1
    assert len(set(m['config']['input_sha256'] for m in manifests))==1
    expected_hash=hashlib.sha256(Path(spec['path']).read_bytes()).hexdigest()
    assert all(m['config']['input_sha256']==expected_hash for m in manifests)
    frames=[]
    for path,m in zip(paths,manifests):
        csv=path.with_suffix('.csv')
        assert hashlib.sha256(csv.read_bytes()).hexdigest()==m['csv_sha256']
        frame=pd.read_csv(csv)
        assert len(frame)==m['n']
        frames.append(frame)
    combined=pd.concat(frames,ignore_index=True).sort_values('row_id').set_index('row_id')
    expected=read_task(spec)
    assert not combined.index.duplicated().any(), ('duplicate output rows',spec['name'])
    assert list(combined.index)==[r['row_id'] for r in expected], ('missing/extraneous rows',spec['name'])
    assert np.array_equal(combined.label,[r['label'] for r in expected])
    assert np.isfinite(combined.select_dtypes(include='number')).all().all()
    assert ((combined.score>=0)&(combined.score<=1)).all()
    assert all(m['model']['requires_grad']==False and m['model']['strict_load'] and m['model']['all_loaded_tensors_equal'] for m in manifests)
    if world>1:
        assert len(set(m['gpu_uuid'] for m in manifests))==world, 'GPU sharing across ranks'
    coverage.append(dict(task=spec['name'],n=len(combined),expected_n=len(expected),unique_row_ids=True,
                         labels_match=True,hashes_verified=True,world_size=world,
                         gpu_uuids=[m['gpu_uuid'] for m in manifests],job_ids=sorted(set(m['job_id'] for m in manifests)),
                         summed_gpu_worker_seconds=sum(m['elapsed_seconds'] for m in manifests)))
    combined.to_csv(OUT/f'{spec["name"]}.csv')
    return combined

for spec in task_specs():
    d=load_task(spec)
    if d is None:missing.append(spec['name'])
    else:complete[spec['name']]=d
if missing and not args.allow_partial:raise RuntimeError('Incomplete tasks: '+', '.join(missing))

def add_metrics(task,d,subset='all',score='score',threshold=.5):
    result=dict(task=task,subset=subset,score_definition=score,**metrics(d.label,d[score],threshold=threshold))
    if score=='score':
        raw='logit_difference' if 'logit_difference' in d else 'logit'
        result['average_precision_raw_logits']=float(average_precision_score(d.label,d[raw]))
        result['auroc_raw_logits']=float(roc_auc_score(d.label,d[raw]))
        result['average_precision_sigmoid_float64']=float(average_precision_score(d.label,expit(d[raw])))
    out.append(result)
    return result

def compare(task,fresh,source,labels):
    a=np.asarray(fresh);b=np.asarray(source)
    assert len(a)==len(b) and np.isfinite(a).all() and np.isfinite(b).all()
    delta=np.abs(a-b)
    comparisons.append(dict(task=task,n=len(a),mae=float(delta.mean()),max_abs=float(delta.max()),
                            median_abs=float(np.median(delta)),p99_abs=float(np.quantile(delta,.99)),
                            spearman=float(spearmanr(a,b).statistic),
                            classification_flips_at_0_5=int(((a>=.5)!=(b>=.5)).sum()),
                            fresh_ap=float(average_precision_score(labels,a)),source_ap=float(average_precision_score(labels,b)),
                            ap_difference=float(average_precision_score(labels,a)-average_precision_score(labels,b))))

for s in SPECIES:
    task=f'cross_{s}'
    if task not in complete:continue
    d=complete[task]
    add_metrics(task,d)
    ref=pd.read_csv(PUBLISHED/f'figure2_PLM-interact_{s}.csv')
    assert np.array_equal(d.label,ref.label)
    compare(task,d.score,ref.score,d.label)
    raw_reference=pd.read_csv(ROOT/f'data/published-extra/Supplementary Figure2/{s}_source_data_sfigure2_scores_all_models.csv')
    comparisons[-1].update(
        ap_float64_sigmoid=float(average_precision_score(d.label,expit(d.logit))),
        ap_float64_sigmoid_minus_source=float(average_precision_score(d.label,expit(d.logit))-average_precision_score(ref.label,ref.score)),
        raw_logit_mae_vs_published=float(np.abs(d.logit.to_numpy()-raw_reference.mask_15.to_numpy()).mean()),
        raw_logit_max_abs_vs_published=float(np.abs(d.logit.to_numpy()-raw_reference.mask_15.to_numpy()).max()),
        publisher_probability_vs_sigmoid64_max_abs=float(np.abs(expit(raw_reference.mask_15.to_numpy())-ref.score.to_numpy()).max()))
    rt=f'reverse_{s}_sample'
    if rt in complete:
        r=complete[rt];add_metrics(rt,r)
        compare(rt+'_vs_fresh_original',r.score,d.loc[r.index].score,r.label)
        rev=pd.read_csv(PUBLISHED/f'supplement6_{s}.csv').iloc[r.index]
        compare(rt+'_vs_published_reverse',r.score,rev.reverse,r.label)

for prefix in ['bernett','mutation_zero','mutation_ft']:
    base=prefix+'_full'
    if base not in complete:continue
    variants={base:complete[base]}
    for cap in (1603,2196):
        name=f'{prefix}_{cap}'
        if name in complete:
            merged=complete[base].copy()
            merged.loc[complete[name].index,complete[name].columns]=complete[name]
            assert len(merged)==len(complete[base]) and np.array_equal(merged.label,complete[base].label)
            variants[name]=merged
            merged.to_csv(OUT/f'{name}_all_rows.csv')
    for name,d in variants.items():
        if prefix=='bernett':
            add_metrics(name,d)
            ref=pd.read_csv(PUBLISHED/'figure4_PLM-interact.csv')
            assert np.array_equal(d.label,ref.label)
            compare(name,d.score,ref.score,d.label)
        else:
            mapping=pd.read_csv(ROOT/'data/mutation-figure5-mapping.csv')
            sub=d.loc[mapping.row_id]
            ref=pd.read_csv(PUBLISHED/'figure5_mutation.csv')
            assert np.array_equal(sub.label,ref.label)
            for subset,frame in [('all841',d),('published598',sub)]:
                add_metrics(name,frame,subset)
                add_metrics(name,frame,subset,score='probability_log_ratio',threshold=0.0)
            score_col='PLM-interact-FT_all_layers_scores' if prefix=='mutation_ft' else 'PLM-interact-zero_shot_scores'
            compare(name+'_published598',sub.score,ref[score_col],sub.label)

pd.DataFrame(out).to_csv(OUT/'fresh-metrics.csv',index=False)
pd.DataFrame(comparisons).to_csv(OUT/'score-concordance.csv',index=False)
(OUT/'coverage.json').write_text(json.dumps(dict(complete=not missing,missing=missing,tasks=coverage),indent=2)+'\n')
print(pd.DataFrame(out)[['task','subset','score_definition','n','average_precision','auroc','precision','recall','f1']].to_string(index=False))
print('Incomplete tasks:',missing)
