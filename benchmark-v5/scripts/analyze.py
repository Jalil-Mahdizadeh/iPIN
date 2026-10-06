"""Fixed-test comparisons, paired protein bootstrap and transparent diagnostics."""
import csv,gzip,hashlib,json,os,time
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,roc_curve
from threadpoolctl import threadpool_limits
from bench_utils import ROOT,atomic,load_npz,now,read,record,save_npz,sha
from collect import NAMES,LABELS,verify
REPLICATES=1000;SEED=20260929

def csvfile(path,records):
 with path.open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)

class Ranking:
 def __init__(self,y,scores):
  self.order=np.argsort(-scores,kind='stable');self.y=y[self.order];ordered=scores[self.order]
  self.ends=np.r_[np.flatnonzero(np.diff(ordered)),len(ordered)-1]
 def compute(self,w):
  w=w[self.order];tp=np.cumsum(w*self.y)[self.ends];fp=np.cumsum(w*(1-self.y))[self.ends];p,n=tp[-1],fp[-1]
  assert p>0 and n>0
  precision=np.divide(tp,tp+fp,out=np.zeros_like(tp),where=(tp+fp)>0)
  return np.array([np.dot(np.diff(np.r_[0.,tp]),precision)/p,np.dot(np.diff(np.r_[0.,fp]),(tp+np.r_[0.,tp[:-1]])/2)/(p*n)])

def metrics(y,scores,prob=None):
 return {'ap':float(average_precision_score(y,scores)),'auroc':float(roc_auc_score(y,scores)),
  'brier':None if prob is None else float(np.mean((prob-y)**2)),'score_min':float(scores.min()),'score_max':float(scores.max()),
  'unique_scores':int(len(np.unique(scores))),'zero_scores':int((scores==0).sum())}

def operating(y,scores,threshold):
 positive=scores>=threshold;tp=int((positive&(y==1)).sum());fp=int((positive&(y==0)).sum());fn=int((~positive&(y==1)).sum());tn=int((~positive&(y==0)).sum())
 return {'threshold':threshold,'tp':tp,'fp':fp,'fn':fn,'tn':tn,'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn),
  'specificity':tn/(tn+fp),'f1':2*tp/(2*tp+fp+fn),'accuracy':(tp+tn)/len(y)}

def bootstrap(test,rows,data):
 y=rows[:,2];rankings=[Ranking(y,data[n]['scores']) for n in NAMES]
 check=np.random.default_rng(71).integers(0,4,len(y)).astype(float)
 for name,ranking in zip(NAMES,rankings):
  expected=[average_precision_score(y,data[name]['scores'],sample_weight=check),roc_auc_score(y,data[name]['scores'],sample_weight=check)]
  assert np.allclose(ranking.compute(check),expected,atol=1e-12,rtol=1e-12),(name,expected,ranking.compute(check))
 _,inverse=np.unique(rows[:,:2],return_inverse=True);endpoints=inverse.reshape(-1,2);proteins=int(endpoints.max()+1);selfpair=endpoints[:,0]==endpoints[:,1]
 cache=ROOT/'results/bootstrap-cache';cache.mkdir(exist_ok=True)
 samples=np.empty((REPLICATES,len(NAMES),2));missing=[];identities={}
 for j,name in enumerate(NAMES):
  identity={'model':name,'test':test,'replicates':REPLICATES,'seed':SEED,'rows':sha(ROOT/'data'/f'{test}.npy'),
   'scores':hashlib.sha256(data[name]['scores'].tobytes()).hexdigest(),'analysis_code':sha(__file__)}
  identities[name]=identity;path=cache/f'{test}-{name}.npz';side=path.with_suffix('.json')
  if side.exists():
   item=read(side);previous=item['identity']
   if previous!=identity:
    prior=ROOT/'archive/before-human-releases/scripts/analyze.py'
    assert previous['analysis_code']==sha(prior)
    assert {k:v for k,v in previous.items() if k!='analysis_code'}=={k:v for k,v in identity.items() if k!='analysis_code'}
   assert sha(path)==item['file']['sha256'];saved=load_npz(path)['samples']
   assert saved.shape==(REPLICATES,2) and np.isfinite(saved).all();samples[:,j]=saved
  else:missing.append(j)
 rng=np.random.default_rng(SEED);started=time.monotonic()
 for rep in range(REPLICATES):
  counts=np.bincount(rng.integers(0,proteins,proteins),minlength=proteins);weights=counts[endpoints[:,0]].astype(float)*counts[endpoints[:,1]];weights[selfpair]=counts[endpoints[selfpair,0]]
  for j in missing:samples[rep,j]=rankings[j].compute(weights)
  if rep in [0,500,999]:
   for j in range(len(NAMES)):
    if j not in missing:assert np.allclose(samples[rep,j],rankings[j].compute(weights),atol=1e-12,rtol=0)
  if rep%200==0:print({'test':test,'bootstrap':rep,'seconds':time.monotonic()-started},flush=True)
 for j in missing:
  name=NAMES[j];path=cache/f'{test}-{name}.npz';save_npz(path,samples=samples[:,j]);atomic(path.with_suffix('.json'),{'identity':identities[name],'file':record(path)})
 save_npz(ROOT/'results'/(test+'-bootstrap.npz'),samples=samples,names=np.array(NAMES),metrics=np.array(['ap','auroc']))
 estimates=np.array([r.compute(np.ones(len(y))) for r in rankings]);intervals=[];diffs=[]
 for j,name in enumerate(NAMES):
  for k,metric in enumerate(['ap','auroc']):
   low,high=np.quantile(samples[:,j,k],[.025,.975]);intervals.append({'test':test,'model':name,'metric':metric,'estimate':float(estimates[j,k]),'low':float(low),'high':float(high)})
 comparisons=[(j,0) for j in range(1,len(NAMES))]+[(1,2)]
 comparisons += [(NAMES.index(n),NAMES.index(ref)) for n in ['ipin-esm2','ipin-esmc'] for ref in ['native-human','tuna-human']]
 comparisons += [(NAMES.index('tuna-human'),NAMES.index('tuna'))]
 for j,other in comparisons:
  for k,metric in enumerate(['ap','auroc']):
   low,high=np.quantile(samples[:,j,k]-samples[:,other,k],[.025,.975]);diffs.append({'test':test,'model':NAMES[j],'reference':NAMES[other],'metric':metric,
    'difference':float(estimates[j,k]-estimates[other,k]),'low':float(low),'high':float(high)})
 return intervals,diffs,{'replicates':REPLICATES,'seed':SEED,'test_unique_proteins':proteins,'self_pairs':int(selfpair.sum()),'sklearn_check_passed':True,
  'method':'Resample endpoints uniformly with replacement; non-self pair multiplicity is product; self-pair uses one multiplicity.'}

def macro(rows,data):
 n=int(rows[:,:2].max()+1);incident=[[] for _ in range(n)]
 for i,(a,b) in enumerate(rows[:,:2]):
  incident[a].append(i)
  if b!=a:incident[b].append(i)
 eligible=[]
 for items in incident:
  if len(items)>=10 and len(np.unique(rows[items,2]))==2:eligible.append(np.array(items))
 out={};prevalence=float(np.mean([rows[ids,2].mean() for ids in eligible]))
 for name in NAMES:
  values=[Ranking(rows[ids,2],data[name]['scores'][ids]).compute(np.ones(len(ids)))[0] for ids in eligible]
  out[name]={'eligible_proteins':len(eligible),'minimum_incident_pairs':10,'requires_both_labels':True,'macro_ap':float(np.mean(values)),'mean_local_prevalence':prevalence}
 return out

def plots(results,intervals,testdata):
 os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 colors=['#202020','#c44e52','#4c72b0','#55a868','#8172b3','#ccb974','#64b5cd','#8c6143','#da8bc3','#e17c05','#146b6b'];labels=[LABELS[n] for n in NAMES]
 fig,axes=plt.subplots(1,2,figsize=(13,6),sharey=True)
 for ax,test in zip(axes,['original','ilp']):
  for j,name in enumerate(NAMES):
   item=next(x for x in intervals if x['test']==test and x['model']==name and x['metric']=='ap');v=item['estimate']
   ax.errorbar(v,j,xerr=[[v-item['low']],[item['high']-v]],fmt='o',color=colors[j],capsize=3)
  ax.set_title('Original Bernett test' if test=='original' else 'Custom ILP-negative test');ax.set_xlabel('Average precision (95% protein-bootstrap interval)');ax.grid(axis='x',alpha=.2);ax.axvline(.5,color='gray',linestyle=':',linewidth=1)
 axes[0].set_yticks(range(len(labels)),labels);axes[0].invert_yaxis();fig.suptitle('Frozen checkpoints | same positives, different negative distributions')
 fig.text(.02,.015,'* Documented supervised-data overlap; common-subset results are reported separately.',fontsize=9,color='#526071');fig.tight_layout(rect=(0,.035,1,1))
 for ext in ['png','pdf']:fig.savefig(ROOT/'results'/('average-precision.'+ext),dpi=180,bbox_inches='tight')
 plt.close(fig)
 fig,axes=plt.subplots(2,2,figsize=(12,10))
 for col,test in enumerate(['original','ilp']):
  y=testdata[test]['rows'][:,2]
  for j,name in enumerate(NAMES):
   z=testdata[test]['data'][name]['scores'];p,r,_=precision_recall_curve(y,z);fpr,tpr,_=roc_curve(y,z)
   axes[0,col].plot(r,p,color=colors[j],label=LABELS[name],linewidth=1.5 if j<3 else 1,alpha=1 if j<3 else .8)
   axes[1,col].plot(fpr,tpr,color=colors[j],linewidth=1.5 if j<3 else 1,alpha=1 if j<3 else .8)
  axes[0,col].set(title='Original Bernett' if test=='original' else 'Custom ILP negatives',xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1));axes[0,col].axhline(.5,color='gray',ls=':',lw=1)
  axes[1,col].set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(0,1),ylim=(0,1));axes[1,col].plot([0,1],[0,1],color='gray',ls=':',lw=1)
 axes[0,0].legend(fontsize=8,loc='lower left')
 fig.text(.02,.012,'* Documented supervised-data overlap; common-subset results are reported separately.',fontsize=9,color='#526071');fig.tight_layout(rect=(0,.03,1,1))
 for ext in ['png','pdf']:fig.savefig(ROOT/'results'/('curves.'+ext),dpi=180,bbox_inches='tight')
 plt.close(fig)

def main():
 threadpool_limits(1);collection=read(ROOT/'results/collection.json');assert collection['complete'] and set(collection['models'])==set(NAMES)
 data={name:load_npz(verify(collection['models'][name]['file'])) for name in NAMES};mapping=load_npz(ROOT/'data/pair-mapping.npz');meta=read(ROOT/'data/sequences.json');lengths=np.array(meta['length'])
 exposed=load_npz(ROOT/'provenance/exposure-flags.npz');pair_exposed=exposed['xpair-default__ankh-normalized__pairs']>0;endpoint_exposed=exposed['xpair-default__ankh-normalized__endpoints']>0
 selection=read(ROOT/'provenance/selection.json');allrows=[];subsetrows=[];oprows=[];intervals=[];differences=[];results={};testdata={}
 for test in ['original','ilp']:
  ids=mapping[test];rows=np.load(ROOT/'data'/f'{test}.npy');y=rows[:,2];values={name:{k:v[ids].copy() for k,v in d.items()} for name,d in data.items()}
  if test=='ilp':
   reverse=mapping['ilp_reversed']
   for name in ['native-plm','ipin-esm2','ipin-esmc','native-human']:values[name]['logits'][reverse]=values[name]['logits'][reverse,::-1]
  counts={'pairs':len(rows),'positives':int(y.sum()),'negatives':int((1-y).sum()),'unique_proteins':len(np.unique(rows[:,:2])),
   'combined_over_2193':int((lengths[rows[:,0]]+lengths[rows[:,1]]>2193).sum()),'rapppid_capped_pairs':int(((lengths[rows[:,0]]>1500)|(lengths[rows[:,1]]>1500)).sum())}
  results[test]={'counts':counts,'models':{}};testdata[test]={'rows':rows,'data':values}
  combined=lengths[rows[:,0]]+lengths[rows[:,1]]
  groups={'combined_at_most_2193':combined<=2193,'combined_over_2193':combined>2193,'rapppid_uncapped':(lengths[rows[:,0]]<=1500)&(lengths[rows[:,1]]<=1500),
   'xpair_default_pair_unexposed':~pair_exposed[ids],'xpair_default_endpoint_unexposed':(~endpoint_exposed[rows[:,0]])&(~endpoint_exposed[rows[:,1]])}
  dp=exposed['dscript__exact__pairs']>0;de=exposed['dscript__exact__endpoints']>0
  groups['dscript_pair_unexposed']=~dp[ids]
  groups['dscript_endpoint_unexposed']=(~de[rows[:,0]])&(~de[rows[:,1]])
  extra=load_npz(verify(read(ROOT/'provenance/human-releases-exposure.json')['flags']))
  hp=extra['pairs']>0;he=extra['endpoints']>0
  groups['human_releases_pair_unexposed']=~hp[ids]
  groups['human_releases_endpoint_unexposed']=(~he[rows[:,0]])&(~he[rows[:,1]])
  groups['human_training_length_range']=(lengths[rows[:,0]]>=50)&(lengths[rows[:,0]]<=800)&(lengths[rows[:,1]]>=50)&(lengths[rows[:,1]]<=800)
  for name in NAMES:
   d=values[name];m=metrics(y,d['scores'],d.get('probabilities'));results[test]['models'][name]=m;allrows.append({'test':test,'model':name,**counts,**m})
   if 'logits' in d:
    m['original_order_ap']=float(average_precision_score(y,d['logits'][:,0]));m['original_order_auroc']=float(roc_auc_score(y,d['logits'][:,0]))
    m['mean_absolute_AB_BA_probability_difference']=float(np.mean(np.abs(expit(d['logits'][:,0])-expit(d['logits'][:,1]))))
   for group,mask in groups.items():
    assert len(np.unique(y[mask]))==2
    subsetrows.append({'test':test,'subset':group,'model':name,'rows':int(mask.sum()),'positives':int(y[mask].sum()),'prevalence':float(y[mask].mean()),
     **metrics(y[mask],d['scores'][mask],None if 'probabilities' not in d else d['probabilities'][mask])})
   if 'probabilities' in d:oprows.append({'test':test,'model':name,'rule':'fixed probability 0.5',**operating(y,d['probabilities'],.5)})
   if name in selection['models']:
    threshold=selection['models'][name]['operating_point']['logit'];oprows.append({'test':test,'model':name,'rule':'max F1 on own frozen DEV',**operating(y,d['scores'],threshold)})
  gold=load_npz(ROOT/'data/native-original-reused.npz')['predictions']
  if test=='original':
   assert np.array_equal(values['native-plm']['logits'],gold[:,2:4]);assert results[test]['models']['native-plm']['ap']==average_precision_score(y,gold[:,2:4].mean(1))
  # Compressed, human-readable per-pair output; prediction key is source row, not label or sorting order.
  path=ROOT/'results'/(test+'-predictions.csv.gz')
  with gzip.open(path,'wt',newline='') as f:
   fields=['row_id','source_row_id','union_id','protein_a','protein_b','label','length_a','length_b']
   for name in NAMES:
    fields+=[name+'_score']
    if 'probabilities' in values[name]:fields+=[name+'_probability']
    if 'logits' in values[name]:fields+=[name+'_AB_logit',name+'_BA_logit']
   writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader()
   for i,(a,b,label,source) in enumerate(rows):
    r={'row_id':i,'source_row_id':int(source),'union_id':int(ids[i]),'protein_a':meta['accessions'][a][0],'protein_b':meta['accessions'][b][0],'label':int(label),'length_a':int(lengths[a]),'length_b':int(lengths[b])}
    for name in NAMES:
     d=values[name];r[name+'_score']=float(d['scores'][i])
     if 'probabilities' in d:r[name+'_probability']=float(d['probabilities'][i])
     if 'logits' in d:r[name+'_AB_logit']=float(d['logits'][i,0]);r[name+'_BA_logit']=float(d['logits'][i,1])
    writer.writerow(r)
  results[test]['protein_macro']=macro(rows,values)
  ci,diff,boot=bootstrap(test,rows,values);intervals+=ci;differences+=diff;results[test]['bootstrap']=boot
  print({'test':test,'metrics':{n:{k:results[test]['models'][n][k] for k in ['ap','auroc']} for n in NAMES}},flush=True)
 csvfile(ROOT/'results/metrics.csv',allrows);csvfile(ROOT/'results/subsets.csv',subsetrows);csvfile(ROOT/'results/operating-points.csv',oprows)
 csvfile(ROOT/'results/confidence-intervals.csv',intervals);csvfile(ROOT/'results/paired-differences.csv',differences)
 csvfile(ROOT/'results/protein-macro.csv',[{'test':t,'model':n,**results[t]['protein_macro'][n]} for t in results for n in NAMES])
 plots(results,intervals,testdata)
 output={'at_utc':now(),'complete':True,'names':NAMES,'tests':results,'confidence_intervals':intervals,'paired_differences':differences,'collection':record(ROOT/'results/collection.json'),
  'protocol':record(ROOT/'PROTOCOL.md'),'analysis_script':record(__file__),'no_model_or_threshold_selected_on_tests':True,'native_original_predictions_reproduced_exactly':True}
 atomic(ROOT/'results/summary.json',output)
 print('Analysis complete',flush=True)
if __name__=='__main__':main()
