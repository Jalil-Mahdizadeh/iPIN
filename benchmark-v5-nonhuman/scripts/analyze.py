"""Fixed-test comparisons, paired protein bootstrap and transparent diagnostics."""
import csv,gzip,hashlib,json,os,time
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score,roc_auc_score,precision_recall_curve,roc_curve
from threadpoolctl import threadpool_limits
from bench_utils import ROOT,atomic,load_npz,now,read,record,save_npz,sha
from collect import NAMES,LABELS,TESTS,PAIR_MODELS,verify
REPLICATES=1000;SEED=20261005

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
  identity={'model':name,'test':test,'replicates':REPLICATES,'seed':SEED,'rows':hashlib.sha256(rows.tobytes()).hexdigest(),
   'scores':hashlib.sha256(data[name]['scores'].tobytes()).hexdigest(),'analysis_code':sha(__file__)}
  identities[name]=identity;path=cache/f'{test}-{name}.npz';side=path.with_suffix('.json')
  if side.exists():
   item=read(side);previous=item['identity']
   if previous!=identity:
    assert previous['analysis_code']==sha(ROOT/'archive/before-xpair-v11/scripts/analyze.py')
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
 comparisons=[(j,NAMES.index(ref)) for ref in ['native-human','native-plm'] for j,name in enumerate(NAMES) if name!=ref]+[(NAMES.index('ipin-esm2'),NAMES.index('ipin-esmc'))]
 comparisons += [(NAMES.index('xpair-v11'),NAMES.index(ref)) for ref in ['ipin-esm2','ipin-esmc','tuna-human','tuna','xpair-bernett','xpair-default']]
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

SPECIES={'mouse':'Mouse','fly':'Fly','worm':'Worm','yeast':'Yeast','ecoli':'E. coli'}
COLORS=['#c83d43','#2879b9','#111111','#737373','#218657','#82b78e','#7957ad','#b89bd7','#d09b29','#98644a','#cc75a5','#e17616']

def plots(results,intervals,testdata,subsets):
 os.environ['MPLCONFIGDIR']=str(ROOT/'cache/matplotlib')
 import matplotlib;matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 labels=[LABELS[n]+('*' if n=='xpair-default' else '') for n in NAMES]
 assert len(COLORS)==len(NAMES)
 for metric in ['ap','auroc']:
  fig,axes=plt.subplots(1,5,figsize=(19,max(6.4,.46*len(NAMES)+1.2)),sharey=True)
  for ax,test in zip(axes,TESTS):
   for j,name in enumerate(NAMES):
    it=next(x for x in intervals if x['test']==test and x['model']==name and x['metric']==metric)
    v=it['estimate'];ax.plot([it['low'],it['high']],[j,j],color=COLORS[j],lw=1.6);ax.scatter(v,j,color=COLORS[j],s=33,zorder=3)
    ax.annotate(f'{v:.3f}',(v,j),xytext=(0,-11),textcoords='offset points',ha='center',fontsize=7)
   ax.set(title=SPECIES[test],xlabel='Average precision' if metric=='ap' else 'AUROC',xlim=(0,1.02));ax.grid(axis='x',alpha=.2)
   ax.axvline(1/11 if metric=='ap' else .5,color='gray',ls=':',lw=1)
  axes[0].set_yticks(range(len(labels)),labels);axes[0].invert_yaxis()
  fig.suptitle('Frozen models on the five released nonhuman tests\nFull source rows; 95% paired protein-bootstrap intervals',fontsize=14)
  fig.text(.24,.01,'* X-PAIR default has substantial known nonhuman supervised exposure. See the common unexposed analysis.',fontsize=10)
  fig.tight_layout(rect=(0,.04,1,.93))
  for ext in ['png','pdf','svg']:fig.savefig(ROOT/'results'/f'{metric}-comparison.{ext}',dpi=190,bbox_inches='tight')
  plt.close(fig)
 # Identical unexposed rows for every predictor; do not compare model-specific subsets.
 fig,axes=plt.subplots(1,2,figsize=(15,max(8,.5*len(NAMES)+1.5)),sharey=True)
 for ax,metric in zip(axes,['ap','auroc']):
  matrix=np.array([[next(r[metric] for r in subsets if r['subset']=='common_known_endpoint_unexposed' and r['test']==t and r['model']==n) for t in TESTS] for n in NAMES])
  im=ax.imshow(matrix,cmap='YlGnBu',vmin=0,vmax=1,aspect='auto')
  for j in range(len(NAMES)):
   for k in range(5):ax.text(k,j,f'{matrix[j,k]:.3f}',ha='center',va='center',color='white' if matrix[j,k]>.65 else 'black',fontsize=10)
  ax.set_xticks(range(5),[SPECIES[t] for t in TESTS]);ax.set_yticks(range(len(labels)),labels);ax.set_title('Average precision' if metric=='ap' else 'AUROC')
  fig.colorbar(im,ax=ax,shrink=.6)
 counts=[next(r for r in subsets if r['subset']=='common_known_endpoint_unexposed' and r['test']==t) for t in TESTS]
 caption='; '.join(f"{SPECIES[t]}: {c['rows']:,} pairs, {c['positives']:,} positives" for t,c in zip(TESTS,counts))
 fig.suptitle('Common subset after removing every known exposed endpoint',fontsize=14)
 fig.text(.15,.035,caption,fontsize=9)
 fig.text(.15,.01,'Known source files only; RAPPPID and X-PAIR Bernett checkpoint membership is not fully established. Homologs remain.',fontsize=9)
 fig.tight_layout(rect=(0,.07,1,.96))
 for ext in ['png','pdf','svg']:fig.savefig(ROOT/'results'/f'common-unexposed-comparison.{ext}',dpi=190,bbox_inches='tight')
 plt.close(fig)
 fig,axes=plt.subplots(2,5,figsize=(20,8))
 for col,test in enumerate(TESTS):
  y=testdata[test]['rows'][:,2]
  for j,name in enumerate(NAMES):
   z=testdata[test]['data'][name]['scores'];p,r,_=precision_recall_curve(y,z);fpr,tpr,_=roc_curve(y,z)
   axes[0,col].plot(r,p,color=COLORS[j],label=labels[j],lw=1.5 if j<3 else .9,alpha=1 if j<3 else .85)
   axes[1,col].plot(fpr,tpr,color=COLORS[j],lw=1.5 if j<3 else .9,alpha=1 if j<3 else .85)
  axes[0,col].set(title=SPECIES[test],xlabel='Recall',ylabel='Precision',xlim=(0,1),ylim=(0,1));axes[0,col].axhline(y.mean(),color='gray',ls=':',lw=1)
  axes[1,col].set(xlabel='False-positive rate',ylabel='True-positive rate',xlim=(0,1),ylim=(0,1));axes[1,col].plot([0,1],[0,1],color='gray',ls=':',lw=1)
 handles,labels_=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels_,loc='lower center',ncol=4,fontsize=9)
 fig.tight_layout(rect=(0,.19,1,.96))
 for ext in ['png','pdf']:fig.savefig(ROOT/'results'/f'curves.{ext}',dpi=180,bbox_inches='tight')
 plt.close(fig)

def main():
 threadpool_limits(1);collection=read(ROOT/'results/collection.json');assert collection['complete'] and set(collection['models'])==set(NAMES)
 data={n:load_npz(verify(collection['models'][n]['file'])) for n in NAMES};mapping=load_npz(ROOT/'data/pair-mapping.npz')
 meta=read(ROOT/'data/sequences.json');lengths=np.array(meta['length']);exposure=load_npz(ROOT/'provenance/exposure-flags.npz')
 audits=read(ROOT/'provenance/exposure.json')['audits'];known_endpoint=np.zeros(len(lengths),bool);human_endpoint=np.zeros(len(lengths),bool)
 # Use the union of known sources, including raw Bernett source conservatively.
 for key in audits:
  known_endpoint |= exposure[key+'__endpoints']>0
  if not key.startswith('xpair-default'):human_endpoint |= exposure[key+'__endpoints']>0
 intervals=[];differences=[];allmetrics=[];subsets=[];order_metrics=[];results={};testdata={};unexposed_counts={}
 for test in TESTS:
  rows=np.load(ROOT/'data'/f'{test}.npy');ids=mapping[test];y=rows[:,2];reverse=mapping[test+'_reversed']
  values={n:{k:v[ids].copy() for k,v in d.items()} for n,d in data.items()}
  for n in PAIR_MODELS:values[n]['logits'][reverse]=values[n]['logits'][reverse,::-1]
  counts={'rows':len(rows),'positives':int(y.sum()),'negatives':int((1-y).sum()),'unique_proteins':len(np.unique(rows[:,:2]))}
  results[test]={'counts':counts,'models':{}};testdata[test]={'rows':rows,'data':values}
  low=np.zeros(len(rows),bool);first={};conflicts=set()
  for i,uid in enumerate(ids):
   if int(uid) in first:
    if y[first[int(uid)]]!=y[i]:conflicts.add(int(uid))
   else:first[int(uid)]=i
  for uid,i in first.items():
   if uid not in conflicts:low[i]=True
  common=~(known_endpoint[rows[:,0]]|known_endpoint[rows[:,1]])
  human_clean=~(human_endpoint[rows[:,0]]|human_endpoint[rows[:,1]])
  groups={'unique_pairs_without_conflicts':low,'common_known_endpoint_unexposed':common,'common_human_sources_endpoint_unexposed':human_clean}
  for key in ['v5-ipin__exact','native-human__exact','xpair-default__ankh-normalized','dscript__exact']:
   seen=exposure[key+'__endpoints']>0;groups[key+'_endpoint_unexposed']=~(seen[rows[:,0]]|seen[rows[:,1]])
  unexposed_counts[test]={'rows':int(common.sum()),'positives':int(y[common].sum()),'negatives':int((1-y[common]).sum()),'unique_proteins':len(np.unique(rows[common,:2]))}
  for name in NAMES:
   d=values[name];m=metrics(y,d['scores'],d.get('probabilities'));results[test]['models'][name]=m;allmetrics.append({'test':test,'model':name,**counts,**m})
   if 'logits' in d:
    order_metrics.append({'test':test,'model':name,'original_order_ap':float(average_precision_score(y,d['logits'][:,0])),
     'original_order_auroc':float(roc_auc_score(y,d['logits'][:,0])),
     'mean_absolute_AB_BA_probability_difference':float(abs(expit(d['logits'][:,0])-expit(d['logits'][:,1])).mean())})
   for group,mask in groups.items():
    assert len(np.unique(y[mask]))==2,(test,group)
    subsets.append({'test':test,'subset':group,'model':name,'rows':int(mask.sum()),'positives':int(y[mask].sum()),'negatives':int((1-y[mask]).sum()),'prevalence':float(y[mask].mean()),
     **metrics(y[mask],d['scores'][mask],None if 'probabilities' not in d else d['probabilities'][mask])})
  ci,di,bo=bootstrap(test,rows,values);intervals+=ci;differences+=di;results[test]['bootstrap']=bo
  clean={n:{k:v[common] for k,v in d.items()} for n,d in values.items()}
  ci,di,bo=bootstrap(test+'__common_unexposed',rows[common],clean);intervals+=ci;differences+=di;results[test]['unexposed_bootstrap']=bo
  # Retain source order and exact sequence hashes; no accession identities were supplied.
  path=ROOT/'results'/f'{test}-predictions.csv.gz'
  with gzip.open(path,'wt',newline='') as stream:
   fields=['source_row_id','union_id','sequence_a_sha256','sequence_b_sha256','label','length_a','length_b','common_known_endpoint_unexposed']
   for n in NAMES:
    fields.append(n+'_score')
    if 'probabilities' in values[n]:fields.append(n+'_probability')
    if 'logits' in values[n]:fields.extend([n+'_AB_logit',n+'_BA_logit'])
   writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
   for i,(a,b,label,source) in enumerate(rows):
    r={'source_row_id':int(source),'union_id':int(ids[i]),'sequence_a_sha256':meta['sha256'][a],'sequence_b_sha256':meta['sha256'][b],
     'label':int(label),'length_a':int(lengths[a]),'length_b':int(lengths[b]),'common_known_endpoint_unexposed':int(common[i])}
    for n in NAMES:
     d=values[n];r[n+'_score']=float(d['scores'][i])
     if 'probabilities' in d:r[n+'_probability']=float(d['probabilities'][i])
     if 'logits' in d:r[n+'_AB_logit']=float(d['logits'][i,0]);r[n+'_BA_logit']=float(d['logits'][i,1])
    writer.writerow(r)
  print({'test':test,'metrics':{n:{k:results[test]['models'][n][k] for k in ['ap','auroc']} for n in NAMES}},flush=True)
 csvfile(ROOT/'results/metrics.csv',allmetrics);csvfile(ROOT/'results/subsets.csv',subsets);csvfile(ROOT/'results/orientation-metrics.csv',order_metrics)
 csvfile(ROOT/'results/confidence-intervals.csv',intervals);csvfile(ROOT/'results/paired-differences.csv',differences)
 plots(results,intervals,testdata,subsets)
 # Independently archived original-order humanV11 scores are a separate convention.
 references=[]
 for test in TESTS:
  path=ROOT.parent/'plm-interact-reproducability/results'/f'cross_{test}.csv'
  with path.open() as stream:old=list(csv.DictReader(stream))
  rows=testdata[test]['rows'];assert np.array_equal([int(r['row_id']) for r in old],rows[:,3]) and np.array_equal([int(r['label']) for r in old],rows[:,2])
  z=np.array([float(r['logit']) for r in old]);fresh=testdata[test]['data']['native-human']['logits'][:,0]
  references.append({'test':test,'source':record(path),'metrics':metrics(rows[:,2],z,expit(z)),
   'fresh_original_order_max_logit_error':float(abs(z-fresh).max()),'fresh_original_order_max_probability_error':float(abs(expit(z)-expit(fresh)).max())})
 atomic(ROOT/'results/paper-convention-native-reference.json',references)
 atomic(ROOT/'results/summary.json',{'at_utc':now(),'complete':True,'names':NAMES,'tests':results,'common_unexposed_counts':unexposed_counts,
  'confidence_intervals':intervals,'paired_differences':differences,'collection':record(ROOT/'results/collection.json'),
  'protocol':record(ROOT/'PROTOCOL.md'),'analysis_script':record(__file__),'no_model_or_threshold_selected_on_tests':True})
 print('All five species analyzed',flush=True)

if __name__=='__main__':main()
