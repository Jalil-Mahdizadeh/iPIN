"""Clearly provisional point estimates while SPRINT finishes; no model selection."""
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from bench_utils import ROOT,read,load_npz,atomic,now,record
from collect import verify
collection=read(ROOT/'results/collection.json');mapping=load_npz(ROOT/'data/pair-mapping.npz');out={}
for test in ['original','ilp']:
 y=np.load(ROOT/'data'/f'{test}.npy')[:,2];out[test]={}
 for name,item in collection['models'].items():
  z=load_npz(verify(item['file']))['scores'][mapping[test]]
  out[test][name]={'ap':float(average_precision_score(y,z)),'auroc':float(roc_auc_score(y,z))}
atomic(ROOT/'results/provisional-point-estimates.json',{'at_utc':now(),'complete':False,'missing':collection['missing'],'tests':out,'collection':record(ROOT/'results/collection.json')})
print(out,flush=True)
