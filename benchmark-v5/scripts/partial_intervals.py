"""Compute reusable, protocol-fixed neural intervals while native SPRINT finishes."""
from bench_utils import ROOT,atomic,load_npz,now,read
from collect import NAMES,verify
import analyze as analysis
import numpy as np
from threadpoolctl import threadpool_limits
threadpool_limits(1)
c=read(ROOT/'results/collection.json');analysis.NAMES=[n for n in NAMES if n in c['models']]
assert analysis.NAMES[:3]==['native-plm','ipin-esm2','ipin-esmc']
mapping=load_npz(ROOT/'data/pair-mapping.npz');union={n:load_npz(verify(c['models'][n]['file'])) for n in analysis.NAMES};result={}
for test in ['original','ilp']:
 data={n:{k:v[mapping[test]] for k,v in dd.items()} for n,dd in union.items()}
 ci,diff,meta=analysis.bootstrap(test,np.load(ROOT/'data'/f'{test}.npy'),data)
 result[test]={'intervals':ci,'differences':diff,'bootstrap':meta}
 print(test,[x for x in diff if x['model'] in ['ipin-esm2','ipin-esmc'] and x['reference']=='native-plm'],flush=True)
atomic(ROOT/'results/provisional-intervals.json',{'at_utc':now(),'complete':False,'missing':c['missing'],'tests':result})
