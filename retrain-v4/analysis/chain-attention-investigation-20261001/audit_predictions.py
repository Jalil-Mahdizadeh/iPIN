"""Read existing full validation predictions; do not run new validation or training."""
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT=Path('/nobackup/proj/disk/theo-storage/personal/jalil/iPIN')
OUT=Path(__file__).resolve().parent
RUNS={'S0':ROOT/'retrain-v3/runs/clean-residue-mean-official-seed2',
      'S1':ROOT/'retrain-v4/runs/esm2-chain-aware-official-seed2',
      'C0':ROOT/'retrain-v4/runs/esmc-standard-official-seed2',
      'C1':ROOT/'retrain-v4/runs/esmc-chain-aware-official-seed2'}
rows=np.load(ROOT/'retrain-v4/data/prepared/official/val.npy')
lengths=np.diff(np.load(ROOT/'retrain-v4/data/prepared/offsets.npy'))
lengths=lengths[rows[:,0]]+lengths[rows[:,1]]+3

def stats(y,z):
    p=expit(z)
    return {'n':len(y),'positives':int(y.sum()),'ap':float(average_precision_score(y,z)),
            'auroc':float(roc_auc_score(y,z)),'bce':float((np.logaddexp(0,z)-y*z).mean()),
            'probability_mean':float(p.mean()),'probability_std':float(p.std()),
            'probability_q01_q50_q99':np.quantile(p,[.01,.5,.99]).tolist(),
            'positive_mean_probability':float(p[y==1].mean()),'negative_mean_probability':float(p[y==0].mean()),
            'logit_std':float(z.std()),'distinct_pooled_logits':len(np.unique(z))}

report={'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':{},'test_data_used':False}
for arm,run in RUNS.items():
    vals=[]
    for p in sorted((run/'validation').glob('update-*.json')):
        meta=json.loads(p.read_text()); f=p.with_suffix('.npz')
        assert hashlib.sha256(f.read_bytes()).hexdigest()==meta['sha256']
        a=np.load(f,allow_pickle=False)['predictions']; assert np.array_equal(a[:,1],rows[:,2])
        y,z=a[:,1],a[:,2:4].mean(1)
        all_stats=stats(y,z)
        assert abs(all_stats['ap']-meta['metrics']['pooled_ap'])<1e-12
        assert abs(all_stats['auroc']-meta['metrics']['auroc'])<1e-12
        groups={}
        for name,mask in [('up_to_512',lengths<=512),('513_to_2196',(lengths>512)&(lengths<=2196)),('above_2196',lengths>2196)]:
            groups[name]=stats(y[mask],z[mask])
        vals.append({'update':meta['update'],'sha256':meta['sha256'],'all':all_stats,'length_groups':groups,
                     'ab_ba_logit_correlation':float(np.corrcoef(a[:,2],a[:,3])[0,1]),
                     'absolute_orientation_logit_gap_mean':float(np.abs(a[:,2]-a[:,3]).mean())})
    events=[json.loads(line) for line in (run/'events.jsonl').read_text().splitlines() if line.strip().endswith('}')]
    updates={e['update']:e for e in events if e.get('event')=='update'}
    bce_windows={}
    for end in [1000,2000,3000,4000,5000,6000]:
        xs=[e for k,e in updates.items() if end-500<k<=end]
        if xs:bce_windows[str(end)]={'n_logged_batches':len(xs),'mean_bce':float(np.mean([e['mean_bce'] for e in xs])),
                    'median_preclip_gradient_norm':float(np.median([e['grad_norm'] for e in xs]))}
    report['runs'][arm]={'path':str(run),'status':json.loads((run/'status.json').read_text()),
                        'validations':vals,'training_bce_last_500_update_windows':bce_windows}
    print(arm,json.dumps([{'step':v['update'],**v['all']} for v in vals]),flush=True)
(OUT/'prediction-audit.json').write_text(json.dumps(report,indent=2)+'\n')
