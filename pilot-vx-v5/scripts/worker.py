"""All depth-256 arms, exact reuse only for unchanged inputs, original fallback."""
import time,traceback
import numpy as np
from study import ROOT,config,read,atomic,now,verify_freeze,check_budget,load_record,save_record,DIMENSIONS
from inputs import prepared


def main():
    started=time.monotonic();cfg=config();fp=verify_freeze();pairs=read(ROOT/'data/pairs.json')
    with np.load(ROOT/'data/source.npz',allow_pickle=False) as f:
        original={uid:{arm:f[name][j].copy() for arm,name in [('T256','true'),('S256','shuffled'),('P256','quality')]}
                  for j,uid in enumerate(f['uids'])}
    encoder=None;completed=0;encoded=0;reused=0
    try:
        for p in sorted(pairs,key=lambda p:(not p.get('depth_increased',False),not p['available'],-p['length'],p['uid'])):
            check_budget(cfg['budgets']['analysis_reserve_seconds']+cfg['budgets']['shutdown_reserve_seconds'])
            if load_record(p['uid'],fp,p) is not None:raise RuntimeError('Unexpected existing feature; no automatic retry')
            if p['available']:
                values=prepared(p)
                if p['depth_increased']:
                    if encoder is None:
                        from depth_encoder import DepthEncoder
                        encoder=DepthEncoder()
                    features={'P256':values['P256']};timing={'kind':'encoded'}
                    for arm in ('T256','S256'):
                        features[arm],timing[arm]=encoder.at_depth(values[arm],p['breakpoint'],p['depth_256'])
                    encoded+=1
                else:
                    features=original[p['uid']];np.testing.assert_array_equal(features['P256'],values['P256'])
                    timing={'kind':'reused_unchanged_depth'};reused+=1
            else:features={k:np.zeros(n) for k,n in DIMENSIONS.items()};timing=None
            save_record(p['uid'],features,fp,p,timing);completed+=1
            if completed%20==0:
                progress={'at_utc':now(),'completed':completed,'total':len(pairs),'encoded_pairs':encoded,
                    'encoded_arms':2*encoded,'unchanged_depth_reused':reused,'current_length':p['length'],
                    'seconds':time.monotonic()-started,'fingerprint':fp}
                atomic(ROOT/'results/worker.json',progress);print(progress,flush=True)
    except BaseException:
        atomic(ROOT/'results/worker-error.json',{'at_utc':now(),'uid':p['uid'] if 'p' in locals() else None,
            'completed':completed,'encoded_pairs':encoded,'traceback':traceback.format_exc()});raise
    atomic(ROOT/'results/worker.done.json',{'at_utc':now(),'completed':completed,'total':len(pairs),'encoded_pairs':encoded,
        'encoded_arms':2*encoded,'unchanged_depth_reused':reused,'eligible_pairs':encoded+reused,
        'budget_stopped':False,'seconds':time.monotonic()-started,'fingerprint':fp})


if __name__=='__main__':main()
