"""Restartable bounded pair workers. Execution failure never becomes fallback."""
import argparse
import fcntl
import json
import os
import time
import traceback
from functools import lru_cache
import numpy as np
from common import ROOT, PROJECT, atomic, config, now, sha, text_hash, verify_freeze
from msa import load_monomer, pair, shuffled, quality_features


def save_record(dest,uid,value,fingerprint):
    p=dest/f'{uid}.json';atomic(p,{**value,'fingerprint':fingerprint})
    atomic(p.with_suffix('.sha.json'),{'sha256':sha(p),'fingerprint':fingerprint})


def load_record(dest,uid,fingerprint):
    p=dest/f'{uid}.json';side=p.with_suffix('.sha.json')
    if not p.exists() or not side.exists():return None
    meta=json.loads(side.read_text())
    if meta['fingerprint']!=fingerprint or sha(p)!=meta['sha256']:
        raise ValueError(f'Corrupt or incompatible feature cache: {uid}')
    value=json.loads(p.read_text())
    if value['fingerprint']!=fingerprint:raise ValueError('Embedded fingerprint mismatch')
    for k,n in [('true',128),('shuffled',128),('quality',68)]:
        a=np.asarray(value[k]);assert a.shape==(n,) and np.isfinite(a).all(),(uid,k)
    return value


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rank',type=int,required=True);ap.add_argument('--world',type=int,default=4)
    ap.add_argument('--seconds',type=float,default=16800);args=ap.parse_args()
    if args.seconds>16800 or args.world!=4 or not 0<=args.rank<4:raise ValueError('Outside authorized production-worker limits')
    started=time.monotonic();cfg=config();fingerprint=verify_freeze()
    dest=ROOT/'features';dest.mkdir(exist_ok=True)
    lock=(dest/f'rank-{args.rank}.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sample=json.loads((ROOT/'data/sample.json').read_text())[args.rank::args.world]
    catalog=json.loads((ROOT/'data/monomer-catalog.json').read_text())
    @lru_cache(maxsize=32)
    def mono(i):
        rec=catalog[str(i)];path=ROOT/rec['path']
        if sha(path)!=rec['sha256']:raise ValueError('Monomer cache checksum mismatch')
        return load_monomer(path)
    encoder=None;complete=0;reasons={}
    try:
        for row in sample:
            if time.monotonic()-started>args.seconds:break
            uid=row['uid'];saved=load_record(dest,uid,fingerprint)
            if saved is None:
                t=time.monotonic();a,b=sorted([row['a'],row['b']])
                value={'uid':uid,'a':a,'b':b,'true':[0.]*128,'shuffled':[0.]*128,'quality':[0.]*68,'gate':0.,'available':False}
                if str(a) not in catalog or str(b) not in catalog:
                    value['reason']='missing_exact_query'
                else:
                    ma,mb=mono(a),mono(b);tokens,meta=pair(ma,mb,cfg);value.update({'reason':meta['reason'],'msa':meta})
                    value['quality']=quality_features(ma,mb,meta).tolist()
                    if tokens is not None:
                        if encoder is None:
                            from encoder import Encoder
                            encoder=Encoder()
                        seed=int(text_hash(f"{cfg['seed']}:{uid}")[:16],16)
                        null,nullmeta=shuffled(tokens,len(ma['query']),meta['taxonomy'],seed)
                        value['true']=encoder.symmetric(tokens,len(ma['query'])).tolist()
                        value['shuffled']=encoder.symmetric(null,len(ma['query'])).tolist()
                        value.update({'gate':meta['gate'],'available':True,'null':nullmeta})
                value.update({'elapsed_seconds':time.monotonic()-t,'at_utc':now()})
                save_record(dest,uid,value,fingerprint);saved=load_record(dest,uid,fingerprint)
            complete+=1;reasons[saved['reason']]=reasons.get(saved['reason'],0)+1
            if complete%20==0:
                atomic(ROOT/f'results/worker-{args.rank}.json',{'at_utc':now(),'complete':complete,'total':len(sample),
                    'seconds':time.monotonic()-started,'reasons':reasons,'fingerprint':fingerprint})
                print(json.dumps({'rank':args.rank,'complete':complete,'total':len(sample)}),flush=True)
    except Exception:
        atomic(ROOT/f'results/worker-{args.rank}-error.json',{'at_utc':now(),'uid':row['uid'] if 'row' in locals() else None,
               'traceback':traceback.format_exc(),'fingerprint':fingerprint,'complete':complete})
        raise
    atomic(ROOT/f'results/worker-{args.rank}.done.json',{'at_utc':now(),'complete':complete,'total':len(sample),
        'budget_stopped':complete!=len(sample),'seconds':time.monotonic()-started,'reasons':reasons,'fingerprint':fingerprint})


if __name__=='__main__':main()
