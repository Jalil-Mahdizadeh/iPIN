"""Exercise real archived inputs across length strata without reading PPI labels."""
import json
import argparse
import time
import numpy as np
import torch
from common import ROOT, atomic, config, now, sha, text_hash
from encoder import Encoder
from msa import load_monomer, pair, shuffled


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--available-prefix',action='store_true');args=ap.parse_args()
    if args.available_prefix:
        catalog={p.name.split('.')[0]:{'path':str(p.relative_to(ROOT))} for p in (ROOT/'data/monomers').glob('*.json.gz')}
    else:
        catalog=json.loads((ROOT/'data/monomer-catalog.json').read_text())
    sample=json.loads((ROOT/'data/sample.json').read_text());proteins=json.loads((ROOT/'data/proteins.json').read_text())
    selected={};examined=0
    for row in sample:
        if row['length']>1536:continue
        group=(row['length']-1)//512
        if group in selected:continue
        a,b=sorted([row['a'],row['b']])
        if str(a) not in catalog or str(b) not in catalog:continue
        paths=[ROOT/catalog[str(i)]['path'] for i in [a,b]]
        digests={str(p.relative_to(ROOT)):sha(p) for p in paths}
        ma,mb=[load_monomer(p) for p in paths]
        assert ma['query']==proteins[str(a)]['sequence'] and mb['query']==proteins[str(b)]['sequence']
        tokens,meta=pair(ma,mb);examined+=1
        if tokens is None:continue
        assert all(sha(ROOT/p)==h for p,h in digests.items())
        selected[group]=(row['uid'],tokens,len(ma['query']),meta,digests)
        if len(selected)==3:break
    if not selected:
        atomic(ROOT/'qualification/real-encoder.json',{'passed':False,'at_utc':now(),'reason':'no eligible paired MSA in sample','examined':examined})
        atomic(ROOT/'results/decision.json',{'status':'no_go_no_eligible_paired_evidence','at_utc':now(),'GPU_study_started':False})
        raise RuntimeError('No eligible paired evidence; no GPU-study submission')
    model=Encoder();cases=[]
    for group,(uid,tokens,bp,meta,digests) in sorted(selected.items()):
        torch.cuda.reset_peak_memory_stats();t=time.monotonic()
        null,nm=shuffled(tokens,bp,meta['taxonomy'],int(text_hash(f"{config()['seed']}:{uid}")[:16],16))
        np.testing.assert_array_equal(null[0],tokens[0])
        np.testing.assert_array_equal(np.sort(null[1:,bp:],axis=0),np.sort(tokens[1:,bp:],axis=0))
        real=model.symmetric(tokens,bp);disrupted=model.symmetric(null,bp);torch.cuda.synchronize()
        case={'uid':uid,'length':tokens.shape[1],'depth':tokens.shape[0],'four_forward_seconds':time.monotonic()-t,
              'peak_gib':torch.cuda.max_memory_allocated()/2**30,'null_changed_fraction':nm['changed_fraction'],
              'finite':bool(np.isfinite(real).all() and np.isfinite(disrupted).all()),
              'real_features_sha256':text_hash(real.tobytes().hex()),'null_features_sha256':text_hash(disrupted.tobytes().hex()),
              'monomer_files':digests}
        assert case['finite'];cases.append(case);print(json.dumps(case),flush=True)
    atomic(ROOT/'qualification/real-encoder.json',{'passed':True,'at_utc':now(),'cases':cases,
        'labels_used':False,'selection':'first eligible sampled pair in each available fixed length band',
        'archive_prefix_qualification':args.available_prefix,
        'code_sha256':{rel:sha(ROOT/rel) for rel in ['config.json','scripts/msa.py','scripts/encoder.py','scripts/qualify_real.py']},
        'examined_pairs':examined,'sample_sha256':sha(ROOT/'data/sample.json')})


if __name__=='__main__':main()
