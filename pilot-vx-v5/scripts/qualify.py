"""Exact original replay, depth-256 execution guards, and bounded two-arm timing."""
import time
import numpy as np
import torch
from study import ROOT,config,read,atomic,now,sha,check_budget
from inputs import prepared,monomer_reader
from depth_encoder import DepthEncoder
from msa import shuffled,quality_features
from common import text_hash


def main():
    started=time.monotonic();cfg=config();check_budget()
    if (ROOT/'qualification/real.json').exists():raise RuntimeError('Qualification already completed')
    pairs=read(ROOT/'data/pairs.json');lookup={p['uid']:p for p in pairs};changed=[p for p in pairs if p.get('depth_increased',False)]
    if not changed:raise ValueError('No deeper inputs; no depth experiment possible')
    maximum_depth=max(p['depth_256'] for p in changed)
    if maximum_depth!=256:raise ValueError('No real depth-256 example for qualification')
    deepest=[p for p in changed if p['depth_256']==256];uids=list(cfg['qualification_uids'])
    for length in cfg['timing_length_targets']:
        uid=min(deepest,key=lambda p:(abs(p['length']-length),p['uid']))['uid']
        if uid not in uids:uids.append(uid)
    previous={p['uid']:p for p in read(ROOT/'data/source_metadata.json')};mono=monomer_reader()
    with np.load(ROOT/'data/source.npz',allow_pickle=False) as f:old={k:f[k].copy() for k in ('uids','true','shuffled','quality')}
    index={uid:i for i,uid in enumerate(old['uids'])};enc=DepthEncoder();cases=[]
    for uid in uids:
        check_budget();p=lookup[uid];before=time.monotonic();v=prepared(p);io=time.monotonic()-before
        t128=v['T256'][:p['paired_depth']];seed=int(text_hash(f"{cfg['seed']}:{uid}")[:16],16)
        s128,null128=shuffled(t128,p['breakpoint'],previous[uid]['msa']['taxonomy'],seed)
        if null128!=previous[uid]['null']:raise ValueError('Original shuffle changed')
        oldtiming={};errors={}
        for arm,t in [('true',t128),('shuffled',s128)]:
            x,timing=enc.at_depth(t,p['breakpoint'],len(t));oldtiming[arm]=timing
            errors[arm]=float(np.max(np.abs(x-old[arm][index[uid]])))
            np.testing.assert_array_equal(x,old[arm][index[uid]])
        ma,mb=mono(p['a']),mono(p['b'])
        np.testing.assert_array_equal(quality_features(ma,mb,previous[uid]['msa']),old['quality'][index[uid]])
        np.testing.assert_array_equal(quality_features(ma,mb,p['msa256']),v['P256'])
        timings={}
        for arm in ('T256','S256'):
            a,ta=enc.at_depth(v[arm],p['breakpoint'],p['depth_256']);b,tb=enc.at_depth(v[arm],p['breakpoint'],p['depth_256'])
            np.testing.assert_array_equal(a,b);timings[arm]=[ta,tb]
            if uid in cfg['qualification_uids']:
                reverse=np.concatenate([v[arm][:,p['breakpoint']:],v[arm][:,:p['breakpoint']]],axis=1)
                other,_=enc.at_depth(reverse,p['length']-p['breakpoint'],p['depth_256']);np.testing.assert_array_equal(a,other)
        cases.append({'uid':uid,'length':p['length'],'paired_depth':p['paired_depth'],'depth_256':p['depth_256'],
            'T256_tokens_sha256':p['T256_tokens_sha256'],'S256_tokens_sha256':p['S256_tokens_sha256'],
            'prepared_input_read_seconds':io,'timings':timings,'depth128_timings':oldtiming,'original_max_error':errors,
            'deterministic_repeat_exact':True,'quality_features_match_original_constructor':True,
            'ab_ba_symmetry_checked':uid in cfg['qualification_uids']})
        print({'qualified':uid,'length':p['length'],'depth':p['depth_256'],'both_arm_seconds':sum(max(t['seconds'] for t in timings[a]) for a in timings)},flush=True)
    buckets=[]
    for c in sorted((c for c in cases if c['depth_256']==256),key=lambda c:c['length']):
        secs=max(v['prepared_input_read_seconds']+sum(max(t['seconds'] for t in v['timings'][a]) for a in ('T256','S256')) for v in cases if v['length']<=c['length'])
        buckets.append((c['length'],secs))
    if buckets[-1][0]<max(p['length'] for p in changed):raise ValueError('Missing maximal length/depth timing coverage')
    estimate=sum(next(secs for length,secs in buckets if length>=p['length']) for p in changed)
    total=cfg['timing_safety_factor']*estimate+cfg['budgets']['io_reserve_seconds']+cfg['budgets']['analysis_reserve_seconds']
    if total>check_budget(cfg['budgets']['shutdown_reserve_seconds']):raise TimeoutError('Full qualified experiment does not fit remaining budget')
    code=[ROOT/'config.json',ROOT/'scripts/qualify.py',ROOT/'scripts/depth_encoder.py',ROOT/'scripts/inputs.py',ROOT/'scripts/study.py',ROOT/'scripts/diagnostics.py']
    atomic(ROOT/'qualification/real.json',{'at_utc':now(),'passed':True,'labels_used':False,'cases':cases,
        'required_pairs':sum(p['available'] for p in pairs),'increased_depth_pairs':len(changed),'encoder_arm_calls':2*len(changed),
        'estimated_extraction_seconds':estimate,'conservative_total_seconds':total,'timing_safety_factor':cfg['timing_safety_factor'],
        'upper_length_timing_buckets':buckets,'timing_anchor_depth':256,'timing_includes_prepared_input_read':True,
        'gpu':torch.cuda.get_device_name(0),'forbidden_heads_evaluated':False,'R_evaluated':False,
        'code_sha256':{str(p.relative_to(ROOT)):sha(p) for p in code},'seconds':time.monotonic()-started})
    print({'qualified':True,'estimated_hours':estimate/3600,'conservative_hours':total/3600},flush=True)


if __name__=='__main__':main()
