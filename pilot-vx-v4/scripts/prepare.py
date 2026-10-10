"""Verified TRAIN/DEV snapshots, one independent draw, and separate coverage audit."""
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from study import ROOT,PROJECT,config,read,sha,token_hash,atomic,atomic_npz,now,check_budget
from inputs import monomer_reader,independent_pair,candidate_pool,validate_tokens
from diagnostics import describe
from heads import references


def main():
    if (ROOT/'provenance/inputs.json').exists():raise RuntimeError('Inputs already prepared')
    started=time.monotonic();cfg=config();check_budget();v3=PROJECT/'pilot-vx-v3'
    fp=sha(v3/'provenance/freeze.json'); manifest=read(v3/'provenance/freeze.json');complete=read(v3/'results/COMPLETE.json')
    if fp!=cfg['v3_fingerprint'] or complete['fingerprint']!=fp or not complete['verified'] or complete['test_accessed']:
        raise ValueError('Unverified Vx v3 source')
    sources={'pilot-vx-v3/provenance/freeze.json':fp,'pilot-vx-v3/results/COMPLETE.json':sha(v3/'results/COMPLETE.json')}
    def snapshot(path,name,expected):
        if sha(path)!=expected:raise ValueError('Changed reference: '+str(path))
        sources[str(path.relative_to(PROJECT))]=expected;(ROOT/'data'/name).write_bytes(path.read_bytes())
    names=['sample.json','proteins.json','baseline.json','dev-groups.json','monomer-catalog.json','source.npz','source_metadata.json',
           'source_heads.json','source_metrics.json','dev_family_witnesses.json','original_config.json','monomer-digests.json',
           'pairs.json','v2_metrics.json','diagnostics.json','vy_dev_predictions.npz','vy_verification.json','q_reference_features.npz',
           'vz_heads.json','vz_metrics.json','vz_dev_predictions.npz','vz_dev_evidence.npz','vz_verification.json']
    for name in names:
        p=v3/'data'/name;snapshot(p,name,manifest['files'][str(p.relative_to(PROJECT))])
    for name in ['heads.json','metrics.json','dev_predictions.npz','dev_evidence.npz','verification.json']:
        snapshot(v3/'results'/name,'v3_'+name,complete['result_hashes'][name])
    for rel in ['pilot-vx/config.json','pilot-vx/scripts/common.py','pilot-vx/scripts/msa.py','pilot-vx/scripts/encoder.py','pilot-vx/scripts/analyze.py']:
        if sha(PROJECT/rel)!=manifest['files'][rel]:raise ValueError('Original encoder/source changed')
        sources[rel]=manifest['files'][rel]
    sample=read(ROOT/'data/sample.json');old_pairs=read(ROOT/'data/pairs.json');previous=read(ROOT/'data/source_metadata.json')
    proteins=read(ROOT/'data/proteins.json');catalog=read(ROOT/'data/monomer-catalog.json');oc=read(ROOT/'data/original_config.json')
    if len(sample)!=cfg['expected']['pairs'] or len(old_pairs)!=len(sample) or len({r['uid'] for r in sample})!=len(sample):raise ValueError('Wrong fixed cohort')
    for r,p,old in zip(sample,old_pairs,previous):
        if r['split'] not in ('train','val') or any(proteins[str(r[k])]['split']!=r['split'] for k in ('a','b')):raise ValueError('Unexpected split; no TEST input allowed')
        if p['uid']!=r['uid'] or [p['a'],p['b']]!=sorted([r['a'],r['b']]) or p['available']!=(p['gate']>0):raise ValueError('Changed cohort')
        if any(old[k]!=p[k] for k in ('uid','a','b','available','gate','reason')):raise ValueError('Changed original eligibility')
    def c_reference(p):
        path=v3/'features'/(p['uid']+'.json');side=read(path.with_suffix('.sha.json'));rec=read(path)
        if side['fingerprint']!=fp or rec['fingerprint']!=fp or sha(path)!=side['sha256'] or any(rec.get(k)!=v for k,v in p.items()):raise ValueError('Corrupt C feature reference')
        x=np.asarray(rec['C'],dtype=float)
        if x.shape!=(128,) or not np.isfinite(x).all() or (not p['available'] and x.any()):raise ValueError('Invalid C vector')
        return x,side['sha256']
    with ThreadPoolExecutor(max_workers=8) as pool:old_c=list(pool.map(c_reference,old_pairs))
    atomic_npz(ROOT/'data/c_reference_features.npz',uids=np.array([p['uid'] for p in old_pairs]),C=np.array([x for x,_ in old_c]))
    references(ROOT,sample)
    mono=monomer_reader();original_digests=read(ROOT/'data/monomer-digests.json');prior={r['uid']:r for r in previous}
    def one(old):
        check_budget(cfg['budgets']['analysis_reserve_seconds']);p={k:v for k,v in old.items() if not k.startswith('c_')}
        if not p['available']:
            if p['length']>oc['maximum_combined_residues']:return p,None,{'potential':False,'reason':'long_query'}
            if any(str(p[k]) not in catalog or 'path' not in catalog[str(p[k])] for k in ('a','b')):return p,None,{'potential':False,'reason':'missing_cached_monomer'}
            ma,mb=mono(p['a']),mono(p['b'])
            if min(m['mask'].count('*')/len(m['mask']) for m in (ma,mb))<oc['minimum_quality_fraction_per_chain']:return p,None,{'potential':False,'reason':'low_quality_coverage'}
            pa,_=candidate_pool(ma,oc['minimum_homolog_query_coverage']);pb,_=candidate_pool(mb,oc['minimum_homolog_query_coverage'])
            n=min(len(pa),len(pb),127)
            return p,None,{'potential':n>=oc['minimum_paired_homologs'],'reason':'independently_available' if n>=oc['minimum_paired_homologs'] else 'shallow_independent_pool','possible_depth':n+1}
        if any(str(p[k]) not in original_digests for k in ('a','b')):raise ValueError('Original eligible monomer lacks historical hash')
        src=v3/old['c_input_file'];rel=str(src.relative_to(PROJECT))
        if sha(src)!=old['c_input_sha256'] or manifest['files'][rel]!=old['c_input_sha256']:raise ValueError('Changed original token source')
        with np.load(src,allow_pickle=False) as f:t=f['true'].copy()
        validate_tokens(t,p['breakpoint'])
        if token_hash(t)!=p['paired_tokens_sha256'] or token_hash(t[:1])!=p['q_tokens_sha256'] or t.shape!=(p['paired_depth'],p['length']):raise ValueError('Original token mismatch')
        ma,mb=mono(p['a']),mono(p['b']);i,selected,pools=independent_pair(ma,mb,t,p)
        audit=describe(t,i,p,prior[p['uid']],selected,pools,(ma,mb),cfg['independent_sampling']['diagnostic_column_pairs_per_kind'])
        path=ROOT/'data/independent_inputs'/(p['uid']+'.npz')
        if path.exists():raise RuntimeError('Refuse to overwrite prepared I input')
        atomic_npz(path,true=t,I=i)
        p.update({'i_tokens_sha256':token_hash(i),'i_selected_keys':selected,'i_input_file':str(path.relative_to(ROOT)),'i_input_sha256':sha(path)})
        return p,audit,{'potential':True,'reason':'original_eligible','possible_depth':len(t)}
    output=[];audits={};coverage={};done=0
    with ThreadPoolExecutor(max_workers=8) as pool:
        for p,a,cov in pool.map(one,old_pairs):
            output.append(p);coverage[p['uid']]=cov
            if a is not None:audits[p['uid']]=a;done+=1
            if len(output)%100==0:print({'prepared_records':len(output),'independent_inputs':done,'required':4453,'seconds':time.monotonic()-started},flush=True)
    counts={'pairs':len(sample),'train':sum(r['split']=='train' for r in sample),'dev':sum(r['split']=='val' for r in sample),
            'eligible_train':sum(p['available'] and r['split']=='train' for r,p in zip(sample,output)),
            'eligible_dev':sum(p['available'] and r['split']=='val' for r,p in zip(sample,output)),
            **{k:sum(r['role']==k for r in sample) for k in ('calibration','assessment','crossing')}}
    if counts!=cfg['expected'] or done!=4453:raise ValueError('Original fixed cohort changed')
    groups={}
    for name in ('train','val','calibration','assessment','crossing'):
        selected=[(r,p) for r,p in zip(sample,output) if r['split']==name or r['role']==name]
        groups[name]={'pairs':len(selected),'original_eligible':sum(p['available'] for _,p in selected),
            'potential_independent_eligible':sum(coverage[r['uid']]['potential'] for r,_ in selected),
            'newly_potential_only':sum(coverage[r['uid']]['potential'] and not p['available'] for r,p in selected),
            'reasons':dict(Counter(coverage[r['uid']]['reason'] for r,_ in selected))}
    atomic(ROOT/'data/pairs.json',output);atomic(ROOT/'data/input-audits.json',audits);atomic(ROOT/'data/potential-coverage.json',coverage)
    atomic(ROOT/'results/coverage.json',{'fixed_original_cohort':counts,'label_free_potential_coverage':groups,'expanded_predictions_computed':False,'used_for_model_selection':False})
    atomic(ROOT/'provenance/source-artifact-digests.json',{'monomers':mono.observed_digests,'additional_monomers_read_only_for_coverage':[k for k in mono.observed_digests if k not in original_digests],'v3_C_feature_records':{p['uid']:d for p,(_,d) in zip(old_pairs,old_c)},'original_token_files':{str((v3/p['c_input_file']).relative_to(PROJECT)):p['c_input_sha256'] for p in old_pairs if p['available']}})
    image=PROJECT/manifest['image']
    if sha(image)!=manifest['image_sha256']:raise ValueError('Wrong pinned image')
    atomic(ROOT/'provenance/inputs.json',{'at_utc':now(),'source_files':sources,'image':manifest['image'],'image_sha256':manifest['image_sha256'],
        'required_pairs':done,'counts':counts,'verified_C_feature_records':len(old_c),'primary_I_replicate':0,'R_evaluated':False,
        'test_accessed':False,'outcome_heads_fitted':False,'labels_used_for_sampling':False,'seconds':time.monotonic()-started})
    print({'prepared':True,'counts':counts,'seconds':time.monotonic()-started},flush=True)


if __name__=='__main__':main()
