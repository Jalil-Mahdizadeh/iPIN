"""TRAIN/DEV snapshots and exact nested depth extension using unmodified Vx pairing."""
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from study import ROOT,PROJECT,config,read,sha,token_hash,atomic,atomic_npz,now,check_budget
from inputs import monomer_reader,extend,quality_at_depth,vector_hash,validate_tokens
from diagnostics import describe
from heads import references

STATE=None


def initialize():
    global STATE
    with np.load(ROOT/'data/source.npz',allow_pickle=False) as f:quality={uid:x.copy() for uid,x in zip(f['uids'],f['quality'])}
    STATE={'mono':monomer_reader(),'old':{p['uid']:p for p in read(ROOT/'data/source_metadata.json')},
           'quality':quality,'original':read(ROOT/'data/original_config.json'),
           'v4':read(ROOT/'data/v4_source_manifest.json')}


def one(old):
    check_budget(config()['budgets']['analysis_reserve_seconds']);p={k:v for k,v in old.items() if not k.startswith('i_')}
    if not p['available']:return p,None
    previous=STATE['old'][p['uid']];src=PROJECT/'pilot-vx-v4'/old['i_input_file']
    if sha(src)!=old['i_input_sha256'] or STATE['v4']['files'][str(src.relative_to(PROJECT))]!=old['i_input_sha256']:
        raise ValueError('Changed original token source')
    with np.load(src,allow_pickle=False) as f:original=f['true'].copy()
    validate_tokens(original,p['breakpoint'])
    if token_hash(original)!=p['paired_tokens_sha256'] or original.shape!=(p['paired_depth'],p['length']):raise ValueError('Original input differs')
    t,s,meta,null=extend(STATE['mono'](p['a']),STATE['mono'](p['b']),original,previous['msa'],p,STATE['original'])
    q=quality_at_depth(STATE['quality'][p['uid']],previous['msa'],meta)
    audit=describe(t,s,p,previous['msa'],meta,null)
    path=ROOT/'data/depth_inputs'/(p['uid']+'.npz')
    if path.exists():raise RuntimeError('Refuse to overwrite depth input')
    atomic_npz(path,T256=t,S256=s,P256=q)
    p.update({'depth_256':len(t),'depth_increased':len(t)>len(original),'msa256':meta,'null256':null,
              'T256_tokens_sha256':token_hash(t),'S256_tokens_sha256':token_hash(s),'P256_sha256':vector_hash(q),
              'depth_input_file':str(path.relative_to(ROOT)),'depth_input_sha256':sha(path)})
    if not p['depth_increased'] and (meta!=previous['msa'] or null!=previous['null']):raise ValueError('Unchanged-depth metadata differs')
    return p,audit


def main():
    if (ROOT/'provenance/inputs.json').exists():raise RuntimeError('Inputs already prepared')
    started=time.monotonic();cfg=config();check_budget();v4=PROJECT/'pilot-vx-v4'
    manifest=read(v4/'provenance/freeze.json');fp=sha(v4/'provenance/freeze.json');complete=read(v4/'results/COMPLETE.json')
    if fp!=cfg['v4_fingerprint'] or complete['fingerprint']!=fp or not complete['verified'] or complete['test_accessed']:
        raise ValueError('Unverified Vx-v4 snapshot source')
    sources={'pilot-vx-v4/provenance/freeze.json':fp,'pilot-vx-v4/results/COMPLETE.json':sha(v4/'results/COMPLETE.json')}
    atomic(ROOT/'data/v4_source_manifest.json',manifest)
    names=['sample.json','proteins.json','baseline.json','dev-groups.json','monomer-catalog.json','source.npz',
           'source_metadata.json','source_heads.json','source_metrics.json','dev_family_witnesses.json',
           'original_config.json','monomer-digests.json','pairs.json','v2_metrics.json','diagnostics.json',
           'vy_dev_predictions.npz','vy_verification.json']
    for name in names:
        src=v4/'data'/name;rel=str(src.relative_to(PROJECT));expected=manifest['files'][rel]
        if sha(src)!=expected:raise ValueError('Changed TRAIN/DEV source: '+rel)
        (ROOT/'data'/name).write_bytes(src.read_bytes());sources[rel]=expected
    for rel in ['pilot-vx/config.json','pilot-vx/scripts/common.py','pilot-vx/scripts/msa.py','pilot-vx/scripts/encoder.py','pilot-vx/scripts/analyze.py']:
        if sha(PROJECT/rel)!=manifest['files'][rel]:raise ValueError('Original scientific code changed')
        sources[rel]=manifest['files'][rel]
    if not read(ROOT/'data/vy_verification.json')['passed']:raise ValueError('Unverified prediction reference')
    sample=read(ROOT/'data/sample.json');old_pairs=read(ROOT/'data/pairs.json');previous=read(ROOT/'data/source_metadata.json')
    proteins=read(ROOT/'data/proteins.json')
    if len(sample)!=cfg['expected']['pairs'] or len(old_pairs)!=len(sample) or len({r['uid'] for r in sample})!=len(sample):raise ValueError('Wrong fixed cohort')
    for r,p,old in zip(sample,old_pairs,previous):
        if r['split'] not in ('train','val') or any(proteins[str(r[k])]['split']!=r['split'] for k in ('a','b')):raise ValueError('Forbidden split')
        if p['uid']!=r['uid'] or [p['a'],p['b']]!=sorted([r['a'],r['b']]) or p['available']!=(p['gate']>0):raise ValueError('Changed cohort')
        if any(old[k]!=p[k] for k in ('uid','a','b','available','gate','reason')):raise ValueError('Changed original eligibility')
    references(ROOT,sample)
    output=[];audits={}
    with ProcessPoolExecutor(max_workers=cfg['preparation_workers'],initializer=initialize) as pool:
        for p,a in pool.map(one,old_pairs,chunksize=4):
            output.append(p)
            if a is not None:audits[p['uid']]=a
            if len(output)%100==0:print({'prepared_records':len(output),'eligible_inputs':len(audits),'seconds':time.monotonic()-started},flush=True)
    counts={'pairs':len(sample),'train':sum(r['split']=='train' for r in sample),'dev':sum(r['split']=='val' for r in sample),
            'eligible_train':sum(p['available'] and r['split']=='train' for r,p in zip(sample,output)),
            'eligible_dev':sum(p['available'] and r['split']=='val' for r,p in zip(sample,output)),
            **{k:sum(r['role']==k for r in sample) for k in ('calibration','assessment','crossing')}}
    if counts!=cfg['expected'] or len(audits)!=4453:raise ValueError('Fixed cohort changed')
    depth={name:{'eligible':sum(p['available'] and keep(r) for r,p in zip(sample,output)),
        'unchanged_depth':sum(p['available'] and not p['depth_increased'] and keep(r) for r,p in zip(sample,output)),
        'increased_depth':sum(p['available'] and p['depth_increased'] and keep(r) for r,p in zip(sample,output)),
        'reached_256':sum(p['available'] and p['depth_256']==256 and keep(r) for r,p in zip(sample,output))}
        for name,keep in [('train',lambda r:r['split']=='train'),('dev',lambda r:r['split']=='val'),('assessment',lambda r:r['role']=='assessment')]}
    atomic(ROOT/'data/pairs.json',output);atomic(ROOT/'data/input-audits.json',audits)
    atomic(ROOT/'results/depth-coverage.json',{'cohort':counts,'depth_groups':depth,'eligibility_expanded':False,'test_accessed':False})
    digests=read(ROOT/'data/monomer-digests.json');used={str(p[k]) for p in output if p['available'] for k in ('a','b')}
    atomic(ROOT/'provenance/source-artifact-digests.json',{'monomers':{k:digests[k] for k in sorted(used)},
        'original_token_files':{str((v4/p['i_input_file']).relative_to(PROJECT)):p['i_input_sha256'] for p in old_pairs if p['available']}})
    if sha(PROJECT/manifest['image'])!=manifest['image_sha256']:raise ValueError('Pinned SIF differs')
    atomic(ROOT/'provenance/inputs.json',{'at_utc':now(),'source_files':sources,'image':manifest['image'],'image_sha256':manifest['image_sha256'],
        'required_pairs':len(audits),'counts':counts,'increased_depth_pairs':sum(p.get('depth_increased',False) for p in output),
        'test_accessed':False,'outcome_heads_fitted':False,'R_evaluated':False,'labels_used_for_sampling':False,'seconds':time.monotonic()-started})
    print({'prepared':True,'depth_groups':depth,'seconds':time.monotonic()-started},flush=True)


if __name__=='__main__':main()
