"""Original pairing at a higher cap, exact prefix, and all-input source replay."""
import hashlib
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache
import numpy as np
from study import ROOT, BASE, config, read, sha, token_hash
from msa import load_monomer, pair, shuffled, encode
from common import text_hash


def vector_hash(x):
    return hashlib.sha256(np.asarray(x, dtype='<f8').tobytes()).hexdigest()


def monomer_reader():
    catalog=read(ROOT/'data/monomer-catalog.json');hashes=read(ROOT/'data/monomer-digests.json')
    proteins=read(ROOT/'data/proteins.json')
    @lru_cache(maxsize=256)
    def get(pid):
        item=catalog[str(pid)];path=BASE/item['path'];expected=hashes[str(pid)]
        if item['sha256']!=expected or sha(path)!=expected:raise ValueError('Changed monomer cache')
        m=load_monomer(path)
        if m['query']!=proteins[str(pid)]['sequence'] or m['query_sha256']!=proteins[str(pid)]['sha256']:
            raise ValueError('Changed monomer query')
        if len({row[0] for row in m['rows']})!=len(m['rows']):raise ValueError('Duplicate accession keys')
        return m
    return get


def validate_tokens(t, bp):
    if t.dtype!=np.uint8 or t.ndim!=2 or len(t)<2 or not 0<bp<t.shape[1]:raise ValueError('Invalid tokens')
    if np.any(t>26) or not np.array_equal(t==26,np.broadcast_to(t[0]==26,t.shape)):
        raise ValueError('Invalid PAD/alphabet')
    if (t[0,:bp]==26).all() or (t[0,bp:]==26).all():raise ValueError('Empty valid chain')


def extend(ma, mb, original, oldmeta, p, oc):
    cfg={**oc,'maximum_paired_rows_including_query':config()['depth_sampling']['new_cap']}
    t,meta=pair(ma,mb,cfg)
    if t is None:raise ValueError('Originally eligible pair lost availability')
    validate_tokens(t,p['breakpoint']);n=len(original)
    if len(t)<n or len(t)>cfg['maximum_paired_rows_including_query'] or not np.array_equal(t[:n],original):
        raise ValueError('Original true prefix/query/PAD changed')
    for key in ('genome_keys','taxonomy'):
        if meta[key][:n-1]!=oldmeta[key]:raise ValueError('Original selected-key/taxonomy prefix changed')
    for key in ('length_a','length_b','quality_a','quality_b','common_keys','reused_homolog_rows',
                'taxonomy_conflicts','low_good_column_coverage_rows','reason','gate'):
        if meta[key]!=oldmeta[key]:raise ValueError('Non-depth pairing metadata changed: '+key)
    if meta['gate']!=p['gate']:raise ValueError('Original gate changed')
    if n<oc['maximum_paired_rows_including_query'] and len(t)!=n:raise ValueError('Unexpected growth of exhausted alignment')
    seed=int(text_hash(f"{config()['seed']}:{p['uid']}")[:16],16)
    s,null=shuffled(t,p['breakpoint'],meta['taxonomy'],seed)
    return t,s,meta,null


def quality_at_depth(old, oldmeta, newmeta):
    x=np.asarray(old,dtype=float).copy()
    if x.shape!=(68,) or not np.isfinite(x).all():raise ValueError('Invalid original quality vector')
    expected=np.array([np.log1p(oldmeta['common_keys']),np.log1p(oldmeta['neff']),
        (oldmeta['quality_a']+oldmeta['quality_b'])/2,abs(oldmeta['quality_a']-oldmeta['quality_b'])])
    np.testing.assert_array_equal(x[-4:],expected)
    # All monomer-profile and non-depth metadata components remain unchanged.
    x[-3]=np.log1p(newmeta['neff'])
    return x


def prepared(p, replay=False, mono=None):
    if not p['available']:raise ValueError('Never encode original fallback')
    path=ROOT/p['depth_input_file']
    if sha(path)!=p['depth_input_sha256']:raise ValueError('Prepared input checksum mismatch')
    with np.load(path,allow_pickle=False) as f:values={k:f[k].copy() for k in ('T256','S256','P256')}
    t,s,q=values['T256'],values['S256'],values['P256'];bp=p['breakpoint']
    for v in (t,s):validate_tokens(v,bp)
    if t.shape!=(p['depth_256'],p['length']) or s.shape!=t.shape or not np.array_equal(t[0],s[0]):
        raise ValueError('Depth/query/length differs')
    if token_hash(t[:p['paired_depth']])!=p['paired_tokens_sha256'] or token_hash(t[:1])!=p['q_tokens_sha256']:
        raise ValueError('Original prefix differs')
    for arm in ('T256','S256'):
        if token_hash(values[arm])!=p[arm+'_tokens_sha256']:raise ValueError('Changed '+arm+' tokens')
    if q.shape!=(68,) or not np.isfinite(q).all() or vector_hash(q)!=p['P256_sha256']:raise ValueError('Changed quality control')
    perm=np.asarray(p['null256']['permutation'])
    if sorted(perm.tolist())!=list(range(len(t))) or perm[0]!=0:raise ValueError('Invalid shuffle permutation')
    np.testing.assert_array_equal(s[:,:bp],t[:,:bp]);np.testing.assert_array_equal(s[:,bp:],t[perm,bp:])
    if replay:
        mono=mono or monomer_reader();ma,mb=mono(p['a']),mono(p['b'])
        oc=read(ROOT/'data/original_config.json');cfg={**oc,'maximum_paired_rows_including_query':config()['depth_sampling']['new_cap']}
        tt,meta=pair(ma,mb,cfg)
        np.testing.assert_array_equal(tt,t)
        if meta!=p['msa256']:raise ValueError('Original selector replay differs')
        ss,null=shuffled(tt,bp,meta['taxonomy'],int(text_hash(f"{config()['seed']}:{p['uid']}")[:16],16))
        np.testing.assert_array_equal(ss,s)
        if null!=p['null256']:raise ValueError('Original shuffler replay differs')
        # Membership checked separately from the constructor, against both source caches.
        da={k:(v,tax) for k,v,tax in ma['rows']};db={k:(v,tax) for k,v,tax in mb['rows']}
        for j,key in enumerate(meta['genome_keys'],1):
            va,ta=da[key];vb,tb=db[key]
            if (ta and tb and ta!=tb) or va.replace('-','')==vb.replace('-',''):raise ValueError('Invalid matched source membership')
            row=encode(va+vb);row[t[0]==26]=26;np.testing.assert_array_equal(row,t[j])
    return values


_AUDIT_MONO=None
def audit_one(p):
    global _AUDIT_MONO
    if _AUDIT_MONO is None:_AUDIT_MONO=monomer_reader()
    prepared(p,replay=True,mono=_AUDIT_MONO)
    return 1


def verify_input_population(pairs):
    with ProcessPoolExecutor(max_workers=config()['preparation_workers']) as pool:
        n=sum(pool.map(audit_one,(p for p in pairs if p['available']),chunksize=8))
    return {'eligible_inputs_verified':n,'exact_true_prefix_query_PAD':True,'matched_source_membership_verified':True,
            'original_selection_and_shuffle_replayed':True,'original_gate_unchanged':True,'R_evaluated':False}
