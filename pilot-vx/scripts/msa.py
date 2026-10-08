"""Label-independent omicMSA parsing, pairing, masks and null controls."""
import gzip
import hashlib
import heapq
import json
import string
from collections import defaultdict
import numpy as np
from common import config, text_hash

ALPHABET = 'ARNDCEQGHILKMFPSTWYVXBZUO-'
LOOKUP = {c: i for i,c in enumerate(ALPHABET)}
PAD = 26
DELETE_INSERTIONS = str.maketrans('', '', string.ascii_lowercase + '.')
DELETE_VALID = str.maketrans('', '', ALPHABET)
TOKEN_LOOKUP = np.full(256, 255, dtype=np.uint8)
for _letter, _token in LOOKUP.items():
    TOKEN_LOOKUP[ord(_letter)] = _token


def records(lines):
    header = None; seq = []
    for raw in lines:
        line = raw.strip()
        if not line: continue
        if line.startswith('>'):
            if header is not None: yield header, ''.join(seq)
            header = line[1:]; seq = []
        else:
            if header is None: raise ValueError('Sequence before header')
            seq.append(line)
    if header is not None: yield header, ''.join(seq)


def aligned(sequence):
    # Lowercase and dots are insertion columns, never query coordinates.
    return sequence.translate(DELETE_INSERTIONS)


def cached_monomer(record_iter, query, quality, forbidden_hashes, maximum=4096):
    if len(query) != len(quality) or set(quality)-{'*','-'}: raise ValueError('Invalid quality-mask coordinates')
    if any(c not in LOOKUP or c=='-' for c in query): raise ValueError('Unsupported query alphabet')
    heap=[]; seen={}; conflicts=set(); stats=defaultdict(int)
    min_coverage=config()['minimum_homolog_query_coverage']
    for header, seq in record_iter:
        key=header.split()[0]
        if key in ['query','mask']: raise ValueError('Repeated query or quality mask')
        stats['raw_rows']+=1; a=aligned(seq)
        # The real archive contains ambiguous J residues; the encoder has no J token.
        # Preserve columns and explicitly represent this ambiguity with its X token.
        if 'J' in a:
            stats['ambiguous_J_residues_to_X']+=a.count('J');a=a.replace('J','X')
        if len(a)!=len(query) or a.translate(DELETE_VALID):
            raise ValueError(f'Invalid alignment coordinates/alphabet for {key}: query_length={len(query)}, row_length={len(a)}, invalid={a.translate(DELETE_VALID)!r}, query_sha256={text_hash(query)}')
        digest=text_hash(a)
        if key in seen:
            stats['duplicate_keys']+=1
            if seen[key]!=digest: conflicts.add(key)
            continue
        seen[key]=digest
        if text_hash(a.replace('-','')) in forbidden_hashes:
            stats['excluded_human_context']+=1;continue
        if (len(a)-a.count('-'))/len(query)<min_coverage:
            stats['low_coverage_rows']+=1;continue
        stats['usable_before_cap']+=1
        # All monomers share the same accession priority, retaining common pairing keys.
        priority=int(hashlib.sha256(key.encode()).hexdigest(),16)
        tax=header.split(maxsplit=1)[1].strip() if len(header.split(maxsplit=1))==2 else ''
        row=(key,a,tax)
        item=(-priority,key,row)
        if len(heap)<maximum:heapq.heappush(heap,item)
        elif item>heap[0]:heapq.heapreplace(heap,item)
    rows=[item[2] for item in sorted(heap,reverse=True) if item[1] not in conflicts]
    stats['conflicting_keys']=len(conflicts);stats['retained_rows']=len(rows)
    return {'query':query,'mask':quality,'rows':rows,'stats':dict(stats),'query_sha256':text_hash(query)}


def load_monomer(path):
    with gzip.open(path,'rt') as f:return json.load(f)


def tax_group(tax):
    # The archive supplies genus:family:order:class:phylum, not species identifiers.
    fields=tax.split(':')
    return fields[1].strip() if len(fields)>1 and fields[1].strip().lower() not in ['', 'na', 'n/a', 'none', 'unknown', 'unclassified'] else '__unknown__'


def encode(seq):
    tokens=TOKEN_LOOKUP[np.frombuffer(seq.encode('ascii'),dtype=np.uint8)]
    if (tokens==255).any():raise ValueError('Unsupported alignment alphabet')
    return tokens


def similarity(a,b,quality,identity=.9,coverage=.8):
    valid=(a!=LOOKUP['-'])&(b!=LOOKUP['-'])&quality
    if valid.sum()<coverage*int(quality.sum()):return False
    return float((a[valid]==b[valid]).mean())>=identity


def pair(a,b,cfg=None):
    cfg=cfg or config(); la,lb=len(a['query']),len(b['query'])
    qa=np.array([c=='*' for c in a['mask']]);qb=np.array([c=='*' for c in b['mask']]);quality=np.r_[qa,qb]
    meta={'length_a':la,'length_b':lb,'quality_a':float(qa.mean()),'quality_b':float(qb.mean()),
          'common_keys':0,'reused_homolog_rows':0,'taxonomy_conflicts':0,'low_good_column_coverage_rows':0,'retained_homologs':0}
    if la+lb>cfg['maximum_combined_residues']:return None,{**meta,'reason':'long_query'}
    if min(qa.mean(),qb.mean())<cfg['minimum_quality_fraction_per_chain']:return None,{**meta,'reason':'low_quality_coverage'}
    da={r[0]:r for r in a['rows']};db={r[0]:r for r in b['rows']};groups=defaultdict(list)
    common=sorted(da.keys()&db.keys(),key=text_hash);meta['common_keys']=len(common)
    for key in common:
        _,sa,ta=da[key];_,sb,tb=db[key]
        if ta and tb and ta!=tb:meta['taxonomy_conflicts']+=1;continue
        if sa.replace('-','')==sb.replace('-',''):meta['reused_homolog_rows']+=1;continue
        ea,eb=encode(sa),encode(sb)
        if min(float((ea[qa]!=LOOKUP['-']).mean()),float((eb[qb]!=LOOKUP['-']).mean()))<cfg['minimum_homolog_query_coverage']:
            meta['low_good_column_coverage_rows']+=1;continue
        groups[tax_group(ta or tb)].append((key,np.r_[ea,eb],ta or tb))
    # Round-robin families to limit multiple assemblies/clades dominating the depth.
    candidates=[]
    group_keys=sorted(groups,key=text_hash)
    for i in range(max(map(len,groups.values()),default=0)):
        for group in group_keys:
            if i<len(groups[group]):candidates.append(groups[group][i])
    selected=[];query=encode(a['query']+b['query'])
    for key,row,tax in candidates:
        if any(similarity(row,other,quality,cfg['redundancy_identity'],cfg['minimum_identity_comparison_coverage']) for _,other,_ in selected):continue
        selected.append((key,row,tax))
        if len(selected)>=cfg['maximum_paired_rows_including_query']-1:break
    meta['retained_homologs']=len(selected);meta['taxonomic_families']=len({tax_group(t) for _,_,t in selected})
    if len(selected)<cfg['minimum_paired_homologs']:return None,{**meta,'reason':'shallow_paired_alignment'}
    tokens=np.stack([query]+[r for _,r,_ in selected]);tokens[:,~quality]=PAD
    # Neff on the retained homologs, excluding the query; same identity/coverage convention.
    hs=[r for _,r,_ in selected];counts=np.ones(len(hs))
    for i in range(len(hs)):
        for j in range(i):
            if similarity(hs[i],hs[j],quality,cfg['redundancy_identity'],cfg['minimum_identity_comparison_coverage']):counts[i]+=1;counts[j]+=1
    meta['neff']=float((1/counts).sum());meta['reason']='available'
    meta['gate']=float(min(1,meta['neff']/32)*min(qa.mean(),qb.mean()))
    meta['genome_keys']=[k for k,_,_ in selected];meta['taxonomy']=[t for _,_,t in selected]
    return tokens,meta


def shuffled(tokens,breakpoint,taxonomy,seed):
    rng=np.random.default_rng(seed);permutation=np.arange(len(tokens));remaining=set(range(1,len(tokens)))
    levels={}
    for level,name in [(1,'family'),(2,'order'),(3,'class')]:
        groups=defaultdict(list)
        for i in sorted(remaining):
            fields=taxonomy[i-1].split(':')
            if len(fields)>level and fields[level].strip().lower() not in ['', 'na', 'n/a', 'none', 'unknown', 'unclassified']:
                groups[':'.join(fields[level:])].append(i)
        levels[name]=0
        for g in sorted(groups):
            indices=np.array(groups[g])
            if len(indices)>1:
                order=rng.permutation(indices)
                permutation[order]=np.roll(order,1)
                remaining.difference_update(indices.tolist());levels[name]+=len(indices)
    levels['unchanged']=len(remaining)
    out=tokens.copy();out[1:,breakpoint:]=tokens[permutation[1:],breakpoint:]
    return out,{'changed_fraction':float((permutation[1:]!=np.arange(1,len(tokens))).mean()),'permutation':permutation.tolist(),
               'level_counts':levels,'kind':'finest-available-family-order-class; residual singletons unchanged'}


def profile(m):
    q=m['query'];good=np.array([c=='*' for c in m['mask']]);rows=m['rows'];depth=len(rows)
    counts=np.zeros((len(q),26),float)
    for _,s,_ in rows:
        t=encode(s);counts[np.arange(len(q)),t]+=1
    probs=counts/max(1,depth); nongap=probs[:,:25];denom=nongap.sum(1,keepdims=True)
    freq=np.divide(nongap,denom,out=np.zeros_like(nongap),where=denom>0)
    entropy=-(freq*np.log(np.maximum(freq,1e-12))).sum(1)
    return np.r_[np.log1p(len(q)),np.log1p(depth),good.mean(),
                 probs[:,25].mean(),np.mean(entropy[good]) if good.any() else 0,
                 np.quantile(entropy[good],[.25,.75]) if good.any() else [0,0],freq.mean(0)]


def quality_features(a,b,meta):
    pa,pb=profile(a),profile(b)
    return np.r_[(pa+pb)/2,np.abs(pa-pb),np.log1p(meta.get('common_keys',0)),
                 np.log1p(meta.get('neff',0)),(meta.get('quality_a',0)+meta.get('quality_b',0))/2,
                 abs(meta.get('quality_a',0)-meta.get('quality_b',0))]
