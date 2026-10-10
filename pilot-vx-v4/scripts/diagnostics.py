"""Label-free depth, diversity, coverage and dependence diagnostics for I."""
from collections import Counter
import hashlib
import numpy as np
from msa import tax_group


def neff(rows, good, identity=.8, coverage=.8):
    h=rows[:,good]; n=len(h); counts=np.ones(n,dtype=int)
    for j in range(1,n):
        valid=(h[:j]!=25)&(h[j]!=25); covered=valid.sum(1)
        equal=((h[:j]==h[j])&valid).sum(1)
        near=(covered>=coverage*len(h[0]))&(equal>=identity*covered)
        counts[j]+=int(near.sum()); counts[:j]+=near
    return float((1/counts).sum())


def sequence_stats(t):
    good=t[0]!=26; h=t[1:,good]; occupied=(h!=25).mean(0)
    match=(h==t[0,good]).mean(1)
    return {'neff80':neff(t[1:],good),'mean_occupancy':float(occupied.mean()),
            'positions_at_half_depth':float((occupied>=.5).mean()),'query_identity_mean':float(match.mean()),
            'query_identity_std':float(match.std()),'gap_fraction_mean':float((h==25).mean()),
            'row_gap_fraction_std':float((h==25).mean(1).std())}


def covariance(t,pairs):
    h=t[1:]; values=[]
    for a,b in pairs:
        joint=np.bincount(h[:,a].astype(int)*26+h[:,b],minlength=676).reshape(26,26)/len(h)
        c=joint-joint.sum(1)[:,None]*joint.sum(0)[None,:];values.append(float((c*c).sum()))
    return float(np.mean(values)) if values else None


def describe(t,i,p,previous,selected,pools,monomers,limit=128):
    bp=p['breakpoint']; good=t[0]!=26
    permutation=np.array(previous['null']['permutation']); s=t.copy();s[1:,bp:]=t[permutation[1:],bp:]
    regions={}; selection={}
    for name,begin,end,pool,other,m in [('A',0,bp,pools[0],pools[1],monomers[0]),('B',bp,t.shape[1],pools[1],pools[0],monomers[1])]:
        keys=selected[name]; old=set(previous['msa']['genome_keys']); counts=Counter(tax_group(pool[k][1]) for k in keys)
        probabilities=np.array(list(counts.values()))/len(keys)
        selection[name]={'cached_depth':len(m['rows']),'filtered_depth':len(pool),'selected_depth':len(keys),
            'fraction_selected_outside_other_pool':sum(k not in other for k in keys)/len(keys),
            'fraction_selected_outside_original_rows':sum(k not in old for k in keys)/len(keys),
            'taxonomic_families':len(counts),'effective_taxonomic_families':float(np.exp(-(probabilities*np.log(probabilities)).sum())),
            'original_taxonomic_families':len({tax_group(v) for v in previous['msa']['taxonomy']})}
        regions[name]={arm:sequence_stats(a[:,begin:end]) for arm,a in [('true',t),('I',i)]}
        mask=good[begin:end]; tt=t[1:,begin:end][:,mask];ii=i[1:,begin:end][:,mask]
        regions[name]['mean_column_profile_TV']=float(sum(np.abs((tt==v).mean(0)-(ii==v).mean(0)) for v in range(26)).mean()/2)
    selection['same_key_row_fraction']=sum(a==b for a,b in zip(selected['A'],selected['B']))/(len(t)-1)
    ma={k:v.replace('-','') for k,v,_ in monomers[0]['rows']}; mb={k:v.replace('-','') for k,v,_ in monomers[1]['rows']}
    selection['identical_sequence_row_fraction']=sum(ma[a]==mb[b] for a,b in zip(selected['A'],selected['B']))/(len(t)-1)
    variable=good&((np.max(np.stack([(t[1:]==v).sum(0) for v in range(27)]),axis=0)<len(t)-1)|
                   (np.max(np.stack([(i[1:]==v).sum(0) for v in range(27)]),axis=0)<len(i)-1))
    rng=np.random.Generator(np.random.PCG64(int(hashlib.sha256(('vx-v4:diagnostics:'+p['uid']).encode()).hexdigest()[:16],16)))
    aa=np.flatnonzero(variable&(np.arange(t.shape[1])<bp));bb=np.flatnonzero(variable&(np.arange(t.shape[1])>=bp))
    cov={}
    for kind,left,right,same in [('within_A',aa,aa,True),('within_B',bb,bb,True),('cross_chain',aa,bb,False)]:
        possible=len(left)*(len(left)-1)//2 if same else len(left)*len(right); chosen=set()
        while len(chosen)<min(limit,possible):
            a,b=int(rng.choice(left)),int(rng.choice(right))
            if same:
                if a==b:continue
                a,b=sorted([a,b])
            chosen.add((a,b))
        pp=sorted(chosen);cov[kind]={arm:covariance(x,pp) for arm,x in [('true',t),('shuffled',s),('I',i)]}
    return {'depth_including_query':len(t),'exact_query_PAD_depth':True,'intact_source_rows':True,'labels_used':False,
            'selection':selection,'chain_diagnostics':regions,'categorical_covariance_energy':cov}
