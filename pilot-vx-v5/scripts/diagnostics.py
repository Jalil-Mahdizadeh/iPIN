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


def describe(t, s, p, oldmeta, meta, null):
    old=t[:p['paired_depth']];bp=p['breakpoint'];good=t[0]!=26
    d={'depth_128':len(old),'depth_256':len(t),'added_homologs':len(t)-len(old),
       'exact_query_PAD_prefix':True,'intact_source_rows':True,'labels_used':False,
       'joint_neff80_128':neff(old[1:],good),'joint_neff80_256':neff(t[1:],good),
       'joint_neff90_128':oldmeta['neff'],'joint_neff90_256':meta['neff'],
       'taxonomic_families_128':oldmeta['taxonomic_families'],'taxonomic_families_256':meta['taxonomic_families'],
       'shuffle_row_fraction_256':null['changed_fraction'],
       'shuffle_token_fraction_256':float((t[1:,bp:][:,good[bp:]]!=s[1:,bp:][:,good[bp:]]).mean()),
       'gate':p['gate']}
    for chain,a,b in [('A',0,bp),('B',bp,t.shape[1])]:
        d[chain]={'depth128':sequence_stats(old[:,a:b]),'depth256':sequence_stats(t[:,a:b])}
    return d

