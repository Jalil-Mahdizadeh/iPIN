"""Apply the declared test boundary; build must-link homology/entry groups."""
import csv
import sys
from collections import defaultdict,Counter
import numpy as np
import kahip
from common import ROOT,read_json,write_json,write_csv,fasta_read,fasta_write,pair,mark,sha

def hits(path):
    with path.open() as f:
        for row in csv.reader(f,delimiter='\t'):
            assert len(row)==9
            yield row[0],row[1],float(row[2]),float(row[3]),float(row[4]),int(row[5]),int(row[6]),int(row[7]),float(row[8])

def qualifies(h):return h[2]>=.4 and h[3]>=.8 and h[4]>=.8

class DSU:
    def __init__(self,ids):self.parent={p:p for p in ids}
    def find(self,p):
        while self.parent[p]!=p:
            self.parent[p]=self.parent[self.parent[p]];p=self.parent[p]
        return p
    def join(self,a,b):
        a,b=self.find(a),self.find(b)
        if a!=b:self.parent[max(a,b)]=min(a,b)

def filter_pool():
    mark('homology_filter','running')
    bad=defaultdict(list)
    for name,reverse in [('eligible-vs-test.tsv',False),('test-vs-eligible.tsv',True)]:
        for h in hits(ROOT/'work'/name):
            if qualifies(h):
                q,t=h[:2];p=q if not reverse else t
                bad[p].append(dict(hit_test=t if not reverse else q,identity=h[2],qcov=h[3],tcov=h[4],direction=name))
    keep=[];removed=[]
    for r in csv.DictReader((ROOT/'work/eligible-before-homology.csv').open()):
        if r['protein1'] in bad or r['protein2'] in bad:removed.append({**r,'reason':'qualifying_test_homology'})
        else:keep.append(r)
    fields=['protein1','protein2','score','evidence_rows','first_source_row_id']
    write_csv(ROOT/'work/eligible-positives.csv',fields,keep)
    write_csv(ROOT/'reports/positive-pairs-excluded-for-homology.csv',fields+['reason'],removed)
    write_json(ROOT/'reports/protein-test-homology-exclusions.json',dict(bad))
    seqs=fasta_read(ROOT/'work/eligible-before-homology.fasta')
    active={p for r in keep for p in (r['protein1'],r['protein2'])}
    assert active and not active & set(bad)
    fasta_write(ROOT/'work/eligible.fasta',{p:seqs[p] for p in active})
    result=dict(excluded_homologous_sequence_entities=len(bad),excluded_positive_pairs=len(removed),retained_positive_pairs=len(keep),retained_sequence_entities=len(active),directions=2)
    write_json(ROOT/'reports/test-homology-filter.json',result)
    mark('homology_filter','complete',**result)

def group():
    mark('groups','running')
    proteins=fasta_read(ROOT/'work/eligible.fasta');registry=read_json(ROOT/'work/protein-registry.json')
    ids=sorted(proteins);dsu=DSU(ids);families=defaultdict(list)
    for p in ids:
        for family in registry[p]['families']:families[family].append(p)
    for members in families.values():
        for p in members[1:]:dsu.join(members[0],p)
    n_hard=0;n_hits=0
    for h in hits(ROOT/'work/development-all-hits.tsv'):
        n_hits+=1
        if h[0]!=h[1] and qualifies(h):dsu.join(h[0],h[1]);n_hard+=1
    components=defaultdict(list)
    for p in ids:components[dsu.find(p)].append(p)
    groups=[sorted(v) for _,v in sorted(components.items())]
    component_of={p:i for i,ps in enumerate(groups) for p in ps}
    soft_edges={}
    for h in hits(ROOT/'work/development-all-hits.tsv'):
        i,j=component_of[h[0]],component_of[h[1]]
        if i==j:continue
        key=pair(i,j);weight=max(1,int(round(100*h[8]/min(h[6],h[7]))))
        soft_edges[key]=max(soft_edges.get(key,0),weight)
    # KaHIP groups the contracted similarity graph. No must-link component can be cut.
    graph=kahip.kahip_graph();graph.set_num_nodes(len(groups))
    for i,ps in enumerate(groups):graph.set_weight(i,len(ps))
    for (i,j),weight in sorted(soft_edges.items()):graph.add_undirected_edge(i,j,weight)
    largest=max(map(len,groups));k=min(100,len(groups),max(2,len(ids)//largest))
    assert len(groups)>=2
    v,x,e,a=graph.get_csr_arrays()
    if len(groups)<=k:labels=list(range(len(groups)));cut=0
    else:cut,labels=kahip.kaffpa(v,x,e,a,k,.03,True,2,kahip.STRONG)
    assert len(labels)==len(groups)
    assignment={p:int(labels[component_of[p]]) for p in ids}
    for members in families.values():assert len({assignment[p] for p in members})==1
    for h in hits(ROOT/'work/development-all-hits.tsv'):
        if qualifies(h):assert assignment[h[0]]==assignment[h[1]],h
    write_json(ROOT/'work/protein-groups.json',assignment)
    write_csv(ROOT/'work/homology-components.tsv',['protein_id','component','group'],
              (dict(protein_id=p,component=component_of[p],group=assignment[p]) for p in ids),delimiter='\t')
    sizes=Counter(assignment.values())
    result=dict(active_proteins=len(ids),all_alignment_rows=n_hits,qualifying_directed_homology_edges=n_hard,
                must_link_components=len(groups),largest_must_link_component=largest,requested_max_groups=100,actual_groups=len(sizes),
                kahip_version=kahip.__version__,kahip_seed=2,kahip_mode='STRONG',cut=int(cut),soft_component_edges=len(soft_edges),
                group_sizes=dict(sorted(sizes.items())),method='MMseqs2 normalized-bit-score graph; qualifying homology and UniProt entry families contracted before KaHIP',
                difference_from_paper='MMseqs2 replaces BLAST; hard component contraction prevents residual qualifying sequence/isoform leakage before the two-way ILP')
    write_json(ROOT/'reports/grouping.json',result);mark('groups','complete',**result)

if __name__=='__main__':
    try:
        if sys.argv[1]=='filter':filter_pool()
        elif sys.argv[1]=='group':group()
        else:raise ValueError(sys.argv[1])
    except Exception as exc:mark('groups' if sys.argv[1]=='group' else 'homology_filter','failed',error=str(exc));raise
