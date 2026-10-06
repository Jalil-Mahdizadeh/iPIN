"""Two-way adaptation of the author's retained-interaction ILP formulation."""
import csv
import resource
import time
from collections import Counter
import cvxpy as cp
import numpy as np
from common import ROOT,read_json,write_json,write_csv,fasta_read,fasta_write,mark,sha

def solve_counts(intra,cross,fractions=(.8,.2),epsilon=.05,time_limit=7200,threads=8):
    n=len(intra);pairs=np.transpose(np.nonzero(np.triu(cross,1)));counts=cross[pairs[:,0],pairs[:,1]] if len(pairs) else np.empty(0)
    x=cp.Variable((2,n),boolean=True);constraints=[cp.sum(x,axis=0)==1]
    kept=[intra@x[s,:] for s in range(2)]
    if len(pairs):
        z=cp.Variable((2,len(pairs)),nonneg=True)
        constraints.extend([z<=x[:,pairs[:,0]],z<=x[:,pairs[:,1]],z>=x[:,pairs[:,0]]+x[:,pairs[:,1]]-1])
        kept=[kept[s]+counts@z[s,:] for s in range(2)]
    total=cp.sum(cp.hstack(kept));n_pos=float(intra.sum()+counts.sum())
    constraints.extend([kept[s]>=(1-epsilon)*fractions[s]*total for s in range(2)])
    problem=cp.Problem(cp.Minimize(n_pos-total),constraints)
    started=time.monotonic()
    problem.solve(solver=cp.HIGHS,verbose=True,time_limit=time_limit,threads=threads,random_seed=2,mip_rel_gap=.01)
    if problem.status not in cp.settings.SOLUTION_PRESENT or x.value is None:
        raise RuntimeError(f'No feasible positive split: {problem.status}')
    values=np.asarray(x.value);assert np.max(np.abs(values-np.round(values)))<1e-5
    assert np.all(np.round(values).sum(axis=0)==1)
    info=problem.solver_stats.extra_stats
    stats=dict(status=problem.status,objective_discarded_pairs=float(problem.value),seconds=time.monotonic()-started,
               configured_time_limit=time_limit,configured_mip_gap=.01,solver='HIGHS',actual_mip_gap=float(info.mip_gap),
               dual_bound=float(info.mip_dual_bound),nodes=int(info.mip_node_count),mathematically_proven_optimal=bool(abs(info.mip_gap)<1e-9))
    return np.argmax(values,axis=0),stats

def main():
    mark('positive_split','running')
    groups=read_json(ROOT/'work/protein-groups.json');unique=sorted(set(groups.values()));indices={g:i for i,g in enumerate(unique)}
    rows=list(csv.DictReader((ROOT/'work/eligible-positives.csv').open()))
    n=len(unique);intra=np.zeros(n);cross=np.zeros((n,n))
    for r in rows:
        i,j=sorted([indices[groups[r['protein1']]],indices[groups[r['protein2']]]])
        if i==j:intra[i]+=1
        else:cross[i,j]+=1
    labels,stats=solve_counts(intra,cross)
    assignment={p:('train' if labels[indices[g]]==0 else 'val') for p,g in groups.items()}
    selected={'train':[],'val':[]};dropped=[]
    for r in rows:
        s,t=assignment[r['protein1']],assignment[r['protein2']]
        if s==t:selected[s].append(r)
        else:dropped.append(r)
    retained=sum(map(len,selected.values()));assert retained+len(dropped)==len(rows)
    assert abs(len(dropped)-stats['objective_discarded_pairs'])<1e-4
    assert len(selected['train'])/retained>=.76-1e-8 and len(selected['val'])/retained>=.19-1e-8
    fields=['protein1','protein2','score','evidence_rows','first_source_row_id'];seqs=fasta_read(ROOT/'work/eligible.fasta')
    counts={}
    for s,rr in selected.items():
        write_csv(ROOT/'work'/f'{s}-positives.csv',fields,rr)
        proteins={p for r in rr for p in (r['protein1'],r['protein2'])}
        fasta_write(ROOT/'work'/f'{s}.fasta',{p:seqs[p] for p in proteins})
        counts[s]=dict(positives=len(rr),protein_sequences=len(proteins),retained_positive_fraction=len(rr)/retained)
    write_csv(ROOT/'reports/positive-pairs-crossing-train-val.csv',fields,dropped)
    write_json(ROOT/'work/protein-to-development-split.json',assignment)
    result=dict(**stats,eligible_positives=len(rows),retained_positives=retained,discarded_crossing_positives=len(dropped),splits=counts,
                peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
                formulation='author retained-PPI objective and relative lower fraction constraints, adapted to two splits; continuous z is exact conditional on binary x',
                input_sha256={p:sha(ROOT/'work'/p) for p in ['protein-groups.json','eligible-positives.csv']})
    write_json(ROOT/'reports/positive-split.json',result);mark('positive_split','complete',summary=result)

if __name__=='__main__':
    try:main()
    except Exception as exc:mark('positive_split','failed',error=str(exc));raise
