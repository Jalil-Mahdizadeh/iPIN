"""Compare both ILPs with independent exhaustive solutions on small instances."""
import itertools
import cvxpy as cp
import numpy as np
from common import ROOT,write_json
from sample_ilp import author,candidate_pool
from split_positives import solve_counts

def main():
    n=6;pos=np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[0,5]],dtype=np.int64)
    go=[frozenset(v) for v in [('g1',),('g1','g2'),('g2',),('g3',),('g3','g4'),('g4',)]]
    blocked=np.array(sorted({i*n+i for i in range(n)}|{int(a*n+b) for a,b in pos}|{0*n+2}))
    candidates=candidate_pool(n,pos,blocked,go,8)
    assert np.array_equal(candidates,candidate_pool(n,pos,blocked,go,8))
    assert len(candidates)==8 and not np.isin(candidates[:,0]*n+candidates[:,1],blocked).any()
    proteins=[str(i) for i in range(n)];index={p:i for i,p in enumerate(proteins)}
    ctx=author.build_context(pos,index,proteins,candidates,1.,go_bp=go)
    cfg=author.SamplingConfig(lambda_degree=1,lambda_jaccard=1,lambda_self_loop=0,lambda_taxon_pair=0,time_limit=20,mip_gap=0,threads=1,seed=2)
    biases=author.assemble_active_biases(cfg)
    for b in biases:b.precompute(ctx)
    problem,x,_=author.build_problem(ctx,[b for b in biases if b.is_active()])
    author.solve(problem,cp.HIGHS,author._solver_options(cp.HIGHS,cfg))
    actual=author.extract_negatives(x.value,ctx)
    def jaccard(pairs):
        return np.array([len(go[a]&go[b])/len(go[a]|go[b]) if go[a]|go[b] else 0 for a,b in pairs])
    plus=np.bincount(pos.ravel(),minlength=n);pool_degrees=np.bincount(candidates.ravel(),minlength=n)
    coef=1/np.log1p(plus);degree_scale=np.sum(coef*np.maximum(plus,pool_degrees-plus))
    target=float(jaccard(pos).mean());jscale=max(target,1-target)
    def objective(chosen):
        minus=np.bincount(chosen.ravel(),minlength=n)
        if np.any(minus>6*plus):return np.inf
        return float(np.sum(coef*np.abs(minus-plus))/degree_scale+abs(jaccard(chosen).mean()-target)/jscale)
    optimum=min(objective(candidates[list(chosen)]) for chosen in itertools.combinations(range(len(candidates)),len(pos)))
    assert abs(objective(actual)-optimum)<1e-7 and abs(problem.value-optimum)<1e-7
    # A four-group positive example permits independent enumeration of all 16 assignments.
    intra=np.array([4.,4.,1.,1.]);cross=np.zeros((4,4));cross[0,1]=2;cross[2,3]=1
    label,stats=solve_counts(intra,cross,time_limit=20,threads=1)
    total=intra.sum()+cross.sum();feasible=[]
    for assignment in itertools.product([0,1],repeat=4):
        retained=np.zeros(2)
        for i in range(4):retained[assignment[i]]+=intra[i]
        for i in range(4):
            for j in range(i+1,4):
                if assignment[i]==assignment[j]:retained[assignment[i]]+=cross[i,j]
        if retained[0]>=.76*retained.sum() and retained[1]>=.19*retained.sum():feasible.append(total-retained.sum())
    assert feasible and abs(stats['objective_discarded_pairs']-min(feasible))<1e-7
    result=dict(passed=True,candidate_exclusions_and_seed=True,negative_objective_exhaustive_optimum=float(optimum),
                positive_split_exhaustive_optimum=float(min(feasible)),original_test_full_array_validation='reports/source-and-test-audit.json')
    write_json(ROOT/'reports/qualification.json',result);print(result)

if __name__=='__main__':main()
