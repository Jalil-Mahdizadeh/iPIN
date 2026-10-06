"""Bounded independent objective, warm-start, interruption and checkpoint checks."""
import itertools
import sys
import numpy as np
from solver import ROOT,Problem,Checkpoint,optimize,SCALE,write_json,author


def main():
    n=6
    pos=np.array([[0,1],[1,2],[2,3],[3,4],[4,5],[0,5]],dtype=np.int64)
    pos=np.sort(pos,axis=1)
    keys={tuple(p) for p in pos}|{(0,2)}
    cand=np.array([p for p in itertools.combinations(range(n),2) if p not in keys])
    go=[frozenset(v) for v in [('g1',),('g1','g2'),('g2',),('g3',),('g3','g4'),('g4',)]]
    p=Problem(pos,cand,[str(i) for i in range(n)],go)
    plus=np.bincount(pos.ravel(),minlength=n)
    capacity=np.bincount(cand.ravel(),minlength=n)
    weights=1/np.log1p(plus)
    normalizer=float(np.sum(weights*np.maximum(plus,capacity-plus)))
    def jaccard(a,b):
        union=go[a]|go[b]
        return len(go[a]&go[b])/len(union) if union else 0.
    target=np.mean([jaccard(a,b) for a,b in pos]);best=float('inf')
    model,matrix,rhs,costs,upper=p.model()
    for selected in itertools.combinations(range(len(cand)),len(pos)):
        selected=np.array(selected);minus=np.bincount(cand[selected].ravel(),minlength=n)
        expected=float(weights@np.abs(minus-plus)/normalizer+
                       abs(np.mean([jaccard(a,b) for a,b in cand[selected]])-target)/max(target,1-target))
        vector=p.vector(selected)
        assert np.max(np.abs(matrix@vector-rhs))<1e-9
        assert (vector<=upper+1e-9).all()
        assert abs(costs@vector/SCALE-expected)<1e-10
        assert abs(p.metrics(selected)['objective']-expected)<1e-10
        best=min(best,expected)
    directory=ROOT/'work/recovery-v1/qualification'
    checkpoint=Checkpoint(p,directory/'optimal','tiny-model-v1')
    result=optimize(p,checkpoint,20,directory/'optimal-highs.log',threads=1,gap=0)
    assert abs(checkpoint.objective-best)<1e-8
    assert result['solver_observed_feasible_incumbent'] and result['returned_solution_feasible']
    interrupted=Checkpoint(p,directory/'interrupted','tiny-model-v1')
    result2=optimize(p,interrupted,20,directory/'interrupted-highs.log',threads=1,gap=0,interrupt_after=0.)
    reloaded=Checkpoint(p,directory/'interrupted','tiny-model-v1')
    p.check(reloaded.selected)
    assert np.array_equal(reloaded.selected,interrupted.selected)
    assert result2['solver_observed_feasible_incumbent']
    assert result2['status']=='kInterrupt' and result2['stop_reason']=='qualification_interrupt'
    rejected=False
    try:checkpoint.update(np.array([],dtype=np.int64),'invalid-zero-output')
    except AssertionError:rejected=True
    assert rejected
    result=dict(passed=True,all_enumerated_objectives_agree=True,exhaustive_optimum=best,
                warm_start_accepted=True,forced_interruption_preserves_checked_selection=True,
                invalid_empty_selection_rejected=True,checkpoint_reload_verified=True,
                interrupted_model_status=result2['status'])
    write_json(ROOT/'reports/recovery-v1-qualification.json',result)
    print(result,flush=True)

if __name__=='__main__':main()
