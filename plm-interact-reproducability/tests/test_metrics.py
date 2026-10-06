"""Small independent checks for metric conventions and the mutation formula."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import numpy as np
from scipy.special import expit,log_expit
from metrics_eval import metrics

def test_perfect_and_tied():
    p=metrics([1,0,1,0],[0.9,0.1,0.8,0.2])
    assert p['average_precision']==p['auroc']==p['f1']==1
    t=metrics([1,0,1,0],[0.5]*4)
    assert t['average_precision']==t['auroc']==0.5
    assert t['tp']==2 and t['fp']==2

def test_pr_integration_definitions_differ():
    m=metrics([0,1,0,1],[0.9,0.8,0.7,0.6])
    assert abs(m['average_precision']-0.5)<1e-12
    assert abs(m['pr_auc_trapezoid']-1/3)<1e-12

def test_mutation_formula_is_not_log_odds_ratio():
    # Same classification direction, different rankings: scientifically material.
    wild=np.array([10.,0.]);mutant=np.array([12.,1.])
    code=mutant-wild
    paper=log_expit(mutant)-log_expit(wild)
    assert code[0]>code[1] and paper[0]<paper[1]
    assert np.array_equal(code>0,paper>0)

if __name__=='__main__':
    test_perfect_and_tied();test_pr_integration_definitions_differ();test_mutation_formula_is_not_log_odds_ratio()
    print('3 independent metric/formula checks passed')
