"""Recover one frozen TRAIN/DEV sampling problem, without regenerating candidates."""
import csv
import hashlib
import json
import os
import resource
import sys
import time
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from solver import ROOT,Problem,Checkpoint,optimize,read_json,write_json,sha
from common import write_csv,mark,pair


def load_problem(split):
    directory=ROOT/f'work/negative-{split}'
    receipt=read_json(directory/'candidate-manifest.json');spec=receipt['specification']
    assert sha(directory/'candidates.npz')==receipt['sha256']
    for path,key in [(ROOT/f'work/{split}-positives.csv','positive_sha256'),
                     (ROOT/'work/protein-registry.json','metadata_sha256'),
                     (ROOT/'work/known-positive-families.tsv','blacklist_sha256'),
                     (ROOT/'sources/author-pipeline/bin/sample_negatives_ilp.py','author_code_sha256'),
                     (ROOT/'scripts/sample_ilp.py','wrapper_code_sha256')]:
        assert sha(path)==spec[key],path
    rows=list(csv.DictReader((ROOT/f'work/{split}-positives.csv').open()))
    proteins=read_json(directory/'protein-order.json')
    assert proteins==spec['proteins']==sorted({p for r in rows for p in [r['protein1'],r['protein2']]})
    index={p:i for i,p in enumerate(proteins)}
    positives=np.array([sorted([index[r['protein1']],index[r['protein2']]]) for r in rows],dtype=np.int64)
    assert np.array_equal(positives,np.load(directory/'positive-pairs.npy'))
    candidates=np.load(directory/'candidates.npz')['pairs']
    registry=read_json(ROOT/'work/protein-registry.json');families=defaultdict(set)
    for p,i in index.items():
        assert registry[p]['taxon']==9606
        for f in registry[p]['families']:families[f].add(i)
    n=len(proteins)
    forbidden={i*n+i for i in range(n)}|set((positives[:,0]*n+positives[:,1]).tolist())
    for r in csv.DictReader((ROOT/'work/known-positive-families.tsv').open(),delimiter='\t'):
        for a in families.get(r['family1'],()):
            for b in families.get(r['family2'],()):
                i,j=sorted((a,b));forbidden.add(i*n+j)
    assert not np.isin(candidates[:,0]*n+candidates[:,1],np.array(sorted(forbidden))).any()
    assert len(candidates)==spec['candidate_target']
    go=[frozenset(registry[p]['go_bp']) for p in proteins]
    problem=Problem(positives,candidates,proteins,go)
    return problem,rows,receipt,n*(n+1)//2-len(forbidden)


def main(split):
    assert split in ['train','val']
    mark(f'negative_{split}','running',recovery='v1')
    contract=read_json(ROOT/'provenance/recovery-v1/contract.json')
    problem,rows,receipt,available=load_problem(split)
    directory=ROOT/f'work/recovery-v1/{split}';directory.mkdir(parents=True,exist_ok=True)
    identity=dict(recovery_contract=contract['fingerprint'],candidate_fingerprint=receipt['fingerprint'])
    checkpoint=Checkpoint(problem,directory,identity)
    if checkpoint.selected is None:checkpoint.update(problem.feasible_start(),'verified_feasible_initialization',force=True)
    initial_path=directory/'initial.json'
    if not initial_path.exists():write_json(initial_path,problem.metrics(checkpoint.selected))
    initial=read_json(initial_path)
    # Account for previous completed or interrupted attempts within the same recovery budget.
    attempts=directory/'attempts';attempts.mkdir(exist_ok=True)
    consumed=0.
    for path in sorted(attempts.glob('*.json')):
        previous=read_json(path)
        if 'consumed_seconds' in previous:consumed+=previous['consumed_seconds']
        else:
            # Unknown interruption time is charged conservatively to prevent unbounded retries.
            consumed+=previous['allocated_seconds']
    remaining=max(0.,contract['policy']['solver_seconds_per_split']-consumed)
    assert remaining>0,'Recovery time budget exhausted; checked incumbent remains saved'
    attempt=attempts/f'{len(list(attempts.glob("*.json")))+1:03d}.json'
    started=time.time()
    state=dict(pid=os.getpid(),started_unix=started,allocated_seconds=remaining)
    write_json(attempt,state)
    result=optimize(problem,checkpoint,remaining,directory/f'highs-{attempt.stem}.log',
                    threads=8,gap=.01,stall_seconds=contract['policy']['no_improvement_seconds'])
    write_json(attempt,dict(**state,finished_unix=time.time(),consumed_seconds=result['solver_seconds']))
    assert result['solver_observed_feasible_incumbent'],'HiGHS did not confirm the initial solution'
    improved=checkpoint.objective < initial['objective']-1e-10
    assert improved or result['actual_mip_gap']<=.01,'No optimization progress; retain diagnostic checkpoint without publishing a final dataset'
    selected=checkpoint.selected;minus=problem.check(selected);negative_pairs=problem.candidates[selected]
    metrics=problem.metrics(selected)
    from solver import author
    pj=author._pairwise_jaccard(problem.pos,problem.membership,problem.go_sizes)
    nj=problem.jac[selected]
    stats=dict(**result,**{k:v for k,v in metrics.items() if k not in result},
        split=split,positives=problem.k,negatives=problem.k,proteins=problem.n,candidates=problem.m,
        admissible_pairs=available,solver='HIGHS',candidate_fingerprint=receipt['fingerprint'],
        recovery_contract=contract['fingerprint'],initial_diagnostic=initial,
        normalized_objective_reduction_fraction=(initial['objective']-checkpoint.objective)/initial['objective'],
        positive_go_jaccard_quantiles=np.quantile(pj,[0,.25,.5,.75,1]).tolist(),
        negative_go_jaccard_quantiles=np.quantile(nj,[0,.25,.5,.75,1]).tolist(),
        positive_negative_degree_spearman=float(spearmanr(problem.plus,minus).statistic),
        proteins_without_go=sum(not g for g in problem.go),proteins_absent_from_negatives=int((minus==0).sum()),
        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
        candidate_adaptations=read_json(ROOT/'reports/test-ilp-negative-sampling.json')['candidate_adaptations'])
    write_csv(directory/'degree-residuals.tsv',['protein_id','positive_degree','negative_degree','residual'],
        (dict(protein_id=p,positive_degree=int(problem.plus[i]),negative_degree=int(minus[i]),residual=int(minus[i]-problem.plus[i])) for i,p in enumerate(problem.proteins)),delimiter='\t')
    output=[]
    for i,r in enumerate(rows):
        output.append(dict(pair_id=f'{split}:positive:{i}',protein1=r['protein1'],protein2=r['protein2'],label=1,
                           original_row_id='',first_source_row_id=r['first_source_row_id']))
    for i,(a,b) in enumerate(negative_pairs):
        output.append(dict(pair_id=f'{split}:negative:{i}',protein1=problem.proteins[a],protein2=problem.proteins[b],label=0,
                           original_row_id='',first_source_row_id=''))
    path=ROOT/f'prepared/{split}.csv'
    write_csv(path,['pair_id','protein1','protein2','label','original_row_id','first_source_row_id'],output)
    stats['output_sha256']=sha(path)
    write_json(ROOT/f'reports/{split}-negative-sampling.json',stats)
    mark(f'negative_{split}','complete',summary=stats)


if __name__=='__main__':
    split=sys.argv[1]
    try:main(split)
    except Exception as exc:mark(f'negative_{split}','failed',recovery='v1',error=str(exc));raise
