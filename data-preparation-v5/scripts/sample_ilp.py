"""Pinned Bernett objective; explicit non-self/known-positive-safe candidate pool."""
import csv
import hashlib
import json
import resource
import sys
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import cvxpy as cp
from scipy.stats import spearmanr
from common import ROOT,read_json,write_json,write_csv,pair,mark,sha

AUTHOR=ROOT/'sources/author-pipeline/bin'
sys.path.insert(0,str(AUTHOR))
import sample_negatives_ilp as author

def candidate_pool(n,pos,forbidden,go_bp,target,seed=2):
    """Author-style degree-weighted draw and GO priority; random excess selection.

    The upstream helper truncates sorted pair keys, privileging low indices.
    This wrapper samples excess candidates within each GO stratum instead.
    """
    rng=np.random.default_rng(seed)
    weights=np.bincount(pos.ravel(),minlength=n).astype(float);weights/=weights.sum()
    membership,sizes=author._build_go_membership(go_bp)
    priority=np.empty(0,dtype=np.int64);filler=np.empty(0,dtype=np.int64)
    for _ in range(200):
        have=len(priority)+len(filler)
        if have>=target:break
        count=max(10000,5*(target-have))
        a=rng.choice(n,size=count,p=weights);b=rng.choice(n,size=count,p=weights)
        keys=np.unique(np.minimum(a,b)*n+np.maximum(a,b))
        keys=keys[~np.isin(keys,forbidden,assume_unique=True)]
        old=np.union1d(priority,filler)
        if len(old):keys=keys[~np.isin(keys,old,assume_unique=True)]
        pairs=np.stack((keys//n,keys%n),axis=1)
        jac=np.concatenate([author._pairwise_jaccard(pairs[i:i+50000],membership,sizes) for i in range(0,len(pairs),50000)]) if len(pairs) else np.empty(0)
        priority=np.union1d(priority,keys[jac>0]);filler=np.union1d(filler,keys[jac==0])
    if len(priority)+len(filler)<target:raise RuntimeError('Candidate generation did not reach its frozen target; no silent fallback.')
    if len(priority)>=target:chosen=rng.choice(priority,size=target,replace=False)
    else:chosen=np.concatenate((priority,rng.choice(filler,size=target-len(priority),replace=False)))
    chosen=np.sort(chosen)
    assert len(chosen)==target and len(np.unique(chosen))==target and not np.isin(chosen,forbidden).any()
    return np.stack((chosen//n,chosen%n),axis=1)

def main(split):
    assert split in ['train','val','test-ilp']
    stage=f'negative_{split}';mark(stage,'running')
    config=read_json(ROOT/'configuration.json')['negative_sampling']
    pos_path=ROOT/('frozen-tests/original-positives.csv' if split=='test-ilp' else f'work/{split}-positives.csv')
    rows=list(csv.DictReader(pos_path.open()));proteins=sorted({p for r in rows for p in (r['protein1'],r['protein2'])});index={p:i for i,p in enumerate(proteins)}
    meta=read_json(ROOT/('work/test-protein-metadata.json' if split=='test-ilp' else 'work/protein-registry.json'))
    go=[frozenset(meta[p]['go_bp']) for p in proteins]
    pos=np.asarray([sorted((index[r['protein1']],index[r['protein2']])) for r in rows],dtype=np.int64)
    n=len(proteins);assert (pos[:,0]!=pos[:,1]).all() and len(np.unique(pos,axis=0))==len(pos)
    families=defaultdict(set)
    for p,i in index.items():
        for f in meta[p]['families']:families[f].add(i)
    blocked={i*n+i for i in range(n)}
    blocked.update((pos[:,0]*n+pos[:,1]).tolist())
    for r in csv.DictReader((ROOT/'work/known-positive-families.tsv').open(),delimiter='\t'):
        for a in families.get(r['family1'],()):
            for b in families.get(r['family2'],()):
                i,j=sorted((a,b));blocked.add(i*n+j)
    forbidden=np.array(sorted(blocked),dtype=np.int64)
    available=n*(n+1)//2-len(forbidden)
    target=min(4*len(pos),available)
    assert target>=len(pos),dict(available=available,needed=len(pos))
    directory=ROOT/'work'/f'negative-{split}';directory.mkdir(exist_ok=True)
    spec=dict(split=split,positive_sha256=sha(pos_path),metadata_sha256=sha(ROOT/('work/test-protein-metadata.json' if split=='test-ilp' else 'work/protein-registry.json')),
              blacklist_sha256=sha(ROOT/'work/known-positive-families.tsv'),author_code_sha256=sha(AUTHOR/'sample_negatives_ilp.py'),wrapper_code_sha256=sha(Path(__file__)),
              configuration=config,proteins=proteins,candidate_target=target,seed=2)
    fingerprint=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()
    candidate_path=directory/'candidates.npz';receipt=directory/'candidate-manifest.json'
    if candidate_path.exists():
        saved=read_json(receipt);assert saved['fingerprint']==fingerprint and saved['sha256']==sha(candidate_path)
        candidates=np.load(candidate_path)['pairs']
    else:
        candidates=candidate_pool(n,pos,forbidden,go,target)
        tmp=candidate_path.with_suffix('.part')
        with tmp.open('wb') as f:np.savez_compressed(f,pairs=candidates)
        tmp.replace(candidate_path)
        write_json(receipt,dict(fingerprint=fingerprint,sha256=sha(candidate_path),specification=spec))
    assert not np.isin(candidates[:,0]*n+candidates[:,1],forbidden).any()
    write_json(directory/'protein-order.json',proteins)
    np.save(directory/'positive-pairs.npy',pos)
    cfg=author.SamplingConfig(lambda_degree=1,lambda_taxon_pair=0,lambda_self_loop=0,lambda_jaccard=1,
          solver='highs',time_limit=config['time_limit_seconds'],mip_gap=config['mip_gap_target'],threads=config['threads'],seed=2,verbose=True,max_candidates=target)
    ctx=author.build_context(pos,index,proteins,candidates,1.,go_bp=go)
    biases=author.assemble_active_biases(cfg)
    for b in biases:b.precompute(ctx)
    active=[b for b in biases if b.is_active()]
    assert {b.name for b in active}=={'degree','jaccard'}
    problem,x,terms=author.build_problem(ctx,active)
    start=time.monotonic()
    result=author.solve(problem,cp.HIGHS,author._solver_options(cp.HIGHS,cfg),verbose=True)
    negatives=author.extract_negatives(x.value,ctx)
    plus=np.bincount(pos.ravel(),minlength=n);minus=np.bincount(negatives.ravel(),minlength=n)
    assert len(negatives)==len(pos) and (negatives[:,0]!=negatives[:,1]).all()
    assert np.all(minus<=6*plus) and not np.isin(negatives[:,0]*n+negatives[:,1],forbidden).any()
    assert len(np.unique(negatives,axis=0))==len(negatives)
    membership,sizes=author._build_go_membership(go)
    pj=author._pairwise_jaccard(pos,membership,sizes);nj=author._pairwise_jaccard(negatives,membership,sizes)
    info=problem.solver_stats.extra_stats
    stats=dict(split=split,positives=len(pos),negatives=len(negatives),proteins=n,candidates=len(candidates),admissible_pairs=available,
               solver='HIGHS',status=result['status'],solver_seconds=result['wall_time_s'],total_solve_and_check_seconds=time.monotonic()-start,
               objective=float(problem.value),configured_mip_gap=cfg.mip_gap,actual_mip_gap=float(info.mip_gap),dual_bound=float(info.mip_dual_bound),
               nodes=int(info.mip_node_count),mathematically_proven_optimal=bool(abs(info.mip_gap)<1e-9),
               objective_terms={b.name:float(expr.value) for b,expr in terms},positive_go_jaccard_mean=float(pj.mean()),negative_go_jaccard_mean=float(nj.mean()),
               positive_go_jaccard_quantiles=np.quantile(pj,[0,.25,.5,.75,1]).tolist(),negative_go_jaccard_quantiles=np.quantile(nj,[0,.25,.5,.75,1]).tolist(),
               proteins_without_go=sum(not g for g in go),proteins_absent_from_negatives=int((minus==0).sum()),
               positive_negative_degree_spearman=float(spearmanr(plus,minus).statistic),degree_absolute_residual_sum=int(np.abs(minus-plus).sum()),
               max_negative_positive_degree_ratio=float((minus/plus).max()),candidate_fingerprint=fingerprint,
               peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,
               candidate_adaptations=['known-positive family blacklist','hard exclusion of all self pairs','random excess selection within GO strata instead of sorted-key truncation'])
    write_csv(directory/'degree-residuals.tsv',['protein_id','positive_degree','negative_degree','residual'],
              (dict(protein_id=p,positive_degree=int(plus[i]),negative_degree=int(minus[i]),residual=int(minus[i]-plus[i])) for i,p in enumerate(proteins)),delimiter='\t')
    original_negatives={}
    if split=='test-ilp':
        for i,r in enumerate(csv.DictReader((ROOT/'frozen-tests/original-test.csv').open())):
            if r['label']=='0':original_negatives[pair(r['Uniprot_a'],r['Uniprot_b'])]=i
    output=[]
    for i,r in enumerate(rows):
        output.append(dict(pair_id=f'{split}:positive:{i}',protein1=r['protein1'],protein2=r['protein2'],label=1,
                           original_row_id=r.get('original_row_id',''),first_source_row_id=r.get('first_source_row_id','')))
    overlap=0
    for i,(a,b) in enumerate(negatives):
        p,q=proteins[int(a)],proteins[int(b)];old=original_negatives.get(pair(p,q),'');overlap+=old!=''
        output.append(dict(pair_id=f'{split}:negative:{i}',protein1=p,protein2=q,label=0,original_row_id=old,first_source_row_id=''))
    if split=='test-ilp':
        assert len(output)==52048 and n==2948
        stats.update(original_negative_overlap=overlap,reusable_original_prediction_rows=26024+overlap,new_negative_predictions_per_historical_model=26024-overlap)
    out=ROOT/(f'frozen-tests/{split}.csv' if split=='test-ilp' else f'prepared/{split}.csv')
    write_csv(out,['pair_id','protein1','protein2','label','original_row_id','first_source_row_id'],output)
    stats['output_sha256']=sha(out)
    write_json(ROOT/'reports'/f'{split}-negative-sampling.json',stats)
    mark(stage,'complete',summary=stats)

if __name__=='__main__':
    split=sys.argv[1]
    try:main(split)
    except Exception as exc:mark(f'negative_{split}','failed',error=str(exc));raise
