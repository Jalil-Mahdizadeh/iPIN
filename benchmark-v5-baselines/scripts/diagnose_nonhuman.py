"""Post-result explanation of the v5 homology baseline's fallback ordering."""
import csv
from nonhuman_common import *
from evaluate import write_csv

def main():
    frozen=read(ROOT/'provenance/nonhuman/predictions-frozen.json');fallback=frozen['references']['v5']['no_hit_propensity']
    values=np.load(NMODELS/'v5-protein-values.npz');hit=values['homolog_indices'][:,0]>=0
    meta=read(NDATA/'sequences.json');rows=[]
    for species in SPECIES:
        data=np.load(NDATA/(species+'-rows.npz'));p,y=data['pairs'],data['labels']
        score=np.load(ROOT/'results'/(species+'-v5-baseline-scores.npz'))['homology-degree']
        nohits=(~hit[p]).sum(1);zero_or_fallback=nohits*fallback
        agreement=np.isclose(score,zero_or_fallback,rtol=0,atol=1e-15)
        for label in [1,0]:
            m=y==label
            rows.append({'dataset':species,'label':label,'pairs':int(m.sum()),'no_hit_fallback':fallback,
                'pairs_matching_zero_for_hit_plus_fallback_for_nohit':int((m&agreement).sum()),
                'fraction_matching_fallback_ordering':float(agreement[m].mean()),
                'zero_unmatched_endpoints':int((m&(nohits==0)).sum()),'one_unmatched_endpoint':int((m&(nohits==1)).sum()),
                'two_unmatched_endpoints':int((m&(nohits==2)).sum()),
                'median_homology_degree_score':float(np.median(score[m])),
                'mean_homology_degree_score':float(score[m].mean())})
    write_csv(ROOT/'results/nonhuman-v5-fallback-diagnostic.csv',rows)
    atomic(ROOT/'provenance/nonhuman/fallback-diagnostic.json',{'timing':'Post-result diagnostic; no primary method or score changed.',
        'purpose':'Check how often existing predictions equal zero for a matched balanced TRAIN target plus the fixed mean fallback for an unmatched endpoint.',
        'no_hit_fallback':fallback,'inversion_performed':False,'alternative_predictor_evaluated':False,
        'artifact':record(ROOT/'results/nonhuman-v5-fallback-diagnostic.csv')})

if __name__=='__main__':main()
