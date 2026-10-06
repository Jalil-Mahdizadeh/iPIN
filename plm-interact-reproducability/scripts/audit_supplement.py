"""Sequence-level equivalence, supplementary ablation and identity-bin audits."""
import collections
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import chi2,binomtest
from sklearn.metrics import average_precision_score
from datasets_eval import ROOT,SPECIES
from metrics_eval import metrics

extra=ROOT/'data/published-extra';audit={};bins=[];masks=[];mcnemar=[]
small=json.loads((ROOT.parent/'literature/plm-interact/source-data-small-tables.json').read_text())
published_mcnemar={}
for row in small['Supplemtrary Figure2']:
    r={''.join(c for c in k if c.isalpha()):v for k,v in row.items()}
    if r.get('A') in SPECIES and r.get('C','').startswith('mask_') and r.get('G','-')!='-':
        published_mcnemar[(r['A'],r['C'].replace('%',''))]=r
seq_counter=lambda d:collections.Counter((*sorted((a,b)),int(y)) for a,b,y in d[['query','text','label']].itertuples(index=False,name=None))
id_counter=lambda d,cols:collections.Counter((*sorted((a,b)),int(y)) for a,b,y in d[cols].itertuples(index=False,name=None))
for s in SPECIES:
    p=pd.read_csv(extra/f'Supplementary Figure7/{s}_identity_plminteract.csv')
    t=pd.read_csv(extra/f'Supplementary Figure7/{s}_identity_TUnA.csv')
    released=pd.read_csv(ROOT/f'data/cross_species_benchmarking/test/{s}.ppi.qrels.seq.test.csv')
    assert np.array_equal(p[['query','text','label']],released[['query','text','label']])
    pub=pd.read_csv(ROOT/f'data/published/figure2_PLM-interact_{s}.csv')
    assert np.array_equal(p[['query_id','text_id','label']],pub[['protein_a','protein_b','label']])
    assert np.allclose(p['PLM-interact_predicted_interaction_probability'],pub.score,atol=1e-12,rtol=0)
    pc,tc=seq_counter(p),seq_counter(t)
    assert pc==tc,('Different sequence-level test sets',s)
    tuna_ids=id_counter(t,['query_id','text_id','label'])
    for model in ['TUnA','TT3D','Topsy-Turvy','D-SCRIPT']:
        d=pd.read_csv(ROOT/f'data/published/figure2_{model}_{s}.csv')
        assert id_counter(d,['protein_a','protein_b','label'])==tuna_ids
    tuna_pub=pd.read_csv(ROOT/f'data/published/figure2_TUnA_{s}.csv')
    assert np.array_equal(t[['query_id','text_id','label']],tuna_pub[['protein_a','protein_b','label']])
    assert np.allclose(t['TUnA_predicted_interaction_probability'],tuna_pub.score,rtol=0,atol=1e-10)
    audit[s]=dict(released_test_equals_publisher_sequences_in_order=True,
                  released_test_bound_to_workbook_ids_labels_scores=True,
                  plm_tuna_unordered_sequence_label_multisets_identical=True,
                  all_baseline_id_label_multisets_identical_to_tuna=True,
                  n=len(p),unique_unordered_sequence_label_pairs=len(pc),duplicate_unordered_sequence_label_rows=len(p)-len(pc))
    # Zero identity is a separate bin in the deposited supplementary figure.
    for model,d,score in [('PLM-interact',p,'PLM-interact_predicted_interaction_probability'),('TUnA',t,'TUnA_predicted_interaction_probability')]:
        selections=[('0',d.max_percent_identity==0)]+[(f'({lo},{lo+20}]',(d.max_percent_identity>lo)&(d.max_percent_identity<=lo+20)) for lo in range(0,100,20)]
        for name,mask in selections:
            sub=d[mask]
            if len(sub)==0:continue
            ap=float(average_precision_score(sub.label,sub[score])) if sub.label.sum()>0 else 0.0
            bins.append(dict(species=s,model=model,identity_bin=name,n=len(sub),positives=int(sub.label.sum()),average_precision=ap))
    maskdata=pd.read_csv(extra/f'Supplementary Figure2/{s}_source_data_sfigure2_scores_all_models.csv')
    assert np.array_equal(maskdata.label,pub.label)
    # Masked-model columns contain raw logits; the unmasked column instead
    # contains probabilities. Apply the appropriate fixed threshold to each.
    assert np.allclose(expit(maskdata.mask_15),pub.score,atol=1e-7,rtol=0)
    correct_a=(maskdata['0%(binary)']>=.5)==maskdata.label
    for column in ['0%(binary)','mask_7','mask_15','mask_22','mask_30']:
        masks.append(dict(species=s,mask=column,**metrics(maskdata.label,maskdata[column],threshold=.5 if column=='0%(binary)' else 0)))
        if column=='0%(binary)':continue
        correct_b=(maskdata[column]>=0)==maskdata.label
        b=int((correct_a&~correct_b).sum());c=int((~correct_a&correct_b).sum())
        reference=published_mcnemar[(s,column)]
        rb,rc=int(reference['G']),int(reference['H'])
        mcnemar.append(dict(species=s,mask=column,b=b,c=c,p_uncorrected=float(chi2.sf((b-c)**2/(b+c),1)),p_continuity_corrected=float(chi2.sf((max(0,abs(b-c)-1))**2/(b+c),1)),more_correct_than_unmasked=c>b,
                            reported_b=rb,reported_c=rc,discordant_counts_match=(b,c)==(rb,rc),reported_p=float(reference['F']),reported_counts_cc_p=float(chi2.sf((max(0,abs(rb-rc)-1))**2/(rb+rc),1)),reported_counts_exact_p=float(binomtest(min(rb,rc),rb+rc,.5).pvalue)))

pd.DataFrame(bins).to_csv(ROOT/'results/published-identity-bin-metrics.csv',index=False)
pd.DataFrame(masks).to_csv(ROOT/'results/published-mask-ablation-metrics.csv',index=False)
pd.DataFrame(mcnemar).to_csv(ROOT/'results/published-mask-mcnemar.csv',index=False)
(ROOT/'results/sequence-identity-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
print('All sequence-level matches passed. McNemar agreement flags are recorded separately; mismatches are findings, not silently corrected.')
