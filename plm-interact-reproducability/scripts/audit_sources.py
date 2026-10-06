"""Audit deposited scores and sample identity independently of model inference."""
import collections
import datetime
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from metrics_eval import metrics
from datasets_eval import ROOT,SPECIES

published=ROOT/'data/published'
records=[];audit={};mutation_map=[]
for s in SPECIES:
    test=pd.read_csv(ROOT/f'data/cross_species_benchmarking/test/{s}.ppi.qrels.seq.test.csv')
    f=published/f'figure2_PLM-interact_{s}.csv';ref=pd.read_csv(f)
    assert len(test)==len(ref) and np.array_equal(test.label,ref.label)
    lengths=test['query'].str.len()+test.text.str.len()+3
    # Sequence identity binding will also be checked against the publisher's
    # supplementary CSV, which includes the actual input sequences.
    audit[s]=dict(n=len(test),positives=int(test.label.sum()),source_label_order_equal=True,
                  max_tokens=int(lengths.max()),over1603=int((lengths>1603).sum()),
                  exact_ordered_sequence_duplicates=int(test.duplicated(['query','text']).sum()))
    audit[s]['baseline_sample_id_comparison']={}
    key_counts=lambda frame: collections.Counter((*sorted((a,b)),int(y)) for a,b,y in frame[['protein_a','protein_b','label']].itertuples(index=False,name=None))
    ref_counts=key_counts(ref)
    for model in ['PLM-interact','TUnA','TT3D','Topsy-Turvy','D-SCRIPT']:
        d=pd.read_csv(published/f'figure2_{model}_{s}.csv')
        dc=key_counts(d)
        audit[s]['baseline_sample_id_comparison'][model]=dict(n=len(d),unordered_id_pair_and_label_multiset_overlap=sum((dc&ref_counts).values()),same_ids_and_labels_in_same_order=bool(np.array_equal(d[['protein_a','protein_b','label']],ref[['protein_a','protein_b','label']])))
        records.append(dict(figure='2',dataset=s,model=model,**metrics(d.label,d.score)))
    rev=pd.read_csv(published/f'supplement6_{s}.csv')
    assert np.array_equal(rev.label,ref.label) and np.array_equal(rev.reverse_label,ref.label)
    assert np.allclose(rev.original,ref.score,rtol=0,atol=1e-12)
    records.append(dict(figure='S5-S6',dataset=s,model='PLM-interact-reversed',**metrics(rev.label,rev.reverse)))
    delta=(rev.original-rev.reverse).abs()
    audit[s]['published_reverse_concordance']=dict(spearman=float(spearmanr(rev.original,rev.reverse).statistic),mae=float(delta.mean()),max_abs=float(delta.max()),classification_flips_at_0_5=int(((rev.original>=0.5)!=(rev.reverse>=0.5)).sum()))

b=pd.read_csv(ROOT/'data/Bernett_benchmarking/pairs_uniprot_seqs_test.csv')
for model in ['PLM-interact','TUnA']:
    d=pd.read_csv(published/f'figure4_{model}.csv')
    assert len(d)==len(b)
    if model=='PLM-interact':
        assert np.array_equal(d[['protein_a','protein_b']],b[['Uniprot_a','Uniprot_b']])
        assert np.array_equal(d.label,b.label)
    records.append(dict(figure='4',dataset='Bernett',model=model,**metrics(d.label,d.score)))
blen=b['query'].str.len()+b.text.str.len()+3
audit['bernett']=dict(n=len(b),positives=int(b.label.sum()),source_ids_labels_order_equal=True,
                     max_tokens=int(blen.max()),over1603=int((blen>1603).sum()),over2196=int((blen>2196).sum()),
                     exact_ordered_sequence_duplicates=int(b.duplicated(['query','text']).sum()))

m=pd.read_csv(ROOT/'data/Mutation_effect_dataset/test_mutation_data.csv',keep_default_na=False,dtype=str)
d=pd.read_csv(published/'figure5_mutation.csv',keep_default_na=False,dtype=str)
keys=['affected_uniprot','parti_uniprot','label','Affected_species','Participant_species',
      'Feature type','Feature range(s)','Original sequence','Resulting sequence','PubMedID','Interaction AC']
lookup=collections.defaultdict(collections.deque)
for i,row in m.iterrows():lookup[tuple(row[k] for k in keys)].append(int(i))
excel_date_repairs=[]
for i,row in d.iterrows():
    key=tuple(row[k] for k in keys)
    if not lookup[key] and row['Feature range(s)'].isdigit():
        # Excel converted some short ranges (e.g. '8-8') into date serials.
        # Require a unique match on every other metadata field, then independently
        # verify the serial's month/day against the released residue range.
        serial=int(row['Feature range(s)'])
        date=datetime.date(1899,12,30)+datetime.timedelta(days=serial)
        candidates=[]
        for candidate_key,ids in lookup.items():
            if ids and all(a==b for k,a,b in zip(keys,key,candidate_key) if k!='Feature range(s)'):
                span=candidate_key[keys.index('Feature range(s)')]
                if span in (f'{date.month}-{date.day}',f'{date.day}-{date.month}'):
                    candidates.append(candidate_key)
        assert len(candidates)==1,('Ambiguous Excel date repair',i,candidates)
        key=candidates[0]
        excel_date_repairs.append(dict(source_row=int(i)+2,original_serial=serial,decoded_date=str(date),released_range=key[keys.index('Feature range(s)')]))
    assert lookup[key],('Unmatched published mutation',i,key)
    mutation_map.append(dict(source_row=int(i)+2,row_id=lookup[key].popleft()))
indices=[r['row_id'] for r in mutation_map]
length_mask=(m[['wild_seq','mutant_seq','participant_sequence']].map(len).max(axis=1)<=2000)
assert set(indices)==set(np.flatnonzero(length_mask)) and len(indices)==598
pd.DataFrame(mutation_map).to_csv(ROOT/'data/mutation-figure5-mapping.csv',index=False)
for column in d.columns:
    if column.endswith('_scores'):
        records.append(dict(figure='5',dataset='mutation_published598',model=column.removesuffix('_scores'),**metrics(d.label.astype(int),d[column].astype(float))))
audit['mutation']=dict(full_n=len(m),published_n=len(d),full_positive=int(m.label.astype(int).sum()),published_positive=int(d.label.astype(int).sum()),
                       published_subset_matched_by_11_metadata_fields_after_verified_excel_date_repair=True,excel_date_repairs=excel_date_repairs,subset_equals_each_protein_at_most2000_residues=True,
                       all_released_cases_human=bool((m.Affected_species=='9606 - Homo sapiens').all() and (m.Participant_species=='9606 - Homo sapiens').all()),
                       rows_with_literal_gap_dot=[int(i) for i in m.index[m.mutant_seq.str.contains('.',regex=False)]],
                       duplicate_sequence_triplets=int(m.duplicated(['wild_seq','mutant_seq','participant_sequence']).sum()))

pd.DataFrame(records).to_csv(ROOT/'results/published-score-metrics.csv',index=False)
(ROOT/'results/data-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(pd.DataFrame(records)[['figure','dataset','model','average_precision','auroc','precision','recall','f1']].to_string(index=False))
