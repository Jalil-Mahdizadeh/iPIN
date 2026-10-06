"""Read-only overlap and length audit; these files are never used for fitting."""
import collections
import csv
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
base=ROOT/'data/Mutation_effect_dataset'
splits={name:list(csv.DictReader(open(base/file))) for name,file in [('train','training_mutation_data.csv'),('validation','val_mutation_data.csv'),('test','test_mutation_data.csv')]}
def triple(row):return hashlib.sha256('\0'.join(row[k] for k in ['wild_seq','mutant_seq','participant_sequence']).encode()).hexdigest()
summary={}
for name,rows in splits.items():
    lengths=sorted(max(len(r['wild_seq']),len(r['mutant_seq']))+len(r['participant_sequence']) for r in rows)
    summary[name]=dict(n=len(rows),positive=sum(int(r['label']) for r in rows),min_combined_residues=min(lengths),max_combined_residues=max(lengths),median_combined_residues=lengths[len(lengths)//2],combined_residues_over2200=sum(x>2200 for x in lengths),duplicate_sequence_triplets=len(rows)-len(set(map(triple,rows))),identical_wild_mutant=sum(r['wild_seq']==r['mutant_seq'] for r in rows))
train=splits['train'];test=splits['test']
train_triples=set(map(triple,train));val_triples=set(map(triple,splits['validation']))
train_pairs={tuple(sorted((r['wild_seq'],r['participant_sequence']))) for r in train}
train_proteins={r[k] for r in train for k in ['wild_seq','participant_sequence']}
overlap=dict(test_exact_triplets_in_training=sum(triple(r) in train_triples for r in test),
             test_exact_triplets_in_validation=sum(triple(r) in val_triples for r in test),
             test_wild_partner_sequence_pairs_seen_in_training=sum(tuple(sorted((r['wild_seq'],r['participant_sequence']))) in train_pairs for r in test),
             test_both_wild_and_partner_sequences_seen_in_training=sum(r['wild_seq'] in train_proteins and r['participant_sequence'] in train_proteins for r in test),
             test_at_least_one_wild_or_partner_sequence_seen_in_training=sum(r['wild_seq'] in train_proteins or r['participant_sequence'] in train_proteins for r in test))
out=dict(purpose='Read-only audit; no training, tuning, or additional inference.',splits=summary,overlap=overlap)
(ROOT/'results/mutation-split-audit.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
