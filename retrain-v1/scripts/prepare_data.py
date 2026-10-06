"""Build auditable human-only benchmark arrays without sampling new negatives."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
    return h.hexdigest()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--finalize',action='store_true');args=parser.parse_args()
    out=ROOT/'data/prepared';out.mkdir(parents=True,exist_ok=True)
    if not args.finalize:
        seqs=[];mapping={};rows={};summary={};vocab=(ROOT/'assets/esm2/vocab.txt').read_text().splitlines();lookup={s:i for i,s in enumerate(vocab)}
        for split in ['train','val','test']:
            seen={};records=[];duplicates=[];conflicts=[]
            with (ROOT/f'data/raw/pairs_uniprot_seqs_{split}.csv').open() as f:
                for idx,r in enumerate(csv.DictReader(f)):
                    pair=[]
                    for key in ['query','text']:
                        seq=r[key].strip();assert seq and all(c in lookup and len(c)==1 for c in seq)
                        if seq not in mapping:mapping[seq]=len(seqs);seqs.append(seq)
                        pair.append(mapping[seq])
                    label=int(r['label']);assert label in (0,1)
                    key=tuple(sorted(pair))
                    if key in seen:
                        (duplicates if seen[key]==label else conflicts).append(idx);continue
                    seen[key]=label;records.append([*pair,label,idx])
            assert not conflicts,(split,conflicts)
            rows[split]=np.asarray(records,dtype=np.int64);np.save(out/f'{split}.raw.npy',rows[split])
            summary[split]={'raw_rows':idx+1,'unique_rows':len(records),'duplicate_rows':duplicates,'conflicting_rows':conflicts}
        offsets=[0];tokenarrays=[]
        for seq in seqs:tokenarrays.append(np.asarray([lookup[c] for c in seq],dtype=np.uint8));offsets.append(offsets[-1]+len(seq))
        np.save(out/'tokens.npy',np.concatenate(tokenarrays));np.save(out/'offsets.npy',np.asarray(offsets,dtype=np.int64))
        protsets={s:set(rows[s][:,:2].ravel().tolist()) for s in rows}
        for split,ids in protsets.items():
            with (out/f'{split}.fasta').open('w') as f:
                for idx in sorted(ids):f.write(f'>p{idx}\n{seqs[idx]}\n')
        with (out/'heldout.fasta').open('w') as f:
            for idx in sorted(protsets['val']|protsets['test']):f.write(f'>p{idx}\n{seqs[idx]}\n')
        summary['exact_sequence_overlaps']={f'{a}_{b}':len(protsets[a]&protsets[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
        summary['unique_proteins']=len(seqs)
        (out/'initial-audit.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
        return
    summary=json.loads((out/'initial-audit.json').read_text());offsets=np.load(out/'offsets.npy');lengths=np.diff(offsets)
    arrays={s:np.load(out/f'{s}.raw.npy') for s in ['train','val','test']}
    sets={s:set(arrays[s][:,:2].ravel().tolist()) for s in arrays}
    excluded_train=sets['train']&(sets['val']|sets['test']);excluded_val=sets['val']&sets['test']
    for filename,exclude,column in [('heldout-vs-train.tsv',excluded_train,1),('val-vs-test.tsv',excluded_val,0)]:
        with (ROOT/'data'/filename).open() as f:
            for line in f:
                fields=line.rstrip().split('\t')
                if float(fields[2])>=.4 and float(fields[3])>=.8 and float(fields[4])>=.8:exclude.add(int(fields[column][1:]))
    report={'benchmark':'Bernett human PPI, authors pinned release','source_revision':'5d2ad03baa165c27df32a2eadf066462a2a83073',
      'test_use':'sequence-only split audit; no test-model evaluation or tuning',
      'homology_rule':'MMseqs2 detected identity >=0.40 and coverage >=0.80 of both sequences; train purged against val+test; val purged against test',
      'new_negative_sampling':False,'negative_interpretation':'original benchmark sampled unreported interactions, not confirmed noninteractions',
      'excluded_training_protein_ids':sorted(excluded_train),'excluded_validation_protein_ids':sorted(excluded_val),'initial_audit':summary,'splits':{}}
    for split,a in arrays.items():
        bad=excluded_train if split=='train' else excluded_val if split=='val' else set()
        mask=np.asarray([int(x) not in bad and int(y) not in bad for x,y in a[:,:2]])
        removed=a[~mask];keep=a[mask];np.save(out/f'{split}.npy',keep)
        np.save(out/f'{split}.excluded.npy',removed)
        ls=lengths[keep[:,0]]+lengths[keep[:,1]]+3
        report['splits'][split]={'rows':len(keep),'positives':int(keep[:,2].sum()),'excluded_for_homology':len(removed),
          'sequence_length_exclusions':0,'max_tokens':int(ls.max()),'p50_tokens':float(np.quantile(ls,.5)),
          'p95_tokens':float(np.quantile(ls,.95)),'p99_tokens':float(np.quantile(ls,.99)),
          'over2196':int((ls>2196).sum()),'over8192':int((ls>8192).sum()),'sha256':sha(out/f'{split}.npy')}
    report['files']={p.name:sha(p) for p in [out/'tokens.npy',out/'offsets.npy',out/'train.npy',out/'val.npy',out/'test.npy']}
    (out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['splits'],indent=2))

if __name__=='__main__':main()
