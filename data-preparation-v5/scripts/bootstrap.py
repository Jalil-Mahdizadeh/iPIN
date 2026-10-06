"""Freeze the original endpoint and inventory HIPPIE without model predictions."""
import csv
import gzip
import shutil
from pathlib import Path
import numpy as np
from common import ROOT, PROJECT, sha, write_json, read_json, write_csv, fasta_write, pair, mark

def main():
    mark('bootstrap','running')
    source=PROJECT/'retrain-v1/data/raw/pairs_uniprot_seqs_test.csv'
    target=ROOT/'frozen-tests/original-test.csv'
    if target.exists(): assert sha(source)==sha(target)
    else: shutil.copy2(source,target)
    rows=list(csv.DictReader(target.open()))
    assert len(rows)==52048 and sum(int(r['label']) for r in rows)==26024
    seqs={}; positive_rows=[]; positive_ids=set(); original_pairs=set()
    bm=PROJECT/'benchmark-v4/data'
    manifest=read_json(bm/'prepared-manifest.json')
    for name in ['test.npy','tokens.npy','offsets.npy']:
        assert sha(bm/name)==manifest['files'][name],name
    a=np.load(bm/'test.npy'); tokens=np.load(bm/'tokens.npy',mmap_mode='r'); offsets=np.load(bm/'offsets.npy')
    vocab=(PROJECT/'retrain-v1/assets/esm2/vocab.txt').read_text().splitlines()
    decoded={}
    def sequence(i):
        i=int(i)
        if i not in decoded:decoded[i]=''.join(vocab[int(t)] for t in tokens[offsets[i]:offsets[i+1]])
        return decoded[i]
    for i,r in enumerate(rows):
        key=pair(r['Uniprot_a'],r['Uniprot_b'])
        assert key not in original_pairs
        original_pairs.add(key)
        assert int(a[i,3])==i and int(a[i,2])==int(r['label'])
        assert sequence(a[i,0])==r['query'] and sequence(a[i,1])==r['text']
        for accession,seq in [(r['Uniprot_a'],r['query']),(r['Uniprot_b'],r['text'])]:
            assert accession not in seqs or seqs[accession]==seq
            seqs[accession]=seq
        assert r['query']!=r['text']
        if r['label']=='1':
            positive_rows.append(dict(protein1=r['Uniprot_a'],protein2=r['Uniprot_b'],label=1,original_row_id=i))
            positive_ids.update(key)
    assert len(seqs)==3022 and len(set(seqs.values()))==3022 and len(positive_ids)==2948
    fasta_write(ROOT/'frozen-tests/original-sequences.fasta',seqs)
    write_csv(ROOT/'frozen-tests/original-positives.csv',['protein1','protein2','label','original_row_id'],positive_rows)
    write_csv(ROOT/'frozen-tests/protected-proteins.tsv',['protein_id','sequence_sha256','length','positive_endpoint'],
              (dict(protein_id=p,sequence_sha256=__import__('hashlib').sha256(seq.encode()).hexdigest(),length=len(seq),positive_endpoint=int(p in positive_ids)) for p,seq in sorted(seqs.items())),delimiter='\t')
    hippie=[]; ids=set(); keys=set(); scores={}; self_count=0
    with gzip.open(ROOT/'sources/hippie-v3-snapshot.txt.gz','rt') as f:
        for i,r in enumerate(csv.DictReader(f,delimiter='\t')):
            p,q=r['uniprot_accession_A'],r['uniprot_accession_B']
            assert p and q and p!='-' and q!='-'
            score=float(r['score']); assert 0<=score<=1
            key=pair(p,q);keys.add(key);scores[key]=score;ids.update(key);self_count+=p==q
            hippie.append(dict(protein1=p,protein2=q,score=score,source_row_id=i))
    author=list(csv.DictReader((ROOT/'sources/author-hippie-v3.csv').open()))
    author_scores={pair(r['protein1'],r['protein2']):float(r['score']) for r in author}
    source_audit=dict(raw_rows=len(hippie),unique_unordered_pairs=len(keys),accessions=len(ids),self_rows=self_count,
                      isoform_accessions=sum('-' in p for p in ids),matches_author_released_pairs=keys==set(author_scores),
                      matches_author_released_scores=scores==author_scores)
    assert source_audit['matches_author_released_pairs'] and source_audit['matches_author_released_scores'],source_audit
    write_csv(ROOT/'work/hippie-positive-evidence.csv',['protein1','protein2','score','source_row_id'],hippie)
    bases=sorted({p.split('-')[0] for p in ids|set(seqs)})
    write_json(ROOT/'work/requested-accessions.json',dict(hippie=sorted(ids),original_test=sorted(seqs),base_accessions=bases))
    result=dict(original_test_rows=len(rows),positives=len(positive_rows),protected_proteins=len(seqs),positive_endpoint_proteins=len(positive_ids),
                original_test_sha256=sha(target),benchmark_v4_rows_sequences_labels_verified=True,hippie=source_audit,
                new_test_target_pairs=52048,new_test_target_negatives=26024,new_test_additional_proteins=0)
    write_json(ROOT/'reports/source-and-test-audit.json',result)
    frozen_names=['original-test.csv','original-sequences.fasta','original-positives.csv','protected-proteins.tsv']
    write_json(ROOT/'frozen-tests/original-manifest.json',dict(files={name:sha(ROOT/'frozen-tests'/name) for name in frozen_names},source_revision=manifest['source_revision'],audit=result))
    mark('bootstrap','complete',**result)

if __name__=='__main__':main()
