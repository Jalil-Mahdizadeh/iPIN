"""Independent final checks; export shared physical pairs and both tokenizations."""
import csv
import hashlib
from collections import Counter
from datetime import datetime,timezone
import numpy as np
from common import ROOT,read_json,write_json,write_csv,fasta_read,fasta_write,pair,mark,sha

def main():
    mark('finalize','running')
    original_manifest=read_json(ROOT/'frozen-tests/original-manifest.json')
    for name,digest in original_manifest['files'].items():assert sha(ROOT/'frozen-tests'/name)==digest,name
    for stage in ['positive_split','boundary_verification','negative_train','negative_val','negative_test-ilp']:
        assert read_json(ROOT/'work'/f'{stage}.status.json')['status']=='complete',stage
    registry=read_json(ROOT/'work/protein-registry.json');test_meta=read_json(ROOT/'work/test-protein-metadata.json')
    original=list(csv.DictReader((ROOT/'frozen-tests/original-test.csv').open()))
    original_pos={i:r for i,r in enumerate(original) if r['label']=='1'}
    blacklist={pair(r['family1'],r['family2']) for r in csv.DictReader((ROOT/'work/known-positive-families.tsv').open(),delimiter='\t')}
    datasets={};protein_sets={};families={};sequence_sets={};summary={}
    for split in ['train','val','test-ilp']:
        meta=test_meta if split=='test-ilp' else registry
        path=ROOT/(f'frozen-tests/{split}.csv' if split=='test-ilp' else f'prepared/{split}.csv')
        rows=list(csv.DictReader(path.open()));datasets[split]=rows
        labels=Counter(int(r['label']) for r in rows);assert set(labels)=={0,1} and labels[0]==labels[1]
        keys=set();seq_keys=set();plus=Counter();minus=Counter();proteins=set()
        for r in rows:
            a,b=r['protein1'],r['protein2'];key=pair(a,b);label=int(r['label']);assert key not in keys and a!=b
            keys.add(key);sa,sb=meta[a]['sequence'],meta[b]['sequence'];assert sa!=sb
            skey=pair(sa,sb);assert skey not in seq_keys;seq_keys.add(skey);proteins.update((a,b))
            assert meta[a]['taxon']==9606 and meta[b]['taxon']==9606
            (plus if label else minus).update([a,b])
            if not label:
                assert not any(pair(f,g) in blacklist for f in meta[a]['families'] for g in meta[b]['families']),(split,a,b)
        assert all(minus[p]<=6*plus[p] for p in proteins)
        protein_sets[split]=proteins;families[split]={f for p in proteins for f in meta[p]['families']};sequence_sets[split]={meta[p]['sequence'] for p in proteins}
        lengths=np.array([len(meta[r['protein1']]['sequence'])+len(meta[r['protein2']]['sequence']) for r in rows])
        summary[split]=dict(pairs=len(rows),positives=labels[1],negatives=labels[0],unique_sequences=len(proteins),
                            combined_length_quantiles=np.quantile(lengths,[0,.5,.95,.99,1]).tolist(),pairs_over_2193=int((lengths>2193).sum()),
                            pairs_over_8192=int((lengths>8192).sum()),proteins_without_go=sum(not meta[p]['go_bp'] for p in proteins),sha256=sha(path))
    assert not protein_sets['train']&protein_sets['val']
    assert not sequence_sets['train']&sequence_sets['val']
    assert not families['train']&families['val']
    protected=set(read_json(ROOT/'work/protected-families.json'));test_sequences=set(fasta_read(ROOT/'frozen-tests/original-sequences.fasta').values())
    for split in ['train','val']:
        assert not families[split]&protected and not sequence_sets[split]&test_sequences
        expected={pair(r['protein1'],r['protein2']) for r in csv.DictReader((ROOT/'work'/f'{split}-positives.csv').open())}
        observed={pair(r['protein1'],r['protein2']) for r in datasets[split] if r['label']=='1'}
        assert observed==expected
    seen=[]
    for r in datasets['test-ilp']:
        if r['label']=='1':
            i=int(r['original_row_id']);old=original_pos[i]
            assert r['protein1']==old['Uniprot_a'] and r['protein2']==old['Uniprot_b']
            assert test_meta[r['protein1']]['sequence']==old['query'] and test_meta[r['protein2']]['sequence']==old['text']
            seen.append(i)
    assert seen==list(original_pos) and len(datasets['test-ilp'])==52048
    assert protein_sets['test-ilp']<=set(test_meta) and len(protein_sets['test-ilp'])==2948
    ids=sorted(protein_sets['train']|protein_sets['val']);index={p:i for i,p in enumerate(ids)}
    fasta_write(ROOT/'prepared/sequences.fasta',{p:registry[p]['sequence'] for p in ids})
    write_csv(ROOT/'prepared/proteins.tsv',['protein_id','members','families','length','sequence_sha256','go_bp'],
              (dict(protein_id=p,members=';'.join(registry[p]['members']),families=';'.join(registry[p]['families']),length=len(registry[p]['sequence']),sequence_sha256=hashlib.sha256(registry[p]['sequence'].encode()).hexdigest(),go_bp=';'.join(registry[p]['go_bp'])) for p in ids),delimiter='\t')
    write_csv(ROOT/'frozen-tests/test-go-annotations.tsv',['protein_id','go_bp','annotation_status'],
              (dict(protein_id=p,go_bp=';'.join(test_meta[p]['go_bp']),annotation_status=test_meta[p]['annotation_status']) for p in sorted(test_meta)),delimiter='\t')
    vocabs=read_json(ROOT/'environment/vocabularies.json')
    for name,spec in vocabs.items():
        directory=ROOT/'prepared'/name;directory.mkdir(exist_ok=True)
        offsets=[0];arrays=[]
        for p in ids:
            seq=registry[p]['sequence'];encoded=np.array([spec['vocabulary'][c] for c in seq],dtype=np.uint8)
            assert len(encoded)==len(seq);arrays.append(encoded);offsets.append(offsets[-1]+len(encoded))
        np.save(directory/'tokens.npy',np.concatenate(arrays));np.save(directory/'offsets.npy',np.array(offsets,dtype=np.int64))
        write_json(directory/'protein-order.json',ids);write_json(directory/'tokenizer-contract.json',spec)
        for split in ['train','val']:
            rows=datasets[split];a=np.array([[index[r['protein1']],index[r['protein2']],int(r['label']),i] for i,r in enumerate(rows)],dtype=np.int64)
            np.save(directory/f'{split}.npy',a)
    # Verify the two model datasets have exactly the same physical pairs and lengths.
    for name in ['offsets.npy','train.npy','val.npy']:
        assert np.array_equal(np.load(ROOT/'prepared/esm2'/name),np.load(ROOT/'prepared/esmc'/name))
    report=dict(passed=True,original_test_unchanged=True,original_positive_rows_and_sequences_preserved=True,
                train_val_shared_identifiers=0,train_val_shared_sequences=0,train_val_shared_entry_families=0,
                development_protected_test_sequence_overlap=0,development_protected_test_family_overlap=0,
                final_directed_homology_checks=read_json(ROOT/'reports/final-development-homology.json'),
                historical_exposure=read_json(ROOT/'reports/historical-exposure.json'),
                duplicate_pairs=0,new_self_pairs=0,new_negative_known_positive_conflicts=0,independent_backbone_tokenization_verified=True,splits=summary)
    write_json(ROOT/'reports/final-audit.json',report)
    files={str(p.relative_to(ROOT)):sha(p) for directory in ['prepared','frozen-tests'] for p in sorted((ROOT/directory).rglob('*')) if p.is_file()}
    manifest=dict(schema_version=1,complete=True,created_utc=datetime.now(timezone.utc).isoformat(),configuration_sha256=sha(ROOT/'configuration.json'),
                  pipeline_contract=read_json(ROOT/'provenance/execution-contract.json'),files=files,audit=report,
                  interpretation='original positives shared; ILP negatives are sampled unreported interactions; tests are dependent; no test model inference performed')
    stage_reports={k:read_json(ROOT/'reports'/f'{k}-negative-sampling.json') for k in ['train','val','test-ilp']}
    lines=['# V5 data preparation report','','**Complete: all final integrity checks passed.**','',
           'The original Bernett test is unchanged. Its 26,024 positive cases are shared with the custom ILP-negative test.','',
           '| Split | Positive pairs | Negative pairs | Unique sequences |','| --- | ---: | ---: | ---: |']
    for k,s in summary.items():lines.append(f"| {k} | {s['positives']:,} | {s['negatives']:,} | {s['unique_sequences']:,} |")
    lines+=['',f"The ILP test shares {stage_reports['test-ilp']['original_negative_overlap']:,} original negatives; {stage_reports['test-ilp']['new_negative_predictions_per_historical_model']:,} negative pairs need new scores per historical model.",'',
            'Both backbone exports use identical physical pairs and full sequences. Model retraining and benchmark inference were not run.','',
            'Solver outcomes (a time-limited feasible solution is not a proven optimum):','','| Split | Solver status | Actual relative gap | Positive GO mean | Negative GO mean |','| --- | --- | ---: | ---: | ---: |']
    for k,s in stage_reports.items():lines.append(f"| {k} | {s['status']} | {s['actual_mip_gap']:.6g} | {s['positive_go_jaccard_mean']:.6f} | {s['negative_go_jaccard_mean']:.6f} |")
    lines+=['','See [normalization exclusions](reports/normalization-audit.json), [homology exclusions](reports/test-homology-filter.json), [positive split](reports/positive-split.json), [final audit](reports/final-audit.json), and [hashed completion manifest](completed.json).','',
            'The author objective is retained. Candidate generation additionally enforces the full known-positive family blacklist and non-self scope, and removes the upstream sorted-index truncation bias. Missing GO annotations remain explicit; selected negatives are not experimentally confirmed noninteractions.']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    mark('finalize','complete',splits=summary)
    write_json(ROOT/'completed.json',manifest)

if __name__=='__main__':
    try:main()
    except Exception as exc:mark('finalize','failed',error=str(exc));raise
