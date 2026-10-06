"""Resolve identifiers, freeze annotations, and exclude test families/identical sequences."""
import csv
import gzip
import hashlib
import io
import json
from collections import Counter,defaultdict
from Bio import SeqIO
from common import ROOT, PROJECT, read_json, write_json, write_csv, fasta_read, fasta_write, pair, mark, sha

def compact_entry(p):
    go=set()
    for x in p.get('uniProtKBCrossReferences',[]):
        if x['database']=='GO' and any(v['key']=='GoTerm' and v['value'].startswith('P:') for v in x.get('properties',[])):
            go.add(x['id'])
    displayed=[]
    for c in p.get('comments',[]):
        if c.get('commentType')=='ALTERNATIVE PRODUCTS':
            for iso in c.get('isoforms',[]):
                if iso.get('sequenceStatus')=='Displayed':displayed.extend(iso['isoformIds'])
    return dict(primary=p['primaryAccession'],secondary=sorted(p.get('secondaryAccessions',[])),taxon=int(p['organism']['taxonId']),
                sequence=p['sequence']['value'],go_bp=sorted(go),displayed_isoforms=sorted(displayed))

def main():
    mark('normalize','running')
    assert read_json(ROOT/'work/uniprot.status.json')['status']=='complete'
    entries={};fastas={}
    for receipt in read_json(ROOT/'provenance/uniprot-downloads.json'):
        path=ROOT/receipt['path'];assert sha(path)==receipt['sha256']
        with gzip.open(path,'rt') as f:
            if receipt['format']=='json':
                for raw in json.load(f)['results']:
                    p=compact_entry(raw);key=p['primary']
                    assert key not in entries or entries[key]==p,key
                    entries[key]=p
            else:
                for record in SeqIO.parse(f,'fasta'):
                    name=record.id.split('|')[1];seq=str(record.seq)
                    assert name not in fastas or fastas[name]==seq,name
                    fastas[name]=seq
    aliases=defaultdict(set)
    for primary,p in entries.items():
        for alias in [primary,*p['secondary']]:aliases[alias].add(primary)
        if primary in fastas:assert p['sequence']==fastas[primary]
        for iso in p['displayed_isoforms']:
            assert iso not in fastas or fastas[iso]==p['sequence'],iso
            fastas[iso]=p['sequence']
    requested=read_json(ROOT/'work/requested-accessions.json')
    test=fasta_read(ROOT/'frozen-tests/original-sequences.fasta')
    alphabet={x for x in (PROJECT/'retrain-v1/assets/esm2/vocab.txt').read_text().splitlines() if len(x)==1}
    def families(p):
        base=p.split('-')[0]
        return sorted(aliases.get(base,{base}))
    protected_families={f for p in test for f in families(p)}
    raw_records={};invalid=[]
    for accession in requested['hippie']:
        base=accession.split('-')[0];matches=aliases.get(base,set())
        # Exact primary IDs are authoritative; a split obsolete accession is ambiguous.
        primary=base if base in entries else next(iter(matches)) if len(matches)==1 else None
        if primary is None:
            invalid.append(dict(protein_id=accession,reason='ambiguous_accession' if matches else 'unresolved_accession'));continue
        entry=entries[primary]
        if entry['taxon']!=9606:
            invalid.append(dict(protein_id=accession,reason='not_human'));continue
        seq=fastas.get(accession) if '-' in accession else entry['sequence']
        if not seq:
            invalid.append(dict(protein_id=accession,reason='unresolved_isoform_sequence'));continue
        if not set(seq)<=alphabet:
            invalid.append(dict(protein_id=accession,reason='unsupported_sequence_alphabet'));continue
        raw_records[accession]=dict(sequence=seq,primary=primary,families=families(accession),go_bp=entry['go_bp'])
    by_sequence=defaultdict(list)
    for p,r in raw_records.items():by_sequence[r['sequence']].append(p)
    registry={};raw_to_entity={};excluded=[]
    test_seqs=set(test.values())
    for i,(seq,members) in enumerate(sorted(by_sequence.items(),key=lambda x:hashlib.sha256(x[0].encode()).hexdigest())):
        key=f'h{i:07d}';members=sorted(members)
        fs=sorted({f for p in members for f in raw_records[p]['families']})
        go=sorted({g for p in members for g in raw_records[p]['go_bp']})
        for p in members:raw_to_entity[p]=key
        reasons=[]
        if set(members)&set(test):reasons.append('test_accession')
        if set(fs)&protected_families:reasons.append('test_entry_or_isoform_family')
        if seq in test_seqs:reasons.append('exact_test_sequence')
        registry[key]=dict(sequence=seq,members=members,families=fs,go_bp=go,taxon=9606,initial_exclusion=reasons)
        if reasons:excluded.append(dict(protein_id=key,reason=';'.join(reasons),members=';'.join(members)))
    test_meta={};test_annotation_status=Counter();changed_test_sequences=0
    for p,seq in test.items():
        fs=families(p);matches=[entries[f] for f in fs if f in entries]
        if len(matches)==1:
            assert matches[0]['taxon']==9606,(p,matches[0]['taxon'])
            go=matches[0]['go_bp'];status='resolved_parent_annotation'
            changed_test_sequences+=matches[0]['sequence']!=seq
        else:
            go=[];status='ambiguous_annotation' if matches else 'missing_annotation'
        test_annotation_status[status]+=1
        test_meta[p]=dict(sequence=seq,families=fs,go_bp=go,annotation_status=status,taxon=9606)
    known_families=set();positive_edges={};losses=Counter();evidence_rows=0
    raw_known=set()
    loss_path=ROOT/'work/positive-row-exclusions-before-homology.csv.gz'
    with (ROOT/'work/hippie-positive-evidence.csv').open() as f,gzip.open(loss_path,'wt',newline='') as out:
        writer=csv.writer(out);writer.writerow(['source_row_id','protein1','protein2','reason'])
        for r in csv.DictReader(f):
            evidence_rows+=1;p,q=r['protein1'],r['protein2'];raw_known.add(pair(p,q))
            for a in families(p):
                for b in families(q):known_families.add(pair(a,b))
            a,b=raw_to_entity.get(p),raw_to_entity.get(q)
            reason=None
            if a is None or b is None:reason='unresolved_or_ineligible_endpoint'
            elif a==b:reason='self_or_identical_sequence'
            elif registry[a]['initial_exclusion'] or registry[b]['initial_exclusion']:reason='protected_test_endpoint'
            if reason:
                losses[reason]+=1;writer.writerow([r['source_row_id'],p,q,reason]);continue
            key=pair(a,b)
            if key in positive_edges:
                losses['duplicate_sequence_pair']+=1;writer.writerow([r['source_row_id'],p,q,'duplicate_sequence_pair'])
                positive_edges[key]['evidence_rows']+=1
                positive_edges[key]['score']=max(positive_edges[key]['score'],float(r['score']))
            else:positive_edges[key]=dict(protein1=key[0],protein2=key[1],score=float(r['score']),evidence_rows=1,first_source_row_id=int(r['source_row_id']))
    # All available historical positive evidence remains ineligible for new negatives.
    # Retiring a validation partition or updating HIPPIE does not establish noninteraction.
    historical_additions=[]
    for split in ['train','val']:
        path=PROJECT/'retrain-v1/data/raw'/f'pairs_uniprot_seqs_{split}.csv'
        for row_id,r in enumerate(csv.DictReader(path.open())):
            if r['label']!='1':continue
            for a in families(r['Uniprot_a']):
                for b in families(r['Uniprot_b']):
                    key=pair(a,b)
                    if key not in known_families:
                        historical_additions.append(dict(source_split=split,source_row_id=row_id,family1=key[0],family2=key[1]))
                        known_families.add(key)
    write_csv(ROOT/'reports/historical-positive-blacklist-additions.csv',['source_split','source_row_id','family1','family2'],historical_additions)
    pos=list(csv.DictReader((ROOT/'frozen-tests/original-positives.csv').open()))
    for r in pos:
        for a in test_meta[r['protein1']]['families']:
            for b in test_meta[r['protein2']]['families']:known_families.add(pair(a,b))
    conflict_counts=Counter();conflicts=[]
    for idx,r in enumerate(csv.DictReader((ROOT/'frozen-tests/original-test.csv').open())):
        if r['label']!='0':continue
        p,q=r['Uniprot_a'],r['Uniprot_b']
        exact=pair(p,q) in raw_known
        family=any(pair(a,b) in known_families for a in test_meta[p]['families'] for b in test_meta[q]['families'])
        if exact or family:
            conflicts.append(dict(original_row_id=idx,protein1=p,protein2=q,exact_accession_positive=int(exact),conservative_family_positive=int(family)))
            conflict_counts['exact_accession_positive']+=exact;conflict_counts['conservative_family_positive']+=family
    write_csv(ROOT/'reports/original-negative-evidence-conflicts.csv',['original_row_id','protein1','protein2','exact_accession_positive','conservative_family_positive'],conflicts)
    write_json(ROOT/'work/protein-registry.json',registry)
    write_json(ROOT/'work/raw-accession-to-entity.json',raw_to_entity)
    write_json(ROOT/'work/test-protein-metadata.json',test_meta)
    write_json(ROOT/'work/protected-families.json',sorted(protected_families))
    write_csv(ROOT/'work/known-positive-families.tsv',['family1','family2'],(dict(family1=a,family2=b) for a,b in sorted(known_families)),delimiter='\t')
    write_csv(ROOT/'reports/unresolved-or-ineligible-proteins.csv',['protein_id','reason'],invalid)
    write_csv(ROOT/'reports/protein-exclusions-before-homology.csv',['protein_id','reason','members'],excluded)
    write_csv(ROOT/'work/eligible-before-homology.csv',['protein1','protein2','score','evidence_rows','first_source_row_id'],(positive_edges[k] for k in sorted(positive_edges)))
    active={p for a,b in positive_edges for p in (a,b)}
    fasta_write(ROOT/'work/eligible-before-homology.fasta',{p:registry[p]['sequence'] for p in active})
    result=dict(uniprot_release='2026_03',canonical_entries=len(entries),fasta_sequences_including_displayed_isoforms=len(fastas),
                resolved_hippie_accessions=len(raw_records),invalid_accessions=len(invalid),invalid_reasons=dict(Counter(r['reason'] for r in invalid)),
                distinct_hippie_sequences=len(registry),exact_sequence_aliases_collapsed=len(raw_records)-len(registry),
                protected_entry_families=len(protected_families),initially_excluded_sequence_entities=len(excluded),
                raw_positive_rows=evidence_rows,positive_row_exclusions=dict(losses),eligible_positive_pairs_before_homology=len(positive_edges),
                active_sequence_entities_before_homology=len(active),test_annotation_status=dict(test_annotation_status),
                test_sequences_differing_from_current_parent=changed_test_sequences,test_sequences_preserved=True,
                original_negative_evidence_conflicts=dict(conflict_counts),known_positive_family_pairs=len(known_families),
                historical_development_positive_family_additions=len(historical_additions),
                go_bp_coverage_active=sum(bool(registry[p]['go_bp']) for p in active)/len(active),
                blacklist_policy='conservative UniProt-family projection of all HIPPIE evidence plus historical train/validation/test positives; prevents isoform/alias contradictions')
    assert sum(losses.values())+len(positive_edges)==evidence_rows
    write_json(ROOT/'reports/normalization-audit.json',result)
    mark('normalize','complete',**result)

if __name__=='__main__':
    try:main()
    except Exception as exc:mark('normalize','failed',error=str(exc));raise
