"""Verify frozen tests; select checkpoints using DEV only; prepare exact pair reuse."""
import csv
import hashlib
import os
import shutil
import numpy as np
from sklearn.metrics import average_precision_score,precision_recall_curve
from bench_utils import ROOT,PROJECT,EXTERNAL,atomic,load_npz,now,read,record,save_npz,sha

def fasta(path):
    result={};name=None
    for line in path.read_text().splitlines():
        if line.startswith('>'):name=line[1:].split()[0];assert name not in result;result[name]=''
        elif line.strip():result[name]+=line.strip()
    return result

def threshold(y,z):
    precision,recall,levels=precision_recall_curve(y,z)
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-30)
    i=np.flatnonzero(f1==f1.max())[-1]
    return {'logit':float(levels[i]),'validation_f1':float(f1[i]),'rows':len(y),'selection':'maximum DEV F1, highest threshold on ties'}

def link(source,target):
    if not target.exists():os.link(source,target)
    assert source.stat().st_size==target.stat().st_size

def main():
    assert not (ROOT/'provenance/prepared.json').exists()
    prep=PROJECT/'data-preparation-v5';completion=read(prep/'completed.json')
    assert completion['complete']
    for name,digest in completion['files'].items():assert sha(prep/name)==digest,name
    trainroot=PROJECT/'retrain-v5'
    selected={'at_utc':now(),'criterion':'maximum full ILP DEV pooled AP; earlier ties retained','training_current_path':str(trainroot),'training_original_path':'retraining-v5','models':{}}
    for backbone,expected in [('esm2',27374),('esmc',21899)]:
        name='ipin-'+backbone;run=trainroot/'runs'/f'{backbone}-native-ilp-seed2'
        best=read(run/'best.json');contract=read(run/'contract.json')
        y=np.load(trainroot/'data/prepared'/backbone/'val.npy')[:,2]
        entries=[]
        for meta_path in sorted((run/'validation').glob('*.json')):
            meta=read(meta_path);path=meta_path.with_suffix('.npz')
            assert sha(path)==meta['sha256'] and meta['fingerprint']==contract['fingerprint']
            arr=load_npz(path)['predictions']
            assert arr.shape==(165742,4) and np.array_equal(arr[:,0],np.arange(len(y))) and np.array_equal(arr[:,1],y)
            ap=float(average_precision_score(y,arr[:,2:4].mean(1)))
            assert abs(ap-meta['metrics']['pooled_ap'])<1e-12
            entries.append({'update':meta['update'],'ap':ap,'sha256':meta['sha256']})
        winner=max(entries,key=lambda item:(item['ap'],-item['update']))
        assert winner['update']==best['update']==expected and winner['ap']==best['best_ap']
        path=run/'checkpoints'/best['file'];assert sha(path)==best['sha256']
        target=ROOT/'checkpoints'/(name+'.pt');link(path,target)
        vp=run/'validation'/f'update-{expected:09d}.npz';shutil.copyfile(vp,ROOT/'data'/(name+'-selected-dev.npz'))
        arr=load_npz(vp)['predictions']
        selected['models'][name]={'checkpoint':record(target),'source_checkpoint':str(path),'update':expected,'source_manifest':best,
            'configuration':contract['configuration'],'base':str((trainroot/'assets'/backbone).resolve()),'contract':contract,
            'validation_audit':entries,'validation_ap':winner['ap'],'operating_point':threshold(arr[:,1],arr[:,2:4].mean(1))}
    selected['preferred_ipin_before_test']='ipin-esm2'
    native_source=PROJECT/'benchmark-v1/checkpoints/native-bernett.bin'
    native_old=read(PROJECT/'benchmark-v1/provenance/selection.json')['models']['native-bernett']
    assert sha(native_source)==native_old['sha256']
    link(native_source,ROOT/'checkpoints/native-plm.bin')
    selected['models']['native-plm']={'checkpoint':record(ROOT/'checkpoints/native-plm.bin'),'release':native_old,'base':str((trainroot/'assets/esm2').resolve())}
    # Frozen original test sequences, not updated UniProt records.
    by_accession=fasta(prep/'frozen-tests/original-sequences.fasta')
    digest_by_id={a:hashlib.sha256(s.encode()).hexdigest() for a,s in by_accession.items()}
    hashes=sorted(set(digest_by_id.values()));seq_by_hash={digest_by_id[a]:s for a,s in by_accession.items()}
    sequence=[seq_by_hash[h] for h in hashes];index={h:i for i,h in enumerate(hashes)}
    ids={a:index[h] for a,h in digest_by_id.items()}
    assert len(ids)==len(hashes)==3022
    originals=[]
    with (prep/'frozen-tests/original-test.csv').open() as stream:
        for i,row in enumerate(csv.DictReader(stream)):
            a,b=row['Uniprot_a'],row['Uniprot_b']
            assert ''.join(row['query'].split())==by_accession[a] and ''.join(row['text'].split())==by_accession[b]
            originals.append((ids[a],ids[b],int(row['label']),i))
    ilp=[]
    with (prep/'frozen-tests/test-ilp.csv').open() as stream:
        for i,row in enumerate(csv.DictReader(stream)):ilp.append((ids[row['protein1']],ids[row['protein2']],int(row['label']),i))
    originals=np.asarray(originals,np.int64);ilp=np.asarray(ilp,np.int64)
    assert originals.shape==ilp.shape==(52048,4)
    assert int(originals[:,2].sum())==int(ilp[:,2].sum())==26024
    sets=[]
    for rows in [originals,ilp]:
        pairs={tuple(sorted(row[:2])):int(row[2]) for row in rows};assert len(pairs)==52048
        sets.append(pairs)
    assert {k for k,v in sets[0].items() if v}=={k for k,v in sets[1].items() if v}
    shared=sets[0].keys()&sets[1].keys();assert len(shared)==27178
    assert all(sets[0][k]==sets[1][k] for k in shared)
    union=originals.tolist();lookup={tuple(sorted(row[:2])):i for i,row in enumerate(union)}
    ilp_indices=[];ilp_reversed=[]
    for row in ilp:
        key=tuple(sorted(row[:2]))
        if key not in lookup:
            lookup[key]=len(union);union.append([int(row[0]),int(row[1]),int(row[2]),len(union)])
        j=lookup[key];ilp_indices.append(j);ilp_reversed.append(tuple(row[:2])!=tuple(union[j][:2]))
    union=np.asarray(union,np.int64);assert union.shape==(76918,4)
    vocab=read(trainroot/'data/prepared/esm2/tokenizer-contract.json')['vocabulary']
    vocab_c=read(trainroot/'data/prepared/esmc/tokenizer-contract.json')['vocabulary']
    alphabet=set(''.join(sequence));assert all(vocab[a]==vocab_c[a] for a in alphabet)
    tokens=np.concatenate([np.array([vocab[a] for a in s],np.int16) for s in sequence])
    offsets=np.concatenate(([0],np.cumsum([len(s) for s in sequence],dtype=np.int64)))
    for name,arr in [('original',originals),('ilp',ilp),('union',union),('tokens',tokens),('offsets',offsets)]:np.save(ROOT/'data'/(name+'.npy'),arr)
    save_npz(ROOT/'data/pair-mapping.npz',original=np.arange(52048),ilp=np.array(ilp_indices),ilp_reversed=np.array(ilp_reversed))
    atomic(ROOT/'data/sequences.json',{'sha256':hashes,'sequence':sequence,'length':list(map(len,sequence)),
                                     'accessions':[[a for a,i in ids.items() if i==j] for j in range(len(sequence))]})
    shutil.copyfile(prep/'frozen-tests/original-sequences.fasta',ROOT/'data/test-sequences.fasta')
    # Verify and reuse the exact original Bernett predictions, including orientations.
    old=PROJECT/'benchmark-v4';manifest={line.split(maxsplit=1)[1].strip():line.split(maxsplit=1)[0] for line in (old/'provenance/artifact-sha256.txt').read_text().splitlines()}
    reuse=read(old/'provenance/reuse-verification.json')
    assert sha(old/'provenance/reuse-verification.json')==manifest[str(old/'provenance/reuse-verification.json')]
    for rel in ['data/test.npy','data/tokens.npy','data/offsets.npy','results/native-bernett-test.npz','results/native-bernett-val.npz']:
        path=old/rel;expected=manifest[str(path)] if str(path) in manifest else reuse['files'][rel]['sha256']
        assert sha(path)==expected,rel
    old_rows=np.load(old/'data/test.npy');old_tokens=np.load(old/'data/tokens.npy');old_offsets=np.load(old/'data/offsets.npy')
    old_to_new={}
    for row,new_row in zip(old_rows,originals):
        assert row[2]==new_row[2]
        for old_id,new_id in zip(row[:2],new_row[:2]):
            if int(old_id) not in old_to_new:
                assert np.array_equal(old_tokens[old_offsets[old_id]:old_offsets[old_id+1]],tokens[offsets[new_id]:offsets[new_id+1]])
                old_to_new[int(old_id)]=int(new_id)
            assert old_to_new[int(old_id)]==new_id
    inherited=load_npz(old/'results/native-bernett-test.npz')['predictions']
    assert inherited.shape==(52048,4) and np.array_equal(inherited[:,0],np.arange(52048)) and np.array_equal(inherited[:,1],originals[:,2])
    save_npz(ROOT/'data/native-original-reused.npz',predictions=inherited)
    vd=load_npz(old/'results/native-bernett-val.npz')['predictions']
    selected['models']['native-plm']['operating_point']=threshold(vd[:,1],vd[:,2:4].mean(1))
    selected['native_reuse']={'source':record(old/'results/native-bernett-test.npz'),'verified_sequence_pairs':52048,'new_union_pairs':24870}
    # Cache reuse is by exact sequence hash, never accession alone.
    cache_audits={}
    for model,source in [('tuna',EXTERNAL/'benchmark/tuna/data/sequences.json'),('xpair',EXTERNAL/'experiments/x_pair_test2_v1/data/sequences.json')]:
        old_seq=read(source);old_lookup={h:i for i,h in enumerate(old_seq['sha256'])}
        mappings=[old_lookup.get(h,-1) for h in hashes]
        if model=='xpair':
            feature_dir=EXTERNAL/'experiments/x_pair_test2_v1/features'
            mappings=[j if j>=0 and (feature_dir/f'{j:05d}.json').exists() and (feature_dir/f'{j:05d}.pt').exists() else -1 for j in mappings]
        np.save(ROOT/'data'/(model+'-cache-indices.npy'),np.array(mappings,np.int64))
        cache_audits[model]={'sequence_catalog':record(source),'matching_cached_sequences':sum(j>=0 for j in mappings),'missing_sequences':sum(j<0 for j in mappings)}
    atomic(ROOT/'provenance/selection.json',selected)
    lengths=np.diff(offsets);report={'prepared_at_utc':now(),'source_completion':record(prep/'completed.json'),'tests':{'original':52048,'ilp':52048},
        'positives_per_test':26024,'shared_positive_pairs':26024,'shared_negative_pairs':1154,'union_pairs':76918,'proteins':3022,
        'max_protein_length':int(lengths.max()),'max_combined_residues':int((lengths[union[:,0]]+lengths[union[:,1]]).max()),
        'cache_candidates':cache_audits,'preferred_ipin_before_test':'ipin-esm2','data_files':{str(p.relative_to(ROOT)):sha(p) for p in sorted((ROOT/'data').iterdir())},
        'selection_sha256':sha(ROOT/'provenance/selection.json'),'historical_training_folder_renamed_to':'retrain-v5'}
    atomic(ROOT/'provenance/prepared.json',report)
    print(report)

if __name__=='__main__':main()
