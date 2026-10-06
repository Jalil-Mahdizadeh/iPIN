"""Original SPRINT with precisely the authorized v5 TRAIN-positive graph."""
import csv
import argparse
import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path
import numpy as np
from bench_utils import ROOT,PROJECT,atomic,load_npz,now,read,record,save_npz,sha

def fasta(path):
    d={};name=None
    for line in path.read_text().splitlines():
        if line.startswith('>'):name=line[1:].split()[0];d[name]=''
        elif line.strip():d[name]+=line.strip()
    return d

def prepare():
    out=ROOT/'predictions/sprint';out.mkdir(exist_ok=True)
    data=ROOT/'data/sprint';data.mkdir(exist_ok=True)
    known=read(ROOT/'data/sequences.json');seq=list(known['sequence']);hashes=list(known['sha256'])
    index={h:i for i,h in enumerate(hashes)};test_hashes=set(hashes)
    trainseq=fasta(PROJECT/'data-preparation-v5/prepared/sequences.fasta')
    edges=[];train_ids=set()
    with (PROJECT/'data-preparation-v5/prepared/train.csv').open() as stream:
        for row in csv.DictReader(stream):
            if int(row['label'])!=1:continue
            pair=[]
            for endpoint in [row['protein1'],row['protein2']]:
                s=trainseq[endpoint];h=hashlib.sha256(s.encode()).hexdigest()
                if h not in index:index[h]=len(seq);seq.append(s);hashes.append(h)
                pair.append(index[h]);train_ids.add(index[h])
            edges.append(pair)
    assert len(edges)==350382 and len(train_ids)==13110
    assert len({tuple(sorted(pair)) for pair in edges})==len(edges)
    (data/'proteins.fasta').write_text(''.join(f'>p{i:05d}\n{s}\n' for i,s in enumerate(seq)))
    (data/'train-positive.txt').write_text(''.join(f'p{a:05d} p{b:05d}\n' for a,b in edges))
    rows=np.load(ROOT/'data/union.npy')
    (data/'pairs.txt').write_text(''.join(f'p{a:05d} p{b:05d}\n' for a,b in rows[:,:2]))
    (data/'empty.txt').touch(exist_ok=True)
    contract={'graph':'v5 TRAIN positives only','graph_edges':len(edges),'train_proteins':len(train_ids),'test_proteins':len(known['sequence']),
              'sequence_corpus':len(seq),'test_train_exact_sequence_overlap':len(train_ids.intersection(range(len(known['sequence'])))),'HSP_flags':['-Thit','15','-Tsim','35','-M','1'],
              'prediction_flags':['-Thc','40'],'scorer':'native serial predict_interactions','sequence_preprocessing':'full-length TRAIN plus TEST sequences; transductive sequence-only preprocessing',
              'files':{str(p.relative_to(ROOT)):sha(p) for p in data.iterdir()}}
    atomic(ROOT/'provenance/sprint-input.json',contract)
    print('SPRINT inputs prepared',flush=True)

def canonicalize():
    out=ROOT/'predictions/sprint';data=ROOT/'data/sprint'
    raw=out/'raw.hsp';canonical=out/'canonical.hsp'
    if not (out/'hsp-complete.json').exists():
        assert (out/'hsp-command-succeeded').exists()
        # Upstream print_HSP() deliberately inserts a full self-HSP for EVERY
        # protein, including sequences shorter than its 20-residue hit window.
        # The older benchmark's parser assumed all proteins were >=20; v5 has
        # shorter TRAIN proteins. Retain those legitimate records unchanged.
        sequences=fasta(data/'proteins.fasta');blocks={};key=None;short=0;total=0;selfs=set()
        # Store byte offsets, not every HSP tuple: bounded memory for the larger corpus.
        with raw.open('rb') as stream:
            while True:
                offset=stream.tell();line=stream.readline()
                if not line:break
                fields=line.decode().split();assert fields
                if fields[0]=='>':
                    if key is not None:blocks[key][1]=offset-blocks[key][0]
                    assert len(fields)==4 and fields[2]=='and'
                    key=(fields[1],fields[3]);assert key not in blocks
                    blocks[key]=[offset,0,0]
                else:
                    assert key is not None and len(fields)==3
                    a,b,n=map(int,fields);assert min(a,b)>=0 and n>0
                    assert a+n<=len(sequences[key[0]]) and b+n<=len(sequences[key[1]])
                    if n<20:
                        assert key[0]==key[1] and a==b==0 and n==len(sequences[key[0]])
                        short+=1
                    if key[0]==key[1] and a==b==0 and n==len(sequences[key[0]]):selfs.add(key[0])
                    blocks[key][2]+=1;total+=1
            if key is not None:blocks[key][1]=stream.tell()-blocks[key][0]
        assert blocks and all(v[2]>0 for v in blocks.values()) and selfs==set(sequences)
        temp=canonical.with_suffix('.tmp')
        with raw.open('rb') as source,temp.open('wb') as stream:
            for key,(offset,size,count) in sorted(blocks.items()):
                source.seek(offset)
                while size:
                    block=source.read(min(size,8*1024**2));assert block
                    stream.write(block);size-=len(block)
        os.replace(temp,canonical)
        census={'protein_pair_blocks':len(blocks),'hsp_records':total,
                'proteins_with_full_self_hsp':len(sequences),'native_short_full_self_hsps_preserved':short,
                'canonical_sha256':sha(canonical),'within_block_order_preserved':True}
        atomic(out/'hsp-complete.json',{'at_utc':now(),'input_sha256':sha(ROOT/'provenance/sprint-input.json'),'census':census,'raw_file':record(raw),'file':record(canonical)})
    else:
        hsp=read(out/'hsp-complete.json');assert hsp['input_sha256']==sha(ROOT/'provenance/sprint-input.json') and sha(canonical)==hsp['file']['sha256']
    print('SPRINT HSPs verified and canonicalized',flush=True)

def collect():
    out=ROOT/'predictions/sprint';data=ROOT/'data/sprint';rows=np.load(ROOT/'data/union.npy')
    assert (out/'prediction-command-succeeded').exists()
    output=out/'scores.txt'
    scores=[]
    for line in output.read_text().splitlines():
        value,label=line.split();assert label=='1';scores.append(float(value))
    scores=np.asarray(scores);assert len(scores)==len(rows) and np.isfinite(scores).all() and (scores>=0).all()
    save_npz(out/'union.npz',scores=scores)
    atomic(out/'done.json',{'at_utc':now(),'rows':len(scores),'score_kind':'native nonnegative score, not a probability',
        'implementation':read(out/'execution.json') if (out/'execution.json').exists() else {'mode':'native serial, see pipeline binary selection'},
        'graph':record(data/'train-positive.txt'),'input_contract':record(ROOT/'provenance/sprint-input.json'),'file':record(out/'union.npz'),
        'zero_scores':int((scores==0).sum()),'unique_scores':int(len(np.unique(scores))),'dummy_routing_labels_discarded':True})
    print('SPRINT complete',flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',required=True,choices=['prepare','canonical','collect'])
    stage=parser.parse_args().stage
    {'prepare':prepare,'canonical':canonicalize,'collect':collect}[stage]()
