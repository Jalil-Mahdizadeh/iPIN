"""Known-source supervised exposure, exact and encoder-normalized identities."""
import csv,hashlib,re
from functools import lru_cache
import numpy as np
from bench_utils import ROOT,PROJECT,EXTERNAL,atomic,now,read,record,save_npz

def fasta(path):
 out={};name=None
 for line in path.read_text().splitlines():
  if line.startswith('>'):name=line[1:].split()[0];out[name]=''
  elif line.strip():out[name]+=line.strip()
 return out

def norm(seq,kind):return re.sub('[UZOBJ]','X',seq.upper()) if kind=='ankh-normalized' else seq

def main():
 meta=read(ROOT/'data/sequences.json');pairs=np.load(ROOT/'data/union.npy');n=len(meta['sequence']);codes=[tuple(sorted(map(int,p))) for p in pairs[:,:2]];index={p:i for i,p in enumerate(codes)}
 reports={};arrays={}
 def audit(name,kind,sources,bounds=None):
  table={}
  for i,s in enumerate(meta['sequence']):table.setdefault(hashlib.sha256(norm(s,kind).encode()).hexdigest(),[]).append(i)
  hit=np.zeros(len(pairs),np.uint8);seen=np.zeros(n,np.uint8);records=[]
  for split,seqfile,pairfile,format_,bit in sources:
   seq={} if seqfile is None else dict(line.rstrip().split('\t',1) for line in seqfile.open() if line.strip()) if seqfile.suffix=='.tsv' else fasta(seqfile)
   mapping={a:table.get(hashlib.sha256(norm(s,kind).encode()).hexdigest(),[]) for a,s in seq.items()}
   @lru_cache(None)
   def match_sequence(s):return table.get(hashlib.sha256(norm(s,kind).encode()).hexdigest(),[])
   missing=set();total=0;matched=0;positives=0;filtered=0
   with pairfile.open() as f:
    reader=csv.reader(f,delimiter='\t') if format_=='tuna' else csv.DictReader(f,delimiter='\t' if format_.startswith('xpair') else ',')
    for row in reader:
     total+=1
     if format_=='sequence-csv':aa,bb=row['query'],row['text'];label=int(row['label'])
     elif format_=='tuna':aa,bb=row[:2];label=int(row[2])
     elif format_.startswith('xpair'):aa,bb=row['id1'],row['id2'];label=int(float(row['interaction_labels'])) if format_=='xpair-interaction' else 1
     else:aa,bb=row['protein1'],row['protein2'];label=int(row['label'])
     positives+=label
     if name=='sprint' and label!=1:filtered+=1;continue
     if format_!='sequence-csv':
      if aa not in seq:missing.add(aa)
      if bb not in seq:missing.add(bb)
     if bounds is not None and (aa not in seq or bb not in seq or not all(bounds[0]<=len(seq[k])<=bounds[1] for k in [aa,bb])):
      filtered+=1;continue
     if format_=='sequence-csv':left=match_sequence(aa);right=match_sequence(bb)
     else:left=mapping.get(aa,[]);right=mapping.get(bb,[])
     for a in left:seen[a]|=bit
     for b in right:seen[b]|=bit
     for a in left:
      for b in right:
       j=index.get(tuple(sorted((a,b))))
       if j is not None:hit[j]|=bit;matched+=1
   records.append({'split':split,'sequence_file':None if seqfile is None else record(seqfile),'pairs_file':record(pairfile),'source_rows':total,'positive_rows':positives,
     'missing_source_fasta_ids':len(missing),'test_pair_matches':matched,'excluded_by_length':filtered})
  out={'sequence_matching':kind,'source_sequence_length_bounds':bounds,'sources':records,'test_endpoints_exposed':int((seen>0).sum()),'union_pairs_exposed':int((hit>0).sum()),'tests':{}}
  maps=np.load(ROOT/'data/pair-mapping.npz')
  for test in ['mouse','fly','worm','yeast','ecoli']:
   ids=maps[test];y=np.load(ROOT/'data'/f'{test}.npy')[:,2]
   out['tests'][test]={'pairs_exposed':int((hit[ids]>0).sum()),'positives_exposed':int(((hit[ids]>0)&(y==1)).sum()),'negatives_exposed':int(((hit[ids]>0)&(y==0)).sum()),
    'pairs_with_at_least_one_exposed_endpoint':int(((seen[pairs[ids,0]]>0)|(seen[pairs[ids,1]]>0)).sum())}
  arrays[name+'__'+kind+'__pairs']=hit;arrays[name+'__'+kind+'__endpoints']=seen;reports[name+'__'+kind]=out
 # TRAIN and DEV audits for new models. SPRINT sees only TRAIN labels.
 prep=PROJECT/'data-preparation-v5/prepared'
 audit('v5-ipin','exact',[(part,prep/'sequences.fasta',prep/f'{part}.csv','v5',bit) for part,bit in [('train',1),('val',2)]])
 audit('sprint','exact',[('train-positive',prep/'sequences.fasta',prep/'train.csv','v5',1)])
 for name,folder,pattern in [('native-human',ROOT/'data/exposure','human.ppi.qrels.seq.{split}.csv'),('native-plm-public-source',PROJECT/'retrain-v1/data/raw','pairs_uniprot_seqs_{split}.csv')]:
  sources=[]
  for part,bit in [('train',1),('val',2)]:
   file=folder/('human.ppi.qrels.seq.train.csv' if part=='train' else 'human.ppi.qrels.seq.test.csv') if name=='native-human' else folder/pattern.format(split=part)
   sources.append((part,None,file,'sequence-csv',bit))
  audit(name,'exact',sources)
 tuna=EXTERNAL/'benchmark/tuna/upstream/TUnA/data'
 audit('tuna','exact',[(part,tuna/'raw/bernett/human_swissprot_oneliner.fasta',tuna/'processed/bernett'/f'{dataset}_interaction_1500_or_less.tsv','tuna',bit) for part,dataset,bit in [('train','Intra1',1),('val','Intra0',2)]])
 audit('tuna-human','exact',[(part,tuna/'processed/xspecies'/f'human_{part}_dictionary.tsv',tuna/'processed/xspecies'/f'human_{part}_interaction.tsv','tuna',bit) for part,bit in [('train',1),('test',2)]])
 ds=EXTERNAL/'benchmark/dscript/upstream/original-exposure'
 audit('dscript','exact',[('public-human-train',ds/'human.fasta',ds/'human_train.tsv','tuna',1)])
 x=EXTERNAL/'experiments/x_pair_test2_v1/sources/datasets/xpair_datasets/X-fair'
 source=[(f'{task}-{split}',x/task/'sequences.fasta',x/task/f'{task}_{split}.tsv','xpair-'+task,bit) for task,split,bit in [('interaction','train',1),('interaction','val',2),('interface','train',4),('interface','val',8)]]
 for kind in ['exact','ankh-normalized']:audit('xpair-default',kind,source)
 audit('xpair-default-length-eligible','exact',source,bounds=[50,2000])
 save_npz(ROOT/'provenance/exposure-flags.npz',**arrays)
 atomic(ROOT/'provenance/exposure.json',{'at_utc':now(),'audits':reports,'flags':record(ROOT/'provenance/exposure-flags.npz'),
  'native_plm_historical_audit':record(PROJECT/'data-preparation-v5/reports/historical-exposure.json'),
  'unknown':['RAPPPID exact released-checkpoint training membership unavailable in bundled assets','X-PAIR Bernett exact processed training files not available locally; nominal Bernett source is not proof of exact checkpoint membership'],
  'limits':'Known public train/validation files, not cryptographic proof of complete checkpoint history. No guarantee about homologs, fragments or PLM pretraining. v5 human test homology protection does not extend to these nonhuman tests. Native Bernett raw train/val audit is conservative because historical length filtering may remove rows; TUnA human-test is its human validation source, not a nonhuman test.'})
 print({k:{'endpoints':v['test_endpoints_exposed'],'pairs':v['union_pairs_exposed'],'tests':v['tests']} for k,v in reports.items()},flush=True)
if __name__=='__main__':main()
