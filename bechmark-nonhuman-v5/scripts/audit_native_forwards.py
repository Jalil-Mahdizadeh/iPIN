"""Independent raw-string tokenization and official-backbone FP32 audit of frozen iPIN.

Runs on the interactive GPU, without altering production workers or their code.
Selects fixed rows by source position and sequence length, not by scores.
"""
import argparse,csv,importlib.util,json,sys,types
import numpy as np
import torch
from scipy.special import expit
from bench_utils import ROOT,PROJECT,atomic,cuda,load_npz,now,read,record,sha
import pair_infer as pair

@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--model',choices=['ipin-esm2','ipin-esmc'],required=True);a=p.parse_args()
 device=cuda();name=a.model;meta=read(ROOT/'data/sequences.json');rows=np.load(ROOT/'data/union.npy');mapping=load_npz(ROOT/'data/pair-mapping.npz')
 data=pair.test_data();ids=[];selection=[]
 for test in ['mouse','fly','worm','yeast','ecoli']:
  source=np.load(ROOT/'data'/f'{test}.npy');lengths=data.lengths[mapping[test]]
  choices=sorted({0,1,len(source)//2,int(np.argmax(lengths))})
  for i in choices:
   uid=int(mapping[test][i]);ids.append(uid);selection.append({'species':test,'source_row':i,'union_id':uid})
 ids=np.array(sorted(set(ids)),np.int64)
 saved=load_npz(ROOT/'results'/f'{name}-union.npz')['logits'][ids]
 model=pair.load_model(name,device)
 fast=np.concatenate([pair.predict(model,name,data,ids[i:i+1],device,fp32=True) for i in range(len(ids))])
 if name=='ipin-esm2':
  # Instantiate the independently qualified, unpatched original HF forward.
  spec=importlib.util.spec_from_file_location('independent_native',PROJECT/'plm-interact-reproducability/scripts/native_model.py')
  native=importlib.util.module_from_spec(spec);spec.loader.exec_module(native)
  oracle=native.NativePLM();oracle.load_state_dict(model.state_dict(),strict=True)
  oracle.eval().requires_grad_(False).to(device)
  from transformers import AutoTokenizer
  tokenizer=AutoTokenizer.from_pretrained('/opt/plm_interact/assets/esm2_650m',local_files_only=True)
  tokenizer.pad_token=tokenizer.eos_token
  def reference(x,y):
   f=native.tokenize(tokenizer,[(x,y)],max_length=None).to(device)
   return f.input_ids[0],oracle(f)[0]
 else:
  # Restore the released ESMC attention and invoke its complete official forward.
  from esm.layers.attention import MultiHeadAttention
  from esm.tokenization import EsmSequenceTokenizer
  tokenizer=EsmSequenceTokenizer()
  for block in model.esmc.transformer.blocks:
   block.attn.forward=types.MethodType(MultiHeadAttention.forward,block.attn)
  def reference(x,y):
   enc=tokenizer(x,y,add_special_tokens=True,return_tensors='pt',padding=False,truncation=False)
   tokens=enc['input_ids'].to(device)
   hidden=model.esmc(tokens,sequence_id=torch.ones_like(tokens,dtype=torch.bool)).embeddings
   z=model.classifier(torch.relu(hidden[:,0]))[0,0]
   return tokens[0],z
 native_values=[];cases=[]
 for k,uid in enumerate(ids):
  aa,bb=map(int,rows[uid,:2]);features=data.batch(np.array([uid]),np.array([0]),2,False,device);gold=[]
  for col,(x,y) in enumerate([(aa,bb),(bb,aa)]):
   tokens,z=reference(meta['sequence'][x],meta['sequence'][y])
   assert torch.equal(tokens,features['clean_ids'][col,:len(tokens)]),(name,int(uid),'tokenization')
   gold.append(float(z))
  gold=np.array(gold);le=float(abs(fast[k]-gold).max());pe=float(abs(expit(fast[k])-expit(gold)).max())
  assert le<.002 and pe<.0002,(name,int(uid),le,pe)
  cases.append({'union_id':int(uid),'tokens':int(data.lengths[uid]),'saved_bf16_logits':saved[k].tolist(),
   'fresh_fp32_optimized_logits':fast[k].tolist(),'official_fp32_logits':gold.tolist(),'optimized_official_max_logit_error':le,
   'optimized_official_max_probability_error':pe,'saved_bf16_official_max_probability_error':float(abs(expit(saved[k])-expit(gold)).max())})
  native_values.append(gold)
  print({'model':name,'row':int(uid),'official_fp32_error':le,'bf16_probability_error':cases[-1]['saved_bf16_official_max_probability_error']},flush=True)
 result={'at_utc':now(),'passed':True,'model':name,'independent_raw_string_pair_tokenization_matches':True,
  'official_backbone_fp32_forward_matches':True,'source_selection':selection,'cases':cases,
  'checkpoint':read(ROOT/'provenance/selection.json')['models'][name]['checkpoint'],'script':record(__file__),
  'test_scores_not_used_for_fixture_selection':True,'production_weights_or_workers_modified':False}
 atomic(ROOT/'qualification'/f'{name}-independent-nonhuman.json',result)
 print('Independent nonhuman forward audit passed',name,flush=True)
if __name__=='__main__':main()
