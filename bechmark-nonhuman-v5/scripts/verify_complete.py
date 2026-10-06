"""Independent coverage/metric/input checks before sealing the five-species report."""
import csv,gzip,hashlib,json
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from bench_utils import ROOT,atomic,load_npz,now,read,record,sha
from collect import NAMES,TESTS,PAIR_MODELS,verify

def main():
 c=read(ROOT/'results/collection.json');s=read(ROOT/'results/summary.json')
 assert c['complete'] and s['complete'] and set(c['models'])==set(NAMES)==set(s['names'])
 assert set(s['tests'])==set(TESTS) and sum(s['tests'][t]['counts']['rows'] for t in TESTS)==242000
 verify(s['collection']);verify(s['protocol']);verify(s['analysis_script'])
 inputs=read(ROOT/'provenance/prepared.json')
 # Preserve exact source CSVs and all prepared inference/mapping arrays.
 for name,item in inputs['source_files'].items():verify(item)
 for path,h in inputs['data_files'].items():assert sha(ROOT/path)==h,path
 import pair_infer
 for name in ['ipin-esm2','ipin-esmc','native-plm']:
  q=read(ROOT/'qualification'/f'{name}.json');assert q['passed'] and q['fingerprint']==pair_infer.signature(name)
 import native_human_fp32
 q=read(ROOT/'qualification/native-human.json');assert q['passed'] and q['fingerprint']==native_human_fp32.signature('native-human')
 for qname in ['tuna','xpair','dscript','rapppid']:
  q=read(ROOT/'qualification'/f'{qname}.json');assert q['passed']
  if 'signature' in q:
   for path,h in q['signature'].items():assert sha(ROOT/path)==h,path
 resume_path=ROOT/'provenance/xpair-default-resume.json'
 if resume_path.exists():
  resume=read(resume_path)
  assert resume['state']=='complete' and resume['completed_logical_shards']==4
  verify(resume['wrapper']);verify(resume['original_qualification'])
  assert resume['native_signature']==read(ROOT/'qualification/xpair.json')['signature']
  assert not resume['new_encoder_inference'] and not resume['changed_model_or_batching']
 # Validate archived source rows independently, not just union score arrays.
 mapping=load_npz(ROOT/'data/pair-mapping.npz');meta=read(ROOT/'data/sequences.json');checked=[]
 for test in TESTS:
  rows=np.load(ROOT/'data'/f'{test}.npy');ids=mapping[test];path=ROOT/'results'/f'{test}-predictions.csv.gz'
  source=inputs['source_files'][test]['path']
  with open(source) as f:original=list(csv.DictReader(f))
  assert len(original)==len(rows)
  for row,(a,b,label,i) in zip(original,rows):
   assert row['query']==meta['sequence'][a] and row['text']==meta['sequence'][b] and int(row['label'])==label
  with gzip.open(path,'rt') as stream:out=list(csv.DictReader(stream))
  assert len(out)==len(rows)
  assert np.array_equal([int(r['source_row_id']) for r in out],rows[:,3])
  assert np.array_equal([int(r['union_id']) for r in out],ids)
  assert np.array_equal([int(r['label']) for r in out],rows[:,2])
  assert all(r['sequence_a_sha256']==meta['sha256'][a] and r['sequence_b_sha256']==meta['sha256'][b] for r,(a,b,*_) in zip(out,rows))
  for name in NAMES:
   union=load_npz(verify(c['models'][name]['file']));pred=np.array([float(r[name+'_score']) for r in out])
   assert np.array_equal(pred,union['scores'][ids]) and np.isfinite(pred).all()
   ap=average_precision_score(rows[:,2],pred);auc=roc_auc_score(rows[:,2],pred)
   expected=s['tests'][test]['models'][name];assert abs(ap-expected['ap'])<1e-12 and abs(auc-expected['auroc'])<1e-12
  checked.append({'test':test,'rows':len(rows),'all_model_metrics_recomputed_from_export':True})
 # Native human FP32 runs reproduce original-runtime original-order predictions.
 for item in read(ROOT/'results/paper-convention-native-reference.json'):
  verify(item['source']);assert item['fresh_original_order_max_logit_error']<.01 and item['fresh_original_order_max_probability_error']<.001,item
 q=read(ROOT/'qualification/sprint-requested/qualification.json');assert q['passed'] and q['native_output_identical_bytes'] and q['self_training_identical_bytes']
 ex=read(ROOT/'predictions/sprint/execution.json');verify(ex['output']);verify(ex['binary']);verify(ex['qualification']);verify(ex['input_contract'])
 recovery_path=ROOT/'provenance/sprint-copy-recovery.json'
 if recovery_path.exists():
  recovery=read(recovery_path)
  for key in ['raw_hsp','previous_wrapper','new_wrapper','copy_helper','copy_qualification','input_contract','unchanged_scorer']:verify(recovery[key])
  assert not recovery['hsp_calculation_repeated'] and not recovery['prediction_started_in_cancelled_job']
  assert recovery['input_validation_and_prediction_ast_unchanged']
  assert ex['binary']==recovery['unchanged_scorer'] and ex['script']==recovery['new_wrapper']
  from sprint_hsp_mmap import verify_qualification,verify_previous_prefix
  verify_qualification(ROOT)
  hsp=read(ROOT/'predictions/sprint/hsp-complete.json')
  verify(hsp['file']);assert hsp['raw_file']==recovery['raw_hsp'] and hsp['copy_qualification']==recovery['copy_qualification']
  verify_previous_prefix(ROOT,ROOT/'predictions/sprint/canonical.hsp')
 for name,entry in read(ROOT/'provenance/selection.json')['models'].items():verify(entry['checkpoint'])
 for name,item in read(ROOT/'provenance/runtime-inputs.json')['items'].items():verify(item)
 ds=read(ROOT/'provenance/dscript-runtime.json')
 # D-SCRIPT's pinned runtime has an explicit image record in its own manifest.
 if isinstance(ds.get('image'),dict):verify(ds['image'])
 figure_models=NAMES
 extension_path=ROOT/'provenance/historical-figure-extension.json'
 if extension_path.exists():
  extension=read(extension_path)
  assert extension['complete'] and all(extension['checks'].values())
  assert not extension['inference_repeated'] and not extension['bootstrap_recomputed']
  assert extension['models']==NAMES+['v2-capped','v2-clean-bce'] and len(extension['figure_files'])==11
  for item in extension['sources']+extension['figure_files']:verify(item)
  for key in ['metrics','intervals','script','plot_code','previous_figures']:verify(extension[key])
  with verify(extension['metrics']).open() as stream:figure_rows=list(csv.DictReader(stream))
  assert len(figure_rows)==195 and {r['model'] for r in figure_rows}==set(extension['models'])
  assert len({(r['test'],r['model'],r['subset']) for r in figure_rows})==195
  with verify(extension['intervals']).open() as stream:figure_intervals=list(csv.DictReader(stream))
  assert len(figure_intervals)==130
  figure_models=extension['models']
 artifacts=[]
 for folder in ['results','qualification','provenance','scripts','slurm']:
  for path in sorted((ROOT/folder).rglob('*')):
   if not path.is_file() or '__pycache__' in path.parts or path.suffix in ['.lock','.tar'] or path.name=='COMPLETE.json' or 'vendor' in path.parts:continue
   artifacts.append(record(path))
 artifacts += [record(ROOT/n) for n in ['README.md','PROTOCOL.md','REPORT.md']]
 atomic(ROOT/'results/COMPLETE.json',{'at_utc':now(),'complete':True,'tests':checked,'models':NAMES,
  'checkpoint_and_runtime_hashes_checked':True,'no_retraining':True,'all_source_rows_retained':True,
  'exported_metrics_independently_recomputed':True,'native_human_fp32_archive_check_passed':True,
  'figure_models':figure_models,'figure_models_count':len(figure_models),
  'artifacts':artifacts,'verifier':record(__file__)})
 print(f'COMPLETE: 11 primary predictors, {len(figure_models)} displayed models × 5 species; 242000 source rows, all checks passed.',flush=True)
if __name__=='__main__':main()
