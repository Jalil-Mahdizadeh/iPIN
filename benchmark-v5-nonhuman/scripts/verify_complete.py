"""Independent coverage/metric/input checks before sealing the five-species report."""
import csv,gzip,hashlib,json
import numpy as np
from sklearn.metrics import average_precision_score,roc_auc_score
from bench_utils import ROOT,atomic,load_npz,now,read,record,sha
from collect import BASE_NAMES,NAMES,TESTS,PAIR_MODELS,verify

def verify_xpair_v11(summary):
 extension=read(ROOT/'provenance/xpair-v11.json')
 for key in ['checkpoint','addendum','prepared','prior_collection','archive_receipt','image','runtime','same_checkpoint_as_human_benchmark','cache_loading_adjustment']:verify(extension[key])
 for rel,h in extension['inference_code'].items():assert sha(ROOT/rel)==h,rel
 for item in extension['existing_feature_runs']:verify(item)
 assert extension['checkpoint']==read(ROOT.parent/'benchmark-v5/provenance/xpair-v11.json')['checkpoint']
 q=read(ROOT/'qualification/xpair-v11.json');done=read(ROOT/'predictions/xpair-v11/done.json')
 assert q['passed'] and q['no_truncation'] and q['state_unchanged'] and not q['test_metrics_read']
 assert q['signature']==done['signature'] and q['signature']['freeze_sha256']==sha(ROOT/'provenance/xpair-v11.json')
 assert max(q['errors'].values())<2e-4 and q['errors']['padded_probability']<1e-5
 assert max(c['length_a']+c['length_b'] for c in q['cases'])>=1598
 assert done['rows']==238025 and done['features_reused']==56634 and done['features_computed']==0
 assert done['state_unchanged'] and done['no_truncation'] and done['world_size']==1
 verify(done['file']);verify(done['qualification'])
 production=read(ROOT/'qualification/xpair-v11-production-check.json')
 assert production['passed'] and production['signature']==done['signature'] and production['state_unchanged']
 assert production['unmodified_native_full_forward'] and production['raw_features_reloaded_independently']
 assert not production['test_performance_metrics_read'] and production['max_absolute_logit_error']<2e-4
 assert len(production['cases'])==15 and {case['test'] for case in production['cases']}==set(TESTS)
 verify(production['script']);verify(production['completed_inference'])
 features=read(ROOT/'provenance/xpair-v11-features.json')
 assert features['all_sequence_and_file_hashes_checked'] and features['features_verified_and_reused']==56634
 assert features['features_computed']==0 and features['residues']==19472345
 prior=ROOT/'archive/before-xpair-v11';manifest=read(prior/'archive.json')
 verify(read(ROOT/'provenance/xpair-v11-archive.json')['manifest'])
 for rel,item in manifest['files'].items():verify({**item,'path':str(prior/rel)})
 old=read(prior/'results/summary.json');assert old['names']==BASE_NAMES and len(NAMES)==12
 for name in BASE_NAMES:assert sha(prior/'results'/f'{name}-union.npz')==sha(ROOT/'results'/f'{name}-union.npz'),name
 assert summary['common_unexposed_counts']==old['common_unexposed_counts']
 for test in TESTS:
  for name in BASE_NAMES:assert summary['tests'][test]['models'][name]==old['tests'][test]['models'][name],(test,name)
  for suffix in ['', '__common_unexposed']:
   a=load_npz(prior/'results'/f'{test}{suffix}-bootstrap.npz');b=load_npz(ROOT/'results'/f'{test}{suffix}-bootstrap.npz')
   for j,name in enumerate(a['names']):
    k=b['names'].tolist().index(name);assert np.array_equal(a['samples'][:,j],b['samples'][:,k]),(test,suffix,name)
 for filename in ['metrics.csv','subsets.csv','orientation-metrics.csv','confidence-intervals.csv','paired-differences.csv','combined-figure-metrics.csv','combined-figure-confidence-intervals.csv']:
  with (prior/'results'/filename).open() as f:a=list(csv.DictReader(f))
  with (ROOT/'results'/filename).open() as f:b=[r for r in csv.DictReader(f) if r['model']!='xpair-v11' and r.get('reference')!='xpair-v11']
  assert a==b,filename
 exposure=read(ROOT/'provenance/xpair-v11-exposure.json');verify(exposure['script']);flags=load_npz(verify(exposure['flags']))
 assert exposure['exact_flags_match_prior_human_release_audit'] and exposure['existing_human_and_all_source_masks_cover_all_identified_exposure']
 previous=load_npz(ROOT/'provenance/exposure-flags.npz')
 for key in ['endpoints','pairs']:assert np.array_equal(flags['exact__'+key],previous['native-human__exact__'+key])
 for p in (ROOT/'results').glob('*.svg'):assert 'X-PAIR (humanV11)' in p.read_text(),p
 return {'all_thirteen_prior_predictors_and_statistics_preserved':True,
         'xpair_v11_full_length_native_qualification':True,'all_56634_features_verified_and_reused':True,
         'production_scores_reproduced_by_native_forward_in_all_five_species':True,
         'existing_common_exposure_masks_cover_xpair_v11':True,'eleven_image_files_include_fourteen_models':True}

def main():
 c=read(ROOT/'results/collection.json');s=read(ROOT/'results/summary.json')
 assert c['complete'] and s['complete'] and set(c['models'])==set(NAMES)==set(s['names'])
 roster=read(verify(c['effective_roster']));assert list(roster['models'])==NAMES
 verify(c['roster_extension'])
 assert set(s['tests'])==set(TESTS) and sum(s['tests'][t]['counts']['rows'] for t in TESTS)==242000
 verify(s['collection']);verify(s['protocol']);verify(s['analysis_script'])
 extension_checks=verify_xpair_v11(s)
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
  assert len(figure_rows)==210 and {r['model'] for r in figure_rows}==set(extension['models'])
  assert len({(r['test'],r['model'],r['subset']) for r in figure_rows})==210
  with verify(extension['intervals']).open() as stream:figure_intervals=list(csv.DictReader(stream))
  assert len(figure_intervals)==140
  figure_models=extension['models']
 artifacts=[]
 for folder in ['results','qualification','provenance','scripts','slurm']:
  for path in sorted((ROOT/folder).rglob('*')):
   if not path.is_file() or '__pycache__' in path.parts or path.suffix in ['.lock','.tar'] or path.name=='COMPLETE.json' or 'vendor' in path.parts:continue
   artifacts.append(record(path))
 artifacts += [record(ROOT/n) for n in ['README.md','PROTOCOL.md','REPORT.md','XPAIR-V11-ADDENDUM.md']]
 atomic(ROOT/'results/COMPLETE.json',{'at_utc':now(),'complete':True,'tests':checked,'models':NAMES,
  'checkpoint_and_runtime_hashes_checked':True,'no_retraining':True,'all_source_rows_retained':True,
  'exported_metrics_independently_recomputed':True,'native_human_fp32_archive_check_passed':True,
  'figure_models':figure_models,'figure_models_count':len(figure_models),
  'extension_checks':extension_checks,'artifacts':artifacts,'verifier':record(__file__)})
 atomic(ROOT/'xpair-v11-status.json',{'at_utc':now(),'complete':True,'phase':'complete','checks':extension_checks,'completion':record(ROOT/'results/COMPLETE.json')})
 print(f'COMPLETE: {len(NAMES)} primary predictors, {len(figure_models)} displayed models × 5 species; 242000 source rows, all checks passed.',flush=True)
if __name__=='__main__':main()
