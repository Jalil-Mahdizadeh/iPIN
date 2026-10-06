"""Seal and subsequently verify the final benchmark's scientific artifacts."""
import argparse,csv,gzip
from pathlib import Path
import numpy as np
from bench_utils import ROOT,atomic,load_npz,now,read,record,sha
from collect import NAMES,verify

def verify_historical_code(rel,expected):
 if sha(ROOT/rel)==expected:return
 receipt=read(ROOT/'provenance/xpair-v11-archive.json')
 entry=next(x for x in receipt['rename_only_code_differences'] if x['path']==rel)
 assert sha(ROOT/rel)==entry['current_sha256'] and expected==entry['sealed_sha256'],rel
 assert sha(ROOT/'archive/before-xpair-v11'/rel)==expected,rel

def xpair_v11_checks(summary):
 extension=read(ROOT/'provenance/xpair-v11.json')
 for key in ['checkpoint','addendum','prepared','prior_collection','archive_receipt','runtime']:verify(extension[key])
 for rel,h in extension['inference_code'].items():assert sha(ROOT/rel)==h,rel
 q=read(ROOT/'qualification/xpair-v11.json');done=read(ROOT/'predictions/xpair-v11/done.json')
 assert q['passed'] and q['no_truncation'] and q['state_unchanged'] and not q['test_metrics_read']
 assert q['signature']==done['signature'] and q['signature']['freeze_sha256']==sha(ROOT/'provenance/xpair-v11.json')
 assert max(q['errors'].values())<2e-4 and q['errors']['padded_probability']<1e-5
 assert max(c['length_a']+c['length_b'] for c in q['cases'])>=7423
 assert done['rows']==76918 and done['features_reused']==3022 and done['features_computed']==0
 assert done['state_unchanged'] and done['no_truncation'] and done['world_size']==1
 verify(done['file']);verify(done['qualification'])
 prior=ROOT/'archive/before-xpair-v11';manifest=read(prior/'artifact-manifest.json')
 for rel,item in manifest['files'].items():
  p=prior/rel;assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],rel
 old=read(prior/'results/summary.json');assert len(old['names'])==11 and len(NAMES)==12
 for name in old['names']:
  assert sha(prior/'results'/f'{name}-union.npz')==sha(ROOT/'results'/f'{name}-union.npz'),name
 for test in ['original','ilp']:
  a=load_npz(prior/'results'/f'{test}-bootstrap.npz');b=load_npz(ROOT/'results'/f'{test}-bootstrap.npz')
  for j,name in enumerate(a['names']):
   k=b['names'].tolist().index(name)
   assert np.array_equal(a['samples'][:,j],b['samples'][:,k]),(test,name)
   assert summary['tests'][test]['models'][name]==old['tests'][test]['models'][name],(test,name)
 for filename in ['metrics.csv','subsets.csv','operating-points.csv','protein-macro.csv','confidence-intervals.csv','paired-differences.csv','combined-exposure-removed-metrics.csv']:
  with (prior/'results'/filename).open() as f:a=list(csv.DictReader(f))
  with (ROOT/'results'/filename).open() as f:b=[r for r in csv.DictReader(f) if r['model']!='xpair-v11' and r.get('reference')!='xpair-v11']
  assert a==b,filename
 exposure=read(ROOT/'provenance/xpair-v11-exposure.json');flags=load_npz(verify(exposure['flags']))
 assert exposure['exact_flags_match_prior_human_release_audit'] and exposure['existing_dscript_endpoint_mask_removes_all_identified_exposure']
 oldflags=load_npz(ROOT/'provenance/exposure-flags.npz')
 assert np.array_equal(flags['ankh-normalized__endpoints']>0,oldflags['dscript__exact__endpoints']>0)
 # Recompute every predictor's AP/AUROC from the actual exported rows.
 from sklearn.metrics import average_precision_score,roc_auc_score
 mapping=load_npz(ROOT/'data/pair-mapping.npz')
 arrays={name:load_npz(ROOT/'results'/f'{name}-union.npz') for name in NAMES}
 for test in ['original','ilp']:
  with gzip.open(ROOT/'results'/f'{test}-predictions.csv.gz','rt') as stream:export=list(csv.DictReader(stream))
  rows=np.load(ROOT/'data'/f'{test}.npy');ids=mapping[test];y=rows[:,2]
  assert len(export)==len(rows)==52048
  assert np.array_equal([int(r['row_id']) for r in export],np.arange(len(rows)))
  assert np.array_equal([int(r['source_row_id']) for r in export],rows[:,3])
  assert np.array_equal([int(r['union_id']) for r in export],ids)
  assert np.array_equal([int(r['label']) for r in export],y)
  for name in NAMES:
   scores=np.array([float(r[name+'_score']) for r in export])
   assert np.array_equal(scores,arrays[name]['scores'][ids])
   m=summary['tests'][test]['models'][name]
   assert abs(average_precision_score(y,scores)-m['ap'])<1e-12
   assert abs(roc_auc_score(y,scores)-m['auroc'])<1e-12
 figures=read(ROOT/'provenance/main-figures.json');assert figures['models']==NAMES and len(figures['figures'])==4
 verify(figures['script']);verify(figures['collection']);verify(figures['confidence_intervals'])
 for item in figures['figures']:verify(item)
 for p in (ROOT/'results').glob('*.svg'):assert 'X-PAIR (humanV11)' in p.read_text(),p
 image_files=[p for p in (ROOT/'results').iterdir() if p.suffix in ['.png','.pdf','.svg']]
 assert len(image_files)==13
 for filename in ['exposure-subset-figures.json','combined-exposure-subset-figure.json']:
  item=read(ROOT/'provenance'/filename)
  tables=[x['data'] for x in item['figures']] if 'figures' in item else [item['data']]
  for table in tables:
   for test in ['original','ilp']:assert set(table[test]['models'])==set(NAMES)
 return {'all_eleven_prior_scores_metrics_tables_and_bootstraps_unchanged':True,
  'xpair_v11_full_length_native_qualification':True,'all_3022_features_verified_and_reused':True,
  'exported_row_identity_and_all_metrics_independently_verified':True,'all_13_images_include_twelve_models':True,
  'xpair_v11_documented_source_exposure_checked':True}

def scientific_checks():
 s=read(ROOT/'results/summary.json');assert s['complete'] and s['names']==NAMES
 collection=read(ROOT/'results/collection.json');assert collection['complete'] and not collection['missing']
 assert s['collection']['sha256']==sha(ROOT/'results/collection.json')
 assert s['protocol']['sha256']==sha(ROOT/'PROTOCOL.md') and s['analysis_script']['sha256']==sha(ROOT/'scripts/analyze.py')
 prep=read(ROOT/'provenance/prepared.json')
 for name,expected in prep['data_files'].items():assert sha(ROOT/name)==expected,name
 assert sha(ROOT/'provenance/selection.json')==prep['selection_sha256']
 assert read(ROOT/'provenance/protocol-freeze.json')['sha256']==sha(ROOT/'PROTOCOL.md')
 assert read(ROOT/'provenance/dscript-runtime.json')['addendum_sha256']==sha(ROOT/'DSCRIPT-ADDENDUM.md')
 for test in ['original','ilp']:
  assert s['tests'][test]['counts']['pairs']==52048 and s['tests'][test]['counts']['positives']==26024
  b=load_npz(ROOT/'results'/f'{test}-bootstrap.npz');assert b['names'].tolist()==NAMES and b['samples'].shape==(1000,len(NAMES),2) and np.isfinite(b['samples']).all()
 for name,item in collection['models'].items():
  a=load_npz(verify(item['file']));assert len(a['scores'])==76918 and np.isfinite(a['scores']).all()
  if name!='sprint':assert np.isfinite(a['probabilities']).all()
 with (ROOT/'results/metrics.csv').open() as f:rows=list(csv.DictReader(f))
 assert len(rows)==2*len(NAMES)
 for r in rows:
  m=s['tests'][r['test']]['models'][r['model']]
  assert abs(float(r['ap'])-m['ap'])<1e-15 and abs(float(r['auroc'])-m['auroc'])<1e-15
  assert int(r['pairs'])==52048
 previous=read(ROOT/'results/provisional-intervals.json')
 for test in ['original','ilp']:
  for old in previous['tests'][test]['differences']:
   new=next(x for x in s['paired_differences'] if all(x[k]==old[k] for k in ['test','model','reference','metric']))
   for field in ['difference','low','high']:assert abs(new[field]-old[field])<1e-12
 sp=read(ROOT/'predictions/sprint/done.json');verify(sp['file']);verify(sp['graph']);verify(sp['input_contract'])
 hsp=read(ROOT/'predictions/sprint/hsp-complete.json');verify(hsp['file']);verify(hsp['raw_file'])
 for rel,expected in read(ROOT/'provenance/sprint-input.json')['files'].items():assert sha(ROOT/rel)==expected,rel
 assert read(ROOT/'qualification/sprint-inherited-native.json')['passed']
 execution=read(ROOT/'predictions/sprint/execution.json')
 assert execution['execution_addendum_sha256']==sha(ROOT/'SPRINT-EXECUTION-ADDENDUM.md')
 assert execution['qualification_sha256']==sha(Path(execution['qualification']))
 assert execution['training_edges']==350382 and execution['candidate_pairs']==76918
 qualification=read(Path(execution['qualification']))
 assert qualification['passed'] and qualification['native_output_identical_bytes'] and qualification['max_score_error']==0
 assert qualification['pairs']==304 and qualification['nonzero_pairs']==293
 assert verify(execution['binary'])==verify(qualification['binary'])
 native=verify(qualification['native_output']);requested=verify(qualification['requested_output'])
 assert native.read_bytes()==requested.read_bytes()
 for item in list(qualification['sources'].values())+qualification['fixture_files']:verify(item)
 figures_path=ROOT/'provenance/exposure-subset-figures.json'
 if figures_path.exists():
  figures=read(figures_path);verify(figures['script'])
  assert len(figures['figures'])==2 and all(figures['checks'].values())
  for item in figures['sources']:verify(item)
  for figure in figures['figures']:
   assert len(figure['files'])==3
   for item in figure['files']:verify(item)
 combined_path=ROOT/'provenance/combined-exposure-subset-figure.json'
 if combined_path.exists():
  combined=read(combined_path);verify(combined['script']);verify(combined['plot_helper'])
  assert all(combined['checks'].values()) and not combined['inference_repeated']
  for item in combined['sources']+combined['figure']['files']:verify(item)
  indices=load_npz(verify(combined['retained_rows']))
  with verify(combined['metrics_csv']).open() as f:subset_rows=list(csv.DictReader(f))
  assert len(subset_rows)==2*len(NAMES) and {(r['test'],r['model']) for r in subset_rows}=={(t,n) for t in ['original','ilp'] for n in NAMES}
  for test,n in [('original',1073),('ilp',1057)]:
   assert combined['data'][test]['counts']['pairs']==n and len(indices[test+'_row_ids'])==n
  for row in subset_rows:
   item=combined['data'][row['test']]
   assert int(row['pairs'])==item['counts']['pairs'] and int(row['positives'])==566
   for metric in ['ap','auroc']:assert abs(float(row[metric])-item['models'][row['model']][metric])<1e-15
 extension=read(ROOT/'provenance/human-releases.json');verify(extension['addendum']);verify(extension['prepared'])
 for rel,h in extension['inference_code'].items():verify_historical_code(rel,h)
 for item in extension['models'].values():verify(item['checkpoint'])
 for item in extension['images'].values():verify(item)
 exposure=read(ROOT/'provenance/human-releases-exposure.json');flags=load_npz(verify(exposure['flags']))
 assert exposure['source_datasets_identical'] and exposure['endpoint_mask_identical_to_dscript_train']
 oldflags=load_npz(ROOT/'provenance/exposure-flags.npz')
 assert np.array_equal(flags['endpoints']>0,oldflags['dscript__exact__endpoints']>0)
 for source in exposure['sources']:
  for key in ['native_file','tuna_dictionary','tuna_interactions']:verify(source[key])
 for name in ['native-human','tuna-human']:
  q=read(ROOT/'qualification'/f'{name}.json');assert q['passed'] and q['no_truncation'] and not q['test_metrics_read']
 assert max(c['tokens'] for c in read(ROOT/'qualification/native-human.json')['cases'])==7426
 prior=ROOT/'archive/before-human-releases';manifest=read(prior/'artifact-manifest.json')
 for rel,item in manifest['files'].items():assert sha(prior/rel)==item['sha256'],rel
 previous_summary=read(prior/'results/summary.json')
 for name in previous_summary['names']:
  old=load_npz(prior/'results'/f'{name}-union.npz');new=load_npz(ROOT/'results'/f'{name}-union.npz')
  assert set(old)==set(new) and all(np.array_equal(old[k],new[k]) for k in old),name
 for test in ['original','ilp']:
  old=load_npz(prior/'results'/f'{test}-bootstrap.npz');new=load_npz(ROOT/'results'/f'{test}-bootstrap.npz')
  for j,name in enumerate(old['names']):
   k=new['names'].tolist().index(name);assert np.array_equal(old['samples'][:,j],new['samples'][:,k]),(test,name)
  for name in previous_summary['names']:
   for key in ['ap','auroc']:assert s['tests'][test]['models'][name][key]==previous_summary['tests'][test]['models'][name][key]
 assert (ROOT/'REPORT.md').exists() and 'complete' in (ROOT/'README.md').read_text().lower()
 return {**xpair_v11_checks(s),'predictors':len(NAMES),'tests':2,'test_rows_per_predictor':104096,'union_rows_per_predictor':76918,
  'finite_full_coverage':True,'neural_intervals_unchanged_after_sprint':True,'frozen_data_unchanged':True,
  'metrics_csv_matches_summary':True,'sprint_graph_and_hsp_hashes_verified':True,'native_original_predictions_reproduced_exactly':s['native_original_predictions_reproduced_exactly'],
  'all_nine_previous_scores_and_bootstraps_unchanged':True,'added_releases_full_length_qualified':True,
  'added_releases_exposure_checked':True,'prior_sealed_artifacts_preserved':True}

def seal():
 checks=scientific_checks();files={}
 for directory in ['scripts','slurm','data','provenance','qualification','results','bin']:
  for p in sorted((ROOT/directory).rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts and not p.name.endswith(('.lock','.tmp')):
    files[str(p.relative_to(ROOT))]={'bytes':p.stat().st_size,'sha256':sha(p)}
 for name in ['README.md','REPORT.md','PROTOCOL.md','RUNNING.md','DSCRIPT-ADDENDUM.md','SPRINT-EXECUTION-ADDENDUM.md','HUMAN-RELEASES-ADDENDUM.md','XPAIR-V11-ADDENDUM.md']:
  p=ROOT/name;files[name]={'bytes':p.stat().st_size,'sha256':sha(p)}
 path=ROOT/'artifact-manifest.json';atomic(path,{'at_utc':now(),'files':files,'scope':'Code, protocol, data, provenance, qualifications and final/retained provisional analyses. Large raw feature/checkpoint inputs have separate verified identities in their manifests.'})
 atomic(ROOT/'completed.json',{'completed_at_utc':now(),'complete':True,'checks':checks,'artifact_manifest':record(path),'summary':record(ROOT/'results/summary.json'),'report':record(ROOT/'REPORT.md')})
 atomic(ROOT/'xpair-v11-status.json',{'at_utc':now(),'complete':True,'phase':'complete','checks':checks,'completion':record(ROOT/'completed.json')})
 print({'sealed':True,'files':len(files),'checks':checks},flush=True)

def check(inputs=False):
 complete=read(ROOT/'completed.json');assert complete['complete'];manifest=read(verify(complete['artifact_manifest']))
 for rel,item in manifest['files'].items():
  p=ROOT/rel;assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256'],rel
 if inputs:
  for item in read(ROOT/'provenance/runtime-inputs.json')['items'].values():verify(item)
  for m in read(ROOT/'provenance/selection.json')['models'].values():verify(m['checkpoint'])
  verify(read(ROOT/'provenance/dscript-runtime.json')['image'])
  extension=read(ROOT/'provenance/human-releases.json')
  for item in extension['models'].values():verify(item['checkpoint'])
  for item in extension['images'].values():verify(item)
  v11=read(ROOT/'provenance/xpair-v11.json');verify(v11['checkpoint']);verify(v11['image'])
 print({'verified':True,'artifacts':len(manifest['files']),'external_inputs_rehashed':inputs},flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--seal',action='store_true');p.add_argument('--inputs',action='store_true');a=p.parse_args()
 if a.seal:seal()
 else:check(a.inputs)
