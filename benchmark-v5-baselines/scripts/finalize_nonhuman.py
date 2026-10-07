"""Seal the combined study only after verifying preserved and new results."""
import csv,re,platform,sys
import sklearn
from nonhuman_common import *

def main():
    freeze_extension();verify_legacy_models()
    verification=read(ROOT/'results/nonhuman-verification.json');assert verification['status']=='passed'
    old=read(ARCHIVE/'results/summary.json');combined=read(ROOT/'results/summary.json');new=read(ROOT/'results/nonhuman-summary.json')
    assert set(combined['datasets'])==set(DATASETS+SPECIES)
    for d in DATASETS:assert combined['datasets'][d]==old['datasets'][d]
    for d in SPECIES:assert combined['datasets'][d]==new['datasets'][d]
    def rows(p):return list(csv.DictReader(Path(p).open()))
    for name,added in [('metrics.csv','nonhuman-metrics.csv'),('confidence-intervals.csv','nonhuman-confidence-intervals.csv'),('paired-differences.csv','nonhuman-paired-differences.csv')]:
        assert rows(ROOT/'results'/name)==rows(ARCHIVE/'results'/name)+rows(ROOT/'results'/added)
    assert len(rows(ROOT/'results/metrics.csv'))==85
    for item in read(ROOT/'provenance/nonhuman/evaluation.json')['artifacts']:verify(item)
    for stem in ['comparison','nonhuman-comparison','nonhuman-paired-differences']:
        for ext in ['png','pdf','svg']:assert (ROOT/'results'/(stem+'.'+ext)).stat().st_size>100
    for name in ['README.md','REPORT.md','PROTOCOL.md','NONHUMAN-ADDENDUM.md']:
        p=ROOT/name
        for link in re.findall(r'\]\(([^)]+)\)',p.read_text()):
            if '://' not in link and not link.startswith('#'):assert (p.parent/link.split('#')[0]).exists(),(p,link)
    paired=rows(ROOT/'results/nonhuman-paired-differences.csv')
    bernett_vs_v2=[r for r in paired if r['model'].startswith('bernett:') and 'exact-degree' not in r['model'] and r['reference'].startswith('v2-')]
    assert len(bernett_vs_v2)==60 and all(float(r['high'])<0 for r in bernett_vs_v2)
    v5_interolog_ap=[r for r in paired if r['model']=='v5:interolog' and r['reference'].startswith('ipin-') and r['metric']=='ap']
    assert len(v5_interolog_ap)==10 and all(float(r['low'])>0 for r in v5_interolog_ap)
    boundary=read(ROOT/'provenance/nonhuman/identity-boundary-check.json')
    assert boundary['direct_identity']>=.25 and boundary['raw_sequence_alignment_segments_verified'] and not boundary['scores_changed']
    runtime={'at_utc':now(),'environment':{'python':sys.version,'numpy':np.__version__,'sklearn':sklearn.__version__,'node':platform.node()},
        'container_identity_reused_from_original_verified_runtime':read(ROOT/'provenance/runtime.json')['container'],
        'container_hash_recomputed_this_extension':False,'cpu_threads_per_search':16,'maximum_parallel_search_threads':32,
        'v5_regressor_refitted':False,'new_neural_training_or_inference':False,'new_slurm_jobs':False,
        'resource_records':[record(p) for p in sorted((ROOT/'logs/nonhuman').glob('*.resources.txt'))]}
    atomic(ROOT/'provenance/nonhuman/runtime.json',runtime)
    allowed_work={'work/alignments.tsv','work/nonhuman/alignments-v5.tsv','work/nonhuman/alignments-bernett.tsv',
        'work/nonhuman/backtrace-v5.tsv','work/nonhuman/backtrace-bernett.tsv','work/nonhuman/boundary-alignment.tsv','work/nonhuman/boundary-query.fasta'}
    destination=ROOT/'results/COMPLETE.json';artifacts=[]
    for p in sorted(ROOT.rglob('*')):
        if not p.is_file() or p==destination:continue
        rel=p.relative_to(ROOT)
        if rel.parts[0] in ['archive','cache'] or '__pycache__' in rel.parts or p.suffix in ['.partial','.lock']:continue
        if rel.parts[0]=='work' and rel.as_posix() not in allowed_work:continue
        if rel.parts[0]=='logs' and not p.name.endswith('.resources.txt'):continue
        artifacts.append({'path':rel.as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    atomic(destination,{'complete':True,'at_utc':now(),'scope':'Original v5 human study plus v5/Bernett five-species extension',
        'human_evaluations':3,'nonhuman_species':5,'nonhuman_rows':242000,'nonhuman_predictors':13,'artifacts':artifacts,
        'original_human_complete_manifest':record(ARCHIVE/'results/COMPLETE.json'),
        'checks':{'original_human_metric_interval_and_difference_rows_identical':True,'original_v5_model_artifacts_unchanged':True,
            'all_five_species_neural_predictions_reused':True,'nonhuman_independent_checks':len(verification['checks']),
            'primary_score_arrays_not_modified_by_precision_audits':True,'all_report_links_resolve':True,
            'all_three_figure_groups_exported_in_three_formats':True}})
    print('Combined study complete:',len(artifacts),'artifacts; 85 metric rows; original human results preserved.',flush=True)

if __name__=='__main__':main()
