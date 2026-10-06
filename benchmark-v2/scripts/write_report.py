"""Write the complete seven-model comparison from audited numerical results."""
import datetime
import json
from pathlib import Path
from analyze import NAMES, LABELS
from common import ROOT, atomic_json, sha256

def main():
    summary = json.loads((ROOT / 'results/benchmark-summary.json').read_text())
    selection = summary['selection']
    primary = summary['primary']
    new = selection['fresh_inference_models']
    def interval(name, metric='ap', group='all'):
        return next(x for x in summary['confidence_intervals'] if x['model']==name and x['metric']==metric and x['group']==group)
    def difference(a, b, metric='ap', group='all'):
        for x in summary['paired_differences']:
            if x['metric']!=metric or x['group']!=group: continue
            if x['comparison']==f'{a} minus {b}': return x
            if x['comparison']==f'{b} minus {a}':
                return {**x,'comparison':f'{a} minus {b}','difference':-x['difference'],'low':-x['high'],'high':-x['low']}
        raise KeyError((a,b,metric,group))
    def ci_text(x): return f"[{x['low']:+.6f}, {x['high']:+.6f}]"
    gains = [LABELS[n] for n in new if primary[n]['ap'] > primary['native-bernett']['ap']]
    opening = (', '.join(gains) + (' has' if len(gains)==1 else ' have') + ' higher pooled test AP than the released native checkpoint on this historical test set.'
               if gains else 'All four v2 models have lower pooled test AP than the released native checkpoint on this historical test set.')
    leader = max(new, key=lambda n:primary[n]['ap'])
    leader_delta = difference(leader, 'native-bernett')
    if leader_delta['low'] <= 0 <= leader_delta['high']:
        opening = (f"{LABELS[leader]} has the highest v2 test AP ({primary[leader]['ap']:.6f}), compared with native PLM-interact's {primary['native-bernett']['ap']:.6f}. "
                   f"Its AP difference from native is {leader_delta['difference']:+.6f}, with conditional 95% interval {ci_text(leader_delta)}. "
                   'This does not demonstrate AP superiority over native.')
    lines = ['**Benchmark v2 completed Bernett test comparison**', '', opening, '',
        'This compares the existing validation-selected checkpoint from every completed initial v2 run with native PLM-interact and both v1 models. All seven results are retained. This historically reused test set provides an exploratory comparison, not independent confirmation of improved generalization.', '',
        f"V2 checkpoint identities were frozen at **{selection['selected_at_utc']}**, before new test inference. Reference, Positive10 and Clean BCE use update 4,000; the length-capped model uses update 8,000. Native and v1 predictions were reused after integrity and compatibility checks; their selected weights are unchanged.", '',
        'All **52,048 test pairs** were evaluated, with **26,024 positives**. Every model receives complete sequences, BF16 computation with FP32 parameters, efficient attention, disabled TF32 and mean A–B/B–A logits. AP means average precision; AP and AUROC use raw logits. Brier is squared probability error and is lower when better.', '',
        '| Model | Selected update | AP [95% interval] ↑ | AUROC ↑ | Brier ↓ |',
        '| --- | ---: | --- | ---: | ---: |']
    for name in NAMES:
        p, i = primary[name], interval(name)
        update = selection['models'][name]['update']
        u = f'{update:,}' if update is not None else 'Unreported'
        lines.append(f"| {LABELS[name]} | {u} | {p['ap']:.6f} [{i['low']:.6f}, {i['high']:.6f}] | {p['auroc']:.6f} | {p['brier']:.6f} |")
    lines += ['', '**V2 differences from native**', '',
        '| V2 model minus native | AP difference | Conditional 95% interval | AUROC difference | Brier difference ↓ |',
        '| --- | ---: | --- | ---: | ---: |']
    for name in new:
        d = difference(name,'native-bernett')
        lines.append(f"| {LABELS[name]} | {d['difference']:+.6f} | {ci_text(d)} | {primary[name]['auroc']-primary['native-bernett']['auroc']:+.6f} | {primary[name]['brier']-primary['native-bernett']['brier']:+.6f} |")
    lines += ['', '**V2 differences from v1**', '',
        '| V2 model | AP difference from V1 reference [95% interval] | AP difference from V1 symmetric [95% interval] |',
        '| --- | --- | --- |']
    for name in new:
        a,b = difference(name,'v1-reference'),difference(name,'v1-symmetric')
        lines.append(f"| {LABELS[name]} | {a['difference']:+.6f} {ci_text(a)} | {b['difference']:+.6f} {ci_text(b)} |")
    lines += ['', '**Interventions against the v2 reference**', '',
        '| Intervention minus v2 reference | AP difference | Conditional 95% interval |',
        '| --- | ---: | --- |']
    for name in new:
        if name=='v2-reference': continue
        d = difference(name,'v2-reference')
        lines.append(f"| {LABELS[name]} | {d['difference']:+.6f} | {ci_text(d)} |")
    lines += ['', 'The v2 reference is the matched control for the v2 interventions. V1 and v2 differ in their training stream and checkpoint-selection details, so differences between versions cannot be attributed to a single model modification. The capped arm changes training coverage; its test inference includes every long pair. Clean BCE removes masking and MLM together. Positive10 changes the class-weighted objective, which can shift raw probabilities without the same change in ranking.', '',
        'Intervals use 1,000 paired protein bootstrap resamples with seed 20260929. Every model receives the same endpoint multiplicities; self-pairs receive one multiplicity. They condition on this interaction graph and the selected checkpoints, excluding training-seed, remote-homology and selection uncertainty. Comparisons are descriptive and not multiplicity-adjusted. Full AP, AUROC and Brier intervals and all 21 pairwise model comparisons are in [the result CSVs](results/paired-differences.csv).', '',
        '**Validation and test separation**', '',
        '| Model | Pooled validation AP | Pooled test AP |', '| --- | ---: | ---: |']
    for name in NAMES:
        val = next(x for x in summary['validation_metrics'] if x['model']==name and x['view']=='pooled')
        lines.append(f"| {LABELS[name]} | {val['ap']:.6f} | {primary[name]['ap']:.6f} |")
    clean_native = difference('v2-clean-bce','native-bernett','brier')
    lines += ['', 'The test ordering differs from the validation ordering: Positive10 had the highest v2 validation AP, while Clean BCE and the capped model score better on test. The earlier observation that clean BCE deteriorated late in training concerns its final weights; this benchmark uses its prespecified update-4,000 selection.', '',
        f"Clean BCE improves on both v1 models in this historical comparison: AP differences are {difference('v2-clean-bce','v1-reference')['difference']:+.6f} and {difference('v2-clean-bce','v1-symmetric')['difference']:+.6f}, respectively, with conditional intervals above zero. Its Brier improvement over native is {clean_native['difference']:+.6f}, interval {ci_text(clean_native)}. This supports lower probability error against these benchmark labels, while the AP/AUROC differences from native remain inconclusive.", '',
        'Checkpoint selection used validation only. These v2 selections follow completion of the full 12,745-update horizon; the v1 selections remain the best available checkpoints after their earlier shutdown. The authors’ exact selected native training update is unreported. The historical test already informed earlier research decisions, so favorable results here would still require independent holdout evaluation and seed replication before a generalization claim.', '',
        '**Operating thresholds and probability accuracy**', '',
        'Each threshold below maximizes F1 on the same 59,258 validation pairs, choosing the highest threshold on an exact tie. Pooled thresholds were fixed before new test inference. Test labels did not choose thresholds or fit calibration.', '',
        '| Model | F1 at probability 0.5 | Validation-selected probability threshold | Test precision | Test recall | Test F1 | Test MCC |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in NAMES:
        m = next(x for x in summary['metrics'] if x['model']==name and x['view']=='pooled' and x['group']=='all' and x['threshold_rule']=='validation_max_f1')
        lines.append(f"| {LABELS[name]} | {primary[name]['f1']:.6f} | {m['probability_threshold']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['mcc']:.6f} |")
    lines += ['', 'AP/AUROC assess ranking. Brier and the thresholded metrics assess different properties; a lower Brier score does not establish better ranking. In particular, positive class weighting changes the meaning of raw sigmoid scores, so the common 0.5 threshold can be inappropriate. No probability recalibration was fitted for this benchmark.', '',
        'At probability 0.5, Positive10 predicts every test pair positive (recall 1, precision 0.5, MCC 0). Its F1 of 0.666667 at that threshold is the all-positive baseline on this balanced test set. Its validation-selected threshold gives a more useful operating point, as shown above.', '',
        '**Sequence length strata**', '',
        '| Model | AP for combined residues ≤2,193 | AP for combined residues >2,193 |',
        '| --- | ---: | ---: |']
    strata = {}
    for name in NAMES:
        values=[]
        for group in ['combined_residues_le_2193','combined_residues_gt_2193']:
            m=next(x for x in summary['metrics'] if x['model']==name and x['view']=='pooled' and x['group']==group and x['threshold_rule']=='fixed_0.5')
            strata[group]={'pairs':m['rows'],'prevalence':m['prevalence']}
            values.append(m['ap'])
        lines.append(f'| {LABELS[name]} | {values[0]:.6f} | {values[1]:.6f} |')
    short,long = strata['combined_residues_le_2193'],strata['combined_residues_gt_2193']
    lines += ['', f"The shorter stratum contains {short['pairs']:,} pairs (positive fraction {short['prevalence']:.6f}); the longer stratum contains {long['pairs']:,} pairs (positive fraction {long['prevalence']:.6f}). Special tokens add three to combined residue length. All strata use the globally validation-selected checkpoint; no subset-specific reselection was performed. Length-stratified AUROC, Brier and paired intervals are also retained.", '',
        '**Inference reuse and verification**', '',
        'Native PLM-interact, v1 reference and v1 symmetric test predictions were reused from benchmark-v1. All four v2 test predictions were generated freshly. Reuse checks verified the original checkpoint identities, SIF, full input arrays, model/data forward implementation, precision, pooling and batching. They checked 428 source prediction chunks across three prior test tasks and native validation, and required their merged arrays to equal the cached predictions exactly. Recomputed baseline metrics and complete bootstrap replicates reproduce benchmark-v1 within 1e-12.', '',
        'Each new checkpoint loaded strictly and reproduced all 160 checked validation-pair logits exactly. The shared inference implementation also matched the frozen v2 implementation exactly on the checked BF16 and FP32 forwards. Tokenization was checked, all 3,710 validation sequences matched the v2 prepared data, and the longest 39,391-token validation input passed. Existing native eager/efficient FP32 and BF16 sensitivity evidence is retained from benchmark-v1.', '',
        'Final merging checked every test row exactly once, labels, finite logits, chunk hashes and four distinct worker GPUs for each fresh model. Checkpoint identities and selection stayed fixed throughout evaluation. Negative labels are sampled unreported interactions rather than experimentally established noninteractions. This comparison concerns the Bernett human PPI task, not the paper’s separate cross-species task.', '',
        'The paper’s rounded AP/AUROC (0.69/0.70) used original-order inference. The primary comparison here gives all seven models the same full-length pooled scoring. Original-order metrics remain available as a secondary view. The [previous benchmark](../benchmark-v1/REPORT.md) documents published-score, sequence-cap and numerical-precision context.', '',
        '![Test precision recall ROC and calibration curves](results/test-curves.png)', '',
        '![Paired protein bootstrap metric intervals](results/test-metrics.png)', '',
        '[Exact results and provenance](results/benchmark-summary.json) · [All metrics](results/metrics.csv) · [Paired differences](results/paired-differences.csv) · [Frozen selection](provenance/selection.json) · [Reuse verification](provenance/reuse-verification.json) · [Qualification](provenance/qualification.json) · [Protocol](PROTOCOL.md)']
    (ROOT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    atomic_json(ROOT/'completed.json', {'completed_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'models':NAMES,'fresh_test_inference_models':new,'reused_test_inference_models':list(selection['reused_predictions']),
        'test_rows_per_model':52048,'prediction_fingerprint':summary['prediction_fingerprint'],
        'summary_sha256':sha256(ROOT/'results/benchmark-summary.json'),'report_sha256':sha256(ROOT/'REPORT.md'),
        'all_coverage_and_numerical_checks_passed':True})
    print(str(ROOT/'REPORT.md'))

if __name__=='__main__': main()
