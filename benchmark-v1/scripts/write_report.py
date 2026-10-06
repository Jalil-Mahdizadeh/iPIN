"""Render the completed fixed-snapshot benchmark without selecting new models."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NAMES = ['native-bernett', 'reference-seed2', 'symmetric-seed2']
LABELS = {'native-bernett': 'Native PLM-interact (Bernett)', 'reference-seed2': 'Reference, update 4,000',
          'symmetric-seed2': 'Symmetric, update 7,000'}


def main():
    report = json.loads((ROOT / 'results/benchmark-summary.json').read_text())
    p = report['primary']

    def interval(name, metric, group='all'):
        r = next(x for x in report['confidence_intervals'] if x['model'] == name and x['metric'] == metric and x['group'] == group)
        return f"{r['estimate']:.6f} [{r['low']:.6f}, {r['high']:.6f}]"

    def difference(comparison, metric='ap', group='all'):
        return next(x for x in report['paired_differences'] if x['comparison'] == comparison and x['metric'] == metric and x['group'] == group)

    def view(name, score_view='pooled', group='all', threshold='fixed_0.5'):
        return next(x for x in report['metrics'] if x['model'] == name and x['view'] == score_view and x['group'] == group and x['threshold_rule'] == threshold)

    rows = ['**Benchmark-v1: interim Bernett test comparison**', '']
    ahead = [n for n in NAMES[1:] if p[n]['ap'] > p['native-bernett']['ap']]
    if not ahead:
        rows.append('Both retrained snapshots have lower pooled test AP than the released native Bernett checkpoint. This benchmark does not demonstrate an AP improvement from retrain-v1 at the selected interim checkpoints.')
    elif len(ahead) == 2:
        rows.append('Both retrained snapshots have higher pooled test AP than the released native Bernett checkpoint. This is an interim, single-seed result; the paired intervals below quantify test-sample uncertainty rather than training-seed robustness.')
    else:
        rows.append(f"{LABELS[ahead[0]]} has higher pooled test AP than native PLM-interact; the other retrained snapshot has lower AP. This is an interim, single-seed comparison.")
    rows += ['', f"The model selection was frozen at **{report['selection']['selected_at_utc']}**, before benchmark inference. Reference update 4,000 corresponds to approximately 1.57 epochs; symmetric update 7,000 to approximately 2.75 epochs. Both were selected by their existing validation rules. The authors' exact selected Bernett epoch/update remains unreported. Checkpoint hashes, source revisions and selection rules are recorded in [selection.json](provenance/selection.json).", '',
        'All **52,048 test pairs** were evaluated, with **26,024 positives** and no test exclusions or sequence truncation. Both sequences and labels were checked against the original release. Every model receives the same full-length inputs, BF16 computation and pooling of A–B/B–A logits. AP means average precision, as in the authors\' code. All intervals below are approximate 95% intervals from the paired protein bootstrap.', '',
        '| Model | AP [95% interval] ↑ | AUROC [95% interval] ↑ | Brier ↓ |',
        '| --- | ---: | ---: | ---: |']
    for name in NAMES:
        rows.append(f"| {LABELS[name]} | {interval(name, 'ap')} | {interval(name, 'auroc')} | {p[name]['brier']:.6f} |")
    rows += ['', 'The native checkpoint has higher AP and AUROC, while both retrained snapshots have lower Brier scores. Lower Brier means smaller mean squared probability error against these benchmark labels; it does not establish improved ranking. The symmetric model\'s small AP advantage over the reference is inconclusive under the paired interval below.', '', 'The paired AP differences are:', '']
    for comparison in ['reference-seed2 minus native-bernett', 'symmetric-seed2 minus native-bernett', 'symmetric-seed2 minus reference-seed2']:
        d = difference(comparison)
        rows.append(f"- {comparison}: **{d['difference']:+.6f}**, interval **[{d['low']:+.6f}, {d['high']:+.6f}]**.")
    rows += ['', f"The bootstrap uses 1,000 identical protein resamples across models, covering {report['bootstrap']['unique_test_protein_sequences']:,} unique test protein sequences. Pair weights are products of their distinct endpoint multiplicities; self-pairs use one multiplicity. Intervals are conditional on this observed interaction graph and do not capture training-seed variability, all homology dependence, or model-selection uncertainty. The weighted AP/AUROC implementation was verified against scikit-learn, including ties and zero weights. See [paired differences](results/paired-differences.csv) and [all intervals](results/confidence-intervals.csv).", '',
        '![Test curves and calibration](results/test-curves.png)', '',
        'Operating points are compared using both probability 0.5 and thresholds selected to maximize F1 on the same 59,258 validation pairs. Exact F1 ties use the highest threshold. No test labels were used to choose thresholds. Retrained models reuse their saved selected-checkpoint validation predictions; native validation predictions were generated freshly.', '',
        '| Model | F1 at 0.5 | Validation-selected probability threshold | Test precision | Test recall | Test F1 | Test MCC |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name in NAMES:
        fixed = view(name)
        chosen = view(name, threshold='validation_max_f1')
        rows.append(f"| {LABELS[name]} | {fixed['f1']:.6f} | {chosen['probability_threshold']:.6f} | {chosen['precision']:.6f} | {chosen['recall']:.6f} | {chosen['f1']:.6f} | {chosen['mcc']:.6f} |")
    rows += ['', 'The sequence-length analysis uses the paper\'s 2,193-residue combined training cutoff, equivalent to 2,196 input tokens including special tokens. All models use complete sequences in both strata. Prevalence is shown because AP values across different prevalences are not directly comparable.', '',
        '| Combined residues | Pairs | Positive fraction | Native AP | Reference AP | Symmetric AP |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for group, label in [('combined_residues_le_2193', '≤ 2,193'), ('combined_residues_gt_2193', '> 2,193')]:
        values = [view(name, group=group) for name in NAMES]
        rows.append(f"| {label} | {values[0]['rows']:,} | {values[0]['prevalence']:.6f} | {values[0]['ap']:.6f} | {values[1]['ap']:.6f} | {values[2]['ap']:.6f} |")
    long_delta = difference('symmetric-seed2 minus native-bernett', group='combined_residues_gt_2193')
    rows += ['', f"For long pairs, symmetric-minus-native AP is {long_delta['difference']:+.6f}, with interval [{long_delta['low']:+.6f}, {long_delta['high']:+.6f}]. This interval spans zero; these snapshots do not demonstrate a long-pair AP gain."]
    context = report['published_context']
    audit = report['native_precision_audit']
    native_ab = view('native-bernett', score_view='ab')
    reference_ab = view('reference-seed2', score_view='ab')
    rows += ['', 'Length-specific AUROC, Brier, threshold metrics and paired uncertainty are retained in the result CSVs. The full-length training policy is shared by both retrained arms; their comparison does not isolate the benefit of removing the native training cutoff.', '',
        'The paper\'s native scores require a separate comparison because they use original-order inference. The common pooled results above are the main matched comparison.', '',
        '| Native evaluation | AP | AUROC |', '| --- | ---: | ---: |',
        '| Paper, rounded | 0.69 | 0.70 |',
        f"| Deposited predictions | {context['deposited_scores_ap']:.6f} | {context['deposited_scores_auroc']:.6f} |",
        f"| Earlier FP32 reproduction, full sequences, original order | {audit['old_ap']:.6f} | {audit['old_auroc']:.6f} |",
        f"| Fresh BF16, full sequences, original order | {native_ab['ap']:.6f} | {native_ab['auroc']:.6f} |",
        f"| Fresh BF16, full sequences, pooled | {p['native-bernett']['ap']:.6f} | {p['native-bernett']['auroc']:.6f} |", '',
        f"For completeness, the reference's prespecified original-order test view has AP {reference_ab['ap']:.6f} and AUROC {reference_ab['auroc']:.6f}. It uses the same update-4,000 weights as its pooled view. Pooling makes all three final predictors order-invariant, so that property is not unique to symmetric retraining. [Orientation diagnostics](results/orientation-diagnostics.csv) describe the underlying unpooled predictions.", '',
        f"Across all native original-order test predictions, changing from the earlier eager FP32 evaluation to the common BF16 evaluation changed AP by {audit['fresh_ap'] - audit['old_ap']:+.6f} and AUROC by {audit['fresh_auroc'] - audit['old_auroc']:+.6f}. The mean absolute probability difference is {audit['mean_absolute_probability_difference']:.6f}, the maximum is {audit['maximum_probability_difference']:.6f}, and {audit['threshold_0_5_disagreements']:,} decisions differ at 0.5. These are numerical sensitivity results, not identical predictions.", '',
        'The earlier reproduction found deposited Bernett predictions consistent with a 3,570-token inference cap. That diagnostic cap was not used here; all three current models see complete sequences. See the [native reproduction report](../plm-interact-reproducability/REPORT.md) and the [paper materials](../literature/plm-interact/READING_NOTES.md).', '',
        'Coverage checks passed for every model and split: complete unique row IDs, exact labels, finite logits and verified chunk hashes. A separate [input audit](provenance/input-audit.json) checked every sequence against the original CSVs. [Qualification](provenance/qualification.json) verified strict loading, tokenizer agreement, efficient versus eager attention in FP32, BF16 sensitivity, consistency with saved retraining validation predictions, and the longest validation input of 39,391 tokens. The supplied SIF\'s SHA-256 was independently verified.', '',
        'This comparison concerns the Bernett PPI task corresponding to Figure 4. Both retrained models start from pretrained ESM-2 and share full-length coverage, small homology exclusions and trainer corrections; the reference is a newly trained control. Differences against native PLM-interact cannot be attributed solely to the symmetric objective. The reference-versus-symmetric comparison isolates that objective within the shared corrected training regime. Negative labels remain sampled unreported interactions, and the historical test set has already been examined during the previous reproduction.', '',
        'Both original training jobs were left unchanged. These fixed best-so-far snapshots do not establish the outcome of the complete five-epoch runs. No checkpoint was selected or training parameter changed using this benchmark.', '',
        '![Protein-aware metric intervals](results/test-metrics.png)', '',
        '[Exact summary and provenance](results/benchmark-summary.json) · [All metrics](results/metrics.csv) · [Validation thresholds](results/validation-thresholds.csv) · [Protocol](PROTOCOL.md) · [Reproduction commands](README.md)', '']
    validation = {r['model']: r for r in report['validation_metrics'] if r['view'] == 'pooled'}
    position = next(i for i, line in enumerate(rows) if line.startswith('Operating points are compared'))
    rows[position:position] = [f"The common pooled validation AP endpoints are native {validation['native-bernett']['ap']:.6f}, reference {validation['reference-seed2']['ap']:.6f}, and symmetric {validation['symmetric-seed2']['ap']:.6f}. Both retrained snapshots score higher on this validation set despite lower test AP. The native endpoint is a fresh evaluation under this protocol, not an author-provided historical training record. See [validation metrics](results/validation-metrics.csv).", '']
    diagnostic_path = ROOT / 'provenance/precision-outlier-audit.json'
    if diagnostic_path.exists():
        diagnostic = json.loads(diagnostic_path.read_text())
        maximum = max(r['fp32_efficient_vs_old_eager_logit_difference'] for r in diagnostic['records'])
        position = next(i for i, line in enumerate(rows) if line.startswith('The earlier reproduction found'))
        rows[position:position] = [f"A post-hoc numerical check reran the five largest native BF16-versus-FP32 probability discrepancies, selected without labels. Efficient FP32 predictions agreed with the earlier eager FP32 predictions to a maximum logit difference of {maximum:.8f}. This supports reduced-precision sensitivity as the source of those outliers. Primary benchmark predictions and model selection were left unchanged. See [the outlier audit](provenance/precision-outlier-audit.json).", '']
    (ROOT / 'REPORT.md').write_text('\n'.join(rows))


if __name__ == '__main__':
    main()
