"""Report the requested C0 benchmark, preserving the stopped-arm history."""
import datetime
import json
from common import ROOT, atomic_json, sha256
from analyze import LABELS, NAMES


def main():
    summary_path = ROOT / 'results/benchmark-summary.json'
    s = json.loads(summary_path.read_text())
    macro_path = ROOT / 'results/protein-macro-summary.json'
    macro = json.loads(macro_path.read_text())
    candidate = 'v4-esmc-standard'
    selected = s['selection']['models'][candidate]
    differences = {name: next(d for d in s['paired_differences'] if d['group'] == 'all'
        and d['metric'] == 'ap' and d['comparison'] == f'{candidate} minus {name}')
        for name in NAMES[:-1]}
    native = differences['native-bernett']
    if native['low'] > 0:
        conclusion = 'ESMC standard attention improves pooled AP over native on this historical test, with a descriptive paired 95% interval above zero.'
    elif native['difference'] > 0:
        conclusion = 'ESMC standard attention has higher point-estimate AP than native, but the paired 95% interval includes zero; superiority is not established.'
    else:
        conclusion = 'ESMC standard attention does not outperform native in pooled test AP.'
    numerical_target = native['difference'] >= .010 and native['low'] > 0
    lines = ['# V4 ESMC test benchmark', '', conclusion, '',
        f"ESMC 600M standard attention (C0), seed 2: selected update **{selected['update']:,}**, validation AP **{selected['validation_primary_ap']:.6f}**. "
        'The complete 12,745-update training horizon and all 13 validations finished before selection. The final update was not selected.', '',
        'All nine models are compared on the same **52,048 Bernett human PPI test pairs**, including **26,024 positives**. '
        'Only C0 received new test inference. Native, both v1 models, all four v2 models and v3 reuse their verified predictions.', '',
        '| Model | Selected update | Test AP | Test AUROC | Brier (lower is better) |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in NAMES:
        m = s['primary'][name]
        update = s['selection']['models'][name].get('update')
        update = f'{update:,}' if update is not None else 'Unreported'
        lines.append(f"| {LABELS[name]} | {update} | {m['ap']:.6f} | {m['auroc']:.6f} | {m['brier']:.6f} |")
    lines += ['', '## Paired AP differences', '',
        'C0 minus each comparator. Intervals use the same 1,000 protein-resampling replicates (seed 20260929) as the previous benchmarks.', '',
        '| Comparator | AP difference | Descriptive 95% interval |', '| --- | ---: | --- |']
    for name, d in differences.items():
        lines.append(f"| {LABELS[name]} | {d['difference']:+.6f} | [{d['low']:+.6f}, {d['high']:+.6f}] |")
    lines += ['', '## Validation and length diagnostics', '',
        'Validation scores below use pooled logits from each already selected checkpoint. All test strata retain those global selections.', '',
        '| Model | Validation AP | Test AP, residues ≤2,193 | Test AP, residues >2,193 | Protein-macro AP |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in NAMES:
        v = next(x for x in s['validation_metrics'] if x['model'] == name and x['view'] == 'pooled')
        strata = {x['group']: x for x in s['metrics'] if x['model'] == name and x['view'] == 'pooled'
                  and x['threshold_rule'] == 'fixed_0.5'}
        lines.append(f"| {LABELS[name]} | {v['ap']:.6f} | {strata['combined_residues_le_2193']['ap']:.6f} | "
            f"{strata['combined_residues_gt_2193']['ap']:.6f} | {macro['macro_ap'][name]:.6f} |")
    lines += ['', 'The shorter and longer strata contain 48,656 and 3,392 pairs, respectively. Special tokens add three to residue length. '
        f"Protein-macro AP gives equal weight to {macro['eligible_proteins']:,} test proteins with both positive and negative incident pairs; "
        f"{macro['excluded_single_class_proteins']:,} single-class proteins are excluded. Self-pairs count once for their protein. "
        'This descriptive diagnostic is not used for model selection or the pooled-AP success criterion.', '',
        '## Validation-selected operating thresholds', '',
        'Each model uses the highest logit threshold attaining maximum validation F1. Thresholds were frozen before C0 test inference; no calibration was fitted.', '',
        '| Model | Probability threshold | Test precision | Test recall | Test F1 | Test MCC |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name in NAMES:
        m = next(x for x in s['metrics'] if x['model'] == name and x['view'] == 'pooled'
                 and x['group'] == 'all' and x['threshold_rule'] == 'validation_max_f1')
        lines.append(f"| {LABELS[name]} | {m['probability_threshold']:.6f} | {m['precision']:.6f} | "
            f"{m['recall']:.6f} | {m['f1']:.6f} | {m['mcc']:.6f} |")
    lines += ['', '## Scope and stopped experiments', '',
        'This is the user-requested test comparison of C0, an originally planned standard-attention backbone control. '
        'The original v4 proposal nominated C1 (ESMC chain-aware) as its primary native-model comparison. C1 was stopped for the diagnosed attention failure; '
        'C0 is not retrospectively relabeled as that primary arm, and the incomplete factorial experiment cannot support an attention interaction claim.', '',
        '| Stopped v4 arm | Stopped update | Best validation AP | Test inference here |',
        '| --- | ---: | ---: | --- |']
    for run, label in [('esm2-chain-aware-official-seed2', 'S1: ESM2 chain-aware'),
                       ('esmc-chain-aware-official-seed2', 'C1: ESMC chain-aware')]:
        stopped = json.loads((ROOT / 'provenance' / f'{run}-stopped.json').read_text())
        best = json.loads((ROOT / 'provenance' / f'{run}-best.json').read_text())
        lines.append(f"| {label} | {stopped['state']['update']:,} | {best['best_ap']:.6f} | Not performed; user stopped this arm |")
    lines += ['', f"The proposal's numerical practical-gain criterion (+0.010 absolute AP over native and a paired interval above zero) "
        f"is {'met' if numerical_target else 'not met'} by C0 descriptively. The original C1 primary comparison remains uncompleted.", '',
        'The train/validation partitions remain 163,085/59,258 pairs. C0 uses full sequences, ESMC pretrained initialization, the v3 residue-MLP design, '
        'clean BCE, initial LR 2e-5 and 2,000-update warmup. C0 is selected at update 7,000 after 13 validation opportunities. '
        'V3/S0 was stopped at update 8,446 and selected update 4,000 from eight validations. V2 completed the full horizon; v1 was stopped early. '
        'These different selection histories and training exposures limit causal attribution to the backbone alone.', '',
        'Scores average AB/BA raw logits for every model. ESMC inference uses the same SIF, frozen forward implementation and token IDs as its training: '
        'FP32 parameters, BF16 autocast, efficient attention, TF32 disabled, and no sequence truncation. The four fixed shards retain the previous benchmarks\' '
        'row assignment, sorting and microbatches. Atomic chunks support restart. Final merging checks unique full row coverage, labels, finite scores and worker identity.', '',
        'Reused prediction hashes and row identities are verified against benchmark-v3. All eight baseline AP/AUROC/Brier values and every protein-bootstrap '
        'replicate reproduce benchmark-v3 within 1e-12. Strict checkpoint loading and selected-validation forward checks precede new test inference. '
        'Original-order scores, pooled scores, fixed-0.5 metrics and full paired AP/AUROC/Brier comparisons remain in the CSV outputs.', '',
        'This remains an exploratory historical-test comparison with one seed per training condition. The test informed earlier research decisions. '
        'Protein-bootstrap intervals condition on the current graph and chosen checkpoints, omit training-seed and some homology/selection uncertainty, '
        'and are not adjusted for multiple comparisons. Negative labels are sampled unreported interactions, not experimentally verified noninteractions. '
        'Lower Brier indicates lower probability error on these benchmark labels and does not by itself establish better AP.', '',
        '## Artifacts', '',
        '![Precision–recall, ROC and probability calibration](results/test-curves.png)', '',
        '![Paired protein-bootstrap metric intervals](results/test-metrics.png)', '',
        '- [Machine-readable comparison](results/benchmark-summary.json)',
        '- [All paired differences](results/paired-differences.csv)',
        '- [All metrics and length strata](results/metrics.csv)',
        '- [Protein-macro definition and results](results/protein-macro-summary.json)',
        '- [Frozen checkpoint selection](provenance/selection.json)',
        '- [Checkpoint forward verification](provenance/qualification.json)',
        '- [Prediction reuse verification](provenance/reuse-verification.json)',
        '- [Benchmark protocol](PROTOCOL.md)', '']
    report_path = ROOT / 'REPORT.md'
    report_path.write_text('\n'.join(lines))
    completed = {'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'benchmark_completed': True, 'scope': 'C0 ESMC standard-attention benchmark',
        'models': NAMES, 'fresh_test_inference_models': [candidate],
        'reused_test_inference_models': NAMES[:-1], 'test_rows_per_model': 52048,
        'selected_update': selected['update'], 'training_horizon_completed': True,
        'prediction_fingerprint': s['prediction_fingerprint'],
        'summary_sha256': sha256(summary_path), 'macro_summary_sha256': sha256(macro_path),
        'report_sha256': sha256(report_path), 'all_coverage_and_numerical_checks_passed': True,
        'c0_descriptive_numerical_practical_gain_criterion_met': numerical_target,
        'original_c1_primary_comparison_completed': False, 'conclusion': conclusion,
        'additional_training_or_model_search_scheduled': False}
    atomic_json(ROOT / 'completed.json', completed)
    finish_closeout(completed)
    print(json.dumps(completed, indent=2), flush=True)


def finish_closeout(completed):
    assert sha256(ROOT / 'REPORT.md') == completed['report_sha256']
    assert sha256(ROOT / 'results/benchmark-summary.json') == completed['summary_sha256']
    assert sha256(ROOT / 'results/protein-macro-summary.json') == completed['macro_summary_sha256']
    (ROOT / 'README.md').write_text('# V4 ESMC benchmark\n\nCompleted. ' + completed['conclusion'] +
        '\n\nSee the [full report](REPORT.md), [machine-readable results](results/benchmark-summary.json), '
        '[protocol](PROTOCOL.md) and [checkpoint selection](provenance/selection.json). '
        'Only ESMC standard attention received new test inference; all eight earlier model predictions were reused.\n')


if __name__ == '__main__':
    main()
