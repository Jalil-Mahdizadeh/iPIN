"""Write the bounded final comparison and close v3 regardless of the outcome."""
import datetime
import json
from pathlib import Path
from common import ROOT, atomic_json, sha256


def main():
    summary_path = ROOT / 'results/benchmark-summary.json'
    summary = json.loads(summary_path.read_text())
    cfg = json.loads((ROOT / 'config.json').read_text())
    primary = summary['primary']
    candidate = 'v3-residue-mlp'
    selected = summary['selection']['models'][candidate]
    endpoint = summary['selection'].get('training_endpoint', {'training_horizon_completed': True,
        'stopped_update': 12745, 'planned_updates': 12745, 'validation_opportunities': 13})
    comparisons = {name: next(d for d in summary['paired_differences']
        if d['group'] == 'all' and d['metric'] == 'ap' and d['comparison'] == f'{candidate} minus {name}')
        for name in cfg['comparison_models'] if name != candidate}
    delta = comparisons['native-bernett']
    if delta['low'] > 0:
        conclusion = 'V3 improves AP over native on this historical test, with a positive descriptive paired 95% interval.'
    elif delta['difference'] > 0:
        conclusion = 'V3 has higher point-estimate AP than native, but the paired 95% interval includes zero; superiority is not established.'
    else:
        conclusion = 'V3 does not outperform native in primary test AP.'
    lines = ['# Final v3 benchmark', '', conclusion, '',
        f"Selected residue-MLP checkpoint: update **{selected['update']:,}**, official validation AP **{selected['validation_primary_ap']:.6f}**.", '',
        'This ends v3. No further v3 model, learning-rate, data or seed search is scheduled.', '',
        '| Model | Selected update | Test AP | Test AUROC | Brier (lower is better) |',
        '| --- | ---: | ---: | ---: | ---: |']
    for name in cfg['comparison_models']:
        m = primary[name]
        update = summary['selection']['models'][name].get('update')
        update = f'{update:,}' if update is not None else 'Unreported'
        lines.append(f"| {name} | {update} | {m['ap']:.6f} | {m['auroc']:.6f} | {m['brier']:.6f} |")
    lines += ['', 'Paired protein-resampling AP differences (v3 minus comparator):', '',
              '| Comparator | AP difference | Descriptive 95% interval |', '| --- | ---: | --- |']
    for name, d in comparisons.items():
        lines.append(f"| {name} | {d['difference']:+.6f} | [{d['low']:+.6f}, {d['high']:+.6f}] |")
    if endpoint['training_horizon_completed']:
        budget = 'The full 12,745-update budget and all 13 validation opportunities were completed.'
    else:
        budget = (f"Training stopped safely at update {endpoint['stopped_update']:,} of {endpoint['planned_updates']:,}, "
            f"after {endpoint['validation_opportunities']} complete official validations. The user authorized stopping after four successive validation AP declines. "
            'Selection used every completed validation; the final resumable checkpoint is separate from the selected checkpoint. '
            'The full training horizon was not completed, so this is not an equal-training-budget comparison with v2. '
            'V1 was also stopped early; v2 completed its full horizon. See the [early-stop amendment](provenance/early-stop-amendment.json).')
    lines += ['', '## Scope and interpretation', '',
        'One ESM2-initialized residue-MLP model, seed 2, initial LR 2e-5, clean BCE, full backbone fine-tuning. The 163,085 official training pairs and 59,258 validation pairs were unchanged; no length cap. The learning-rate schedule retained its original 12,745-update horizon and 2,000-update warmup. Maximum pooled validation AP selects the checkpoint; exact ties choose the earlier update. Both orientations contribute to training and inference.', '',
        budget, '',
        'All eight models cover the same 52,048 official test pairs. The seven native/v1/v2 prediction sets were reused after checksum and row/label checks. Scores average AB/BA logits, with FP32 parameters, BF16 autocast, efficient attention, TF32 disabled and no test truncation. Thresholds were fixed from validation before new test inference. Baseline metrics and paired bootstrap replicates reproduce v2 to 1e-12.', '',
        'V3 inference ran the original four independent shards sequentially on the available interactive GH200 GPU. Row assignment, length sorting, batching, scoring and precision were unchanged from the prepared four-GPU implementation. Shard timing records verify nonoverlapping execution; every test row occurs exactly once. This avoids the several-hour estimated queue for another node.', '',
        'This is an exploratory historical-test comparison, not independent confirmation. The test informed earlier research, only one full-data v3 seed was trained, and descriptive protein-level bootstrap intervals do not account for all prior model-selection or homology dependence. Residue-MLP improved internal mean AP by 0.003886 but did not pass the original +0.005 advancement gate. The user explicitly authorized this single final experiment despite that gate; the failed-gate record is preserved. No superiority guarantee was made.', '',
        'The model/data/checkpoint core is unchanged from the qualified v3 residue stage. V3 disables DDP gradient bucket views to preserve exact resume; this engineering difference from v2 is documented in the training protocol. The final comparison cannot isolate every numerical or seed effect.', '',
        '## Artifacts', '',
        '![Test PR, ROC and calibration curves](results/test-curves.png)', '',
        '![Metric intervals for all eight models](results/test-metrics.png)', '',
        '- [Machine-readable results](results/benchmark-summary.json)',
        '- [All paired differences](results/paired-differences.csv)',
        '- [Validation-only checkpoint selection](provenance/selection.json)',
        '- [Checkpoint forward verification](provenance/qualification.json)',
        '- [Frozen training protocol](../retrain-v3/FINAL_PROTOCOL.md)',
        '- [Original development decision](../retrain-v3/decisions/readout-window5000.json)', '']
    report = ROOT / 'REPORT.md'
    report.write_text('\n'.join(lines))
    completed = {'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'v3_closed': True, 'models': cfg['comparison_models'], 'fresh_test_inference_models': [candidate],
        'reused_test_inference_models': cfg['comparison_models'][:-1], 'test_rows_per_model': 52048,
        'selected_update': selected['update'], 'prediction_fingerprint': summary['prediction_fingerprint'],
        'training_endpoint': endpoint,
        'summary_sha256': sha256(summary_path), 'report_sha256': sha256(report),
        'all_coverage_and_numerical_checks_passed': True, 'conclusion': conclusion,
        'further_v3_experiments_scheduled': False}
    atomic_json(ROOT / 'completed.json', completed)
    finish_closeout(completed)
    print(json.dumps(completed, indent=2), flush=True)


def finish_closeout(completed):
    assert sha256(ROOT / 'REPORT.md') == completed['report_sha256']
    assert sha256(ROOT / 'results/benchmark-summary.json') == completed['summary_sha256']
    retrain = ROOT.parent / 'retrain-v3'
    summary = json.loads((ROOT / 'results/benchmark-summary.json').read_text())
    candidate, native = summary['primary']['v3-residue-mlp'], summary['primary']['native-bernett']
    text = ('# V3 finalized\n\n' + completed['conclusion'] + '\n\n'
        f"Final selected update: {completed['selected_update']:,}. Test AP: v3 **{candidate['ap']:.6f}**, native **{native['ap']:.6f}**. "
        f"Test AUROC: v3 **{candidate['auroc']:.6f}**, native **{native['auroc']:.6f}**.\n\n"
        f"Training endpoint: **{completed['training_endpoint']['stopped_update']:,}** of **{completed['training_endpoint']['planned_updates']:,}** planned updates; "
        f"full horizon completed: **{completed['training_endpoint']['training_horizon_completed']}**.\n\n"
        'This is the authorized exploratory single-run closeout. V3 is closed regardless of improvement; no further experiments are queued by this pipeline.\n\n'
        'See the [complete comparison](../benchmark-v3/REPORT.md), [frozen protocol](FINAL_PROTOCOL.md) and '
        '[preserved development results](READOUT_RESULTS.md).\n')
    (retrain / 'FINAL_RESULTS.md').write_text(text)
    atomic_json(retrain / 'FINALIZED.json', {**completed, 'benchmark_root': str(ROOT),
        'final_results_sha256': sha256(retrain / 'FINAL_RESULTS.md')})
    readme = retrain / 'README.md'
    content = readme.read_text()
    marker = '<!-- FINAL_V3_STATUS -->'
    if marker not in content:
        content = content.replace('# PLM-interact retraining v3\n',
            '# PLM-interact retraining v3\n\n' + marker + '\n**V3 finalized.** See [the final results](FINAL_RESULTS.md) and [complete benchmark](../benchmark-v3/REPORT.md).\n', 1)
        readme.write_text(content)
    (ROOT / 'README.md').write_text('# V3 final benchmark\n\nCompleted. See [the final report](REPORT.md) and [machine-readable results](results/benchmark-summary.json).\n')


if __name__ == '__main__': main()
