"""Publish the completed diagnostic and the independent external feasibility gate."""
from io_v8 import ROOT, read, write, now, sha


def value(x):
    return 'unsupported' if x is None else f'{x:.6f}'


def interval(r):
    if r['ci95'] is None:
        return 'not reported: support floor failed'
    return f"[{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}]"


def main():
    verification = read(ROOT / 'results/verification.json')
    if not verification['passed']:
        raise RuntimeError('Numerical verification failed')
    data = read(ROOT / 'results/validation_diagnostics.json')
    external = read(ROOT / 'results/external_feasibility.json')
    primary = data['views'][data['primary']]; support = primary['support']
    lines = ['# V8: frozen partner diagnostics and external-assay feasibility', '',
        f"Partner research screen: **{data['research_screen']['status']}**.",
        f"External assay gate: **{external['status']}**.", '',
        'No models were trained, no new encoder predictions were generated, and no original Bernett TEST data or scores were opened. The pre-existing TEST sequence-hash exclusion list was reused.', '',
        '## Primary validation diagnostics', '',
        f"The unchanged assessment excluding pilot rows contains {support['rows']:,} pairs, {support['positive']:,} positives, {support['participating_proteins']:,} proteins and {support['participating_components']:,} inherited sequence components. This is retrospective analysis of previously examined validation data.", '',
        f"Within-anchor support: {support['anchors']} anchors / {support['anchor_components']} components. Retrieval support: {support['retrieval_anchors']} anchors / {support['retrieval_anchor_components']} components, requiring at least 20 negative-labelled candidates per anchor. Swaps: {support['selected_quartets']} selected from {support['eligible_quartet_candidates']} admissible quartets, spanning {support['quartet_components']} components.", '',
        '| Frozen score / fixed control | Within-anchor concordance | Recall@10 | Swap preference |',
        '|---|---:|---:|---:|']
    for model in data['models']:
        lines.append('| ' + model + ' | ' + ' | '.join(value(primary['metrics'][metric][model]['point'])
                       for metric in ['within_anchor', 'recall@10', 'swap_preference']) + ' |')
    lines += ['', '| Primary contrast | Difference | Paired 95% component interval |', '|---|---:|---|']
    for metric in ['within_anchor', 'recall@10', 'swap_preference']:
        for comparator in ['native', 'quality']:
            r = primary['contrasts'][metric]['vx_minus_' + comparator]
            lines.append(f"| {metric}: Vx − {comparator} | {value(r['point'])} | {interval(r)} |")
    lines += ['', 'The research screen requires both within-anchor and swap improvements of at least 0.010 over native, positive paired lower bounds over native and quality on both, and positive swap lower bounds over composition and 3-mer controls. Every fixed control and the secondary all-assessment view remain in the JSON. A failed screen does not establish equivalence or negate the completed modest Bernett AP gain.', '',
        'A nonconstant additive endpoint score cancels exactly on every selected quartet. All alternatives were existing negative-labelled candidates; none was manufactured or newly scored. Sequence similarity can also favor observed pairings, so swap preference is not evidence of physical interfaces or co-evolution.', '',
        '## External experimental panel', '',
        '| Prespecified population | Pairs | Positives | Confirmed assay non-detections |', '|---|---:|---:|---:|']
    for name, r in external['views'].items():
        lines.append(f"| {name} | {r['pairs']} | {r['positive']} | {r['assay_nondetection']} |")
    lines += ['', 'The primary starts from the historical exposure-excluded Y2H panel, removes exact original Bernett TRAIN/VAL sequence pairs, then conservatively excludes every pair touching a previously evaluated TEST protein hash. That final exclusion is stricter than removing exact TEST pairs and changes the population. Earlier views are transparent attrition counts, not alternative evaluation populations selected after the audit.', '']
    if external['failures']:
        lines += ['Required earlier support floors failed: ' + ', '.join(external['failures']) + '.', '',
                  'External model inference was therefore not run. Current-training homology, new component construction and exact paired-MSA reconstruction were not pursued after the decisive earlier gate. Cache absence is unknown availability; the sequence-length limit is a genuine maximum possible Vx coverage. This is a panel-feasibility result, not a failed external Vx benchmark.']
    else:
        lines += ['The early class and length ceilings pass. The external study remains incomplete pending current-training homology, component independence and exact paired-MSA qualification. No external inference is authorized by these preliminary counts alone.']
    lines += ['', 'Assay non-detection does not establish universal noninteraction. The source used selected candidates and reference sequences rather than verified exact constructs. Even a panel meeting the minimum floors would only support a bounded exploratory transfer study; it need not resolve a small improvement.', '',
        '## Verification and interpretation', '',
        'All reused logits and their sequence/pair identities match the frozen parent records. Independent checks rebuild every quartet selection and cap, all anchor/retrieval points, all reported interval quantiles and paired contrasts, every external exposure exclusion, and 12 full raw bootstrap draws per validation view. Synthetic qualification checks ties, component multiplicities, deterministic sampling and fail-closed duplicate handling.', '',
        'Intervals use 2,000 paired component draws, condition on the fitted models and candidate design, and omit model-selection uncertainty. Retrieval intervals resample anchor components conditional on fixed lists and leave shared-partner dependence. The inherited 40%-identity/80%-coverage graph is heuristic and does not certify unseen biological families.', '',
        'See [validation results](validation_diagnostics.json), [external feasibility](external_feasibility.json), [verification](verification.json), and [the frozen proposal](../../improvment-proposal-v8.md). No outcome-dependent model changes or new experiment follow automatically.']
    with (ROOT / 'results/REPORT.md').open('x') as f:
        f.write('\n'.join(lines) + '\n')
    write(ROOT / 'results/decision.json', {'at_utc': now(), 'partner_screen': data['research_screen'],
        'external_status': external['status'], 'external_failures': external['failures'],
        'further_external_audit_required': not bool(external['failures']),
        'new_training_authorized': False, 'test_accessed': False})


if __name__ == '__main__':
    main()
