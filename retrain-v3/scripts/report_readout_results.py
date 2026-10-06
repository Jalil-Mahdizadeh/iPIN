"""Render the audited readout decision and saved validation trajectories."""
import datetime
import hashlib
import json
import statistics as stats
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    decision_path = ROOT / 'decisions/readout-window5000.json'
    audit_path = ROOT / 'provenance/readout-completion-3196827.json'
    decision, audit = [json.loads(p.read_text()) for p in [decision_path, audit_path]]
    assert audit['passed'] and len(audit['runs']) == 6
    assert decision['selected_candidate'] is None and not decision['confirmation_recommended']
    capacity, residue, mechanism = [decision[k] for k in ['cls_capacity_vs_linear', 'residue_vs_linear', 'residue_vs_capacity']]
    groups = {'CLS-linear control': [x['control_metrics'] for x in residue['folds']],
              'CLS-MLP': [x['candidate_metrics'] for x in capacity['folds']],
              'Residue-MLP': [x['candidate_metrics'] for x in residue['folds']]}
    means = {name: {k: stats.mean(v[k] for v in values) for k in ['ap', 'auroc', 'macro_ap', 'brier']}
             for name, values in groups.items()}
    report = [
        '# V3 readout results — 2026-10-01', '',
        '**All six jobs completed their planned screening window successfully. The residue head produced a small, consistent development improvement, but neither candidate met the complete advancement rule.**', '',
        f"Residue-MLP mean selected-checkpoint AP was **{means['Residue-MLP']['ap']:.6f}**, compared with **{means['CLS-linear control']['ap']:.6f}** for the preserved v3 CLS-linear controls: **+{residue['mean_deltas']['ap']:.6f}**, or +{100*residue['mean_deltas']['ap']:.3f} percentage points. The rule fixed before this stage required at least +0.005 mean AP. CLS-MLP changed mean AP by {capacity['mean_deltas']['ap']:+.6f}.", '',
        '## Best checkpoints from the same five-validation window', '',
        'Each model is selected by maximum pooled validation AP at updates 1,000–5,000, with the earliest exact tie retained. AUROC, protein macro-AP and Brier below use that same AP-selected checkpoint. Means give each fold equal weight; AP is average precision, not classification accuracy.', '',
        '| Design | Fold 0 AP | Fold 1 AP | Fold 2 AP | Mean AP | Mean AUROC | Mean protein macro-AP | Mean Brier ↓ |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for label, values in groups.items():
        m = means[label]
        report.append('| ' + label + ' | ' + ' | '.join(f'{v["ap"]:.6f}' for v in values) +
                      ' | ' + ' | '.join(f'{m[k]:.6f}' for k in ['ap', 'auroc', 'macro_ap', 'brier']) + ' |')
    report += ['', 'The reference is our v3 ESM2-initialized, clean-BCE CLS-linear model on each development fold. It is **not the authors’ released native checkpoint**. These results cannot be directly compared with native scores on a different test population.', '',
               '| Array task | Design | Fold | Best update | AP | AUROC | AP change vs CLS-linear | Runtime |',
               '| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |']
    for index, run in enumerate(audit['runs']):
        comparison = capacity if index < 3 else residue
        f = comparison['folds'][index % 3]
        sec = run['accounting']['elapsed_seconds']
        runtime = f'{sec//3600}h {(sec%3600)//60}m {sec%60}s'
        report.append(f"| {run['job_id']} | {'CLS-MLP' if index < 3 else 'Residue-MLP'} | {index%3} | {f['candidate_best_update']:,} | {f['candidate_metrics']['ap']:.6f} | {f['candidate_metrics']['auroc']:.6f} | {f['deltas']['ap']:+.6f} | {runtime} |")
    report += ['', 'CLS-linear control checkpoint updates are 3,000 / 3,000 / 4,000. Each new run retains its selected checkpoint plus update-5,000 fallback and update-5,001 resume state under `runs/<configuration>/checkpoints/`; `best.json` and `latest.json` identify the roles.', '',
               '## Prespecified decision', '',
               '| Comparison | Mean AP change | Positive folds | Failed rules | Decision |',
               '| --- | ---: | ---: | --- | --- |']
    for label, comp in [('CLS-MLP vs CLS-linear', capacity), ('Residue-MLP vs CLS-linear', residue), ('Residue-MLP vs CLS-MLP', mechanism)]:
        failed = ', '.join(k for k, v in comp['checks'].items() if not v) or 'None'
        report.append(f"| {label} | {comp['mean_deltas']['ap']:+.6f} | {comp['positive_folds']}/3 | {failed} | {'Pass' if comp['promoted'] else 'Do not advance'} |")
    report += ['', 'Residue-MLP improved AP, AUROC and protein macro-AP on every fold against CLS-linear, and met the comparison rule against the equally sized CLS-MLP head. It missed only the minimum mean-AP improvement against CLS-linear. Because the residue candidate must pass against **both** controls, the frozen decision selects no candidate for confirmation. Its mean Brier score is slightly worse than CLS-linear, so the ranking gain is not a uniform improvement in all metrics.', '',
               '## Uncertainty and training trajectory', '',
               'The descriptive paired endpoint bootstrap uses 1,000 replicates per fold and comparison. Every individual-fold interval below includes zero. The intervals are conditional on the observed graph and selected checkpoints; they exclude training-seed and model-selection uncertainty. Folds also share training data. Passing a development rule is not a significance test.', '',
               '| Residue-MLP vs CLS-linear | AP change | Conditional 95% bootstrap interval |',
               '| --- | ---: | --- |']
    for f in residue['folds']:
        lo, hi = f['conditional_node_bootstrap_ap_delta_95_ci']
        report.append(f"| {f['fold']} | {f['deltas']['ap']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    report += ['', 'Five of the six new runs peaked at update 3,000; residue fold 2 peaked at 4,000. All six had lower AP and higher Brier at 5,000 than at their selected checkpoint, consistent with late overfitting. Completing more updates under this schedule cannot be assumed to improve validation performance.', '']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = ROOT / 'reports'; figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharex=True, sharey=True, layout='constrained')
    styles = [('cls-linear', 'CLS-linear control', '#354f75', 'o'),
              ('cls-mlp', 'CLS-MLP', '#d08023', 's'), ('residue-mean', 'Residue-MLP', '#008576', '^')]
    curves = {}
    for fold, ax in enumerate(axes):
        for head, label, color, marker in styles:
            out = ROOT / 'runs' / f'clean-{head}-lr2e-5-fold{fold}-seed2'
            updates = [1000, 2000, 3000, 4000, 5000]
            values = [json.loads((out / 'validation' / f'update-{u:09d}.json').read_text())['metrics']['pooled_ap'] for u in updates]
            curves[out.name] = {'updates': updates, 'validation_ap': values}
            ax.plot(updates, values, color=color, marker=marker, lw=1.6, ms=4, label=label)
        ax.set_title(f'Fold {fold}')
        ax.set_xlabel('Optimizer update')
        ax.set_xticks([1000, 3000, 5000])
        ax.grid(alpha=.22); ax.spines[['top', 'right']].set_visible(False)
    all_ap = [v for c in curves.values() for v in c['validation_ap']]
    axes[0].set_ylim(min(all_ap) - .004, max(all_ap) + .006)
    axes[0].set_ylabel('Validation average precision')
    axes[1].legend(loc='lower right', fontsize=8, frameon=False)
    fig.suptitle('V3 readout screen: equal five-checkpoint selection window', fontsize=12)
    for suffix in ['png', 'svg']:
        fig.savefig(figures / f'readout-validation-ap.{suffix}', dpi=180)
    plt.close(fig)
    report += ['![Validation AP trajectories across all three folds](reports/readout-validation-ap.png)', '',
               '## Length-stratified observation', '',
               'These are descriptive results at the already selected checkpoints; they do not select a new model or threshold.', '',
               '| Validation pair length | Fold 0 residue AP gain | Fold 1 gain | Fold 2 gain | Mean gain |',
               '| --- | ---: | ---: | ---: | ---: |']
    for key, label in [('at-most-2193-residues', 'At most 2,193 combined residues'), ('over-2193-residues', 'Over 2,193 combined residues')]:
        delta = [f['length_strata'][key]['candidate']['ap'] - f['length_strata'][key]['control']['ap'] for f in residue['folds']]
        report.append('| ' + label + ' | ' + ' | '.join(f'{x:+.6f}' for x in delta + [stats.mean(delta)]) + ' |')
    report += ['', 'The larger gains against CLS-linear occur among long pairs. CLS-MLP also improves AP in the long-pair stratum on all three folds, so this does not by itself isolate a residue-specific mechanism. These strata have no separate confirmation gate or uncertainty analysis here.', '',
               '## Execution and integrity', '',
               f"All six SLURM tasks exited `COMPLETED`, `0:0`, without restarts, OOM reports or nonfinite-metric failures. Total allocation was **{audit['allocated_gpu_hours']:.2f} GPU-hours**, within the planned 125–225. Maximum logged allocated GPU memory ranged from {min(x['maximum_logged_gpu_allocated_gib'] for x in audit['runs']):.2f} to {max(x['maximum_logged_gpu_allocated_gib'] for x in audit['runs']):.2f} GiB.", '',
               'Every run has exactly the five planned validations. The window watcher stopped each at update **5,001**, one safe-boundary update after validation 5,000; that extra update is excluded from selection. This completed the amended screening window, not the full declared 6,000-update optimizer schedule. All stopped states are committed with no pending validation, and retain the original optimizer/RNG/sampling resume state.', '',
               f"The audit SHA-256 verified all **{len(audit['all_retained_payloads'])} retained checkpoint payloads**, totaling {audit['payload_bytes_verified']/1e9:.2f} GB, including best, fallback and latest state. Frozen release/input/qualification integrity was verified before executing its comparison script. Validation rows, labels, prediction hashes and strict maximum-AP selection were recomputed from saved predictions. No new inference, retraining or historical-test evaluation was needed.", '',
               'The new heads share the independently stored-gradient trainer amendment qualified before submission; original linear controls keep their original bucket-view trainer. That DDP amendment changed only gradient storage, leaving model computation, objective, data and optimizer settings intact. Bitwise equality of numerical trajectories across the two trainer implementations is not asserted. The common window was chosen after the LR screen and before readout production. These are development results; superiority over the native released model remains unestablished.', '',
               'Evidence: [frozen readout protocol](releases/20260930-readout/protocol/READOUT_PROTOCOL.md), [machine-readable decision](decisions/readout-window5000.json), [completion and checkpoint audit](provenance/readout-completion-3196827.json), [SLURM accounting](provenance/readout-accounting-3196827.txt), [comparison log](logs/readout-window5000-comparison.log), [checkpoint audit log](logs/readout-completion-audit-3196827.log). No further production jobs were submitted during this review.', '']
    output = ROOT / 'READOUT_RESULTS.md'; output.write_text('\n'.join(report))
    files = [decision_path, audit_path, Path(__file__), output, figures / 'readout-validation-ap.png', figures / 'readout-validation-ap.svg']
    provenance = {'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'analysis_command': 'bash retrain-v3/scripts/container.sh python -B retrain-v3/releases/20260930-readout/code/compare_readout.py',
                  'analysis_bootstrap_replicates': 1000, 'release_sha256': audit['release_sha256'],
                  'files_sha256': {str(p.relative_to(ROOT)): sha(p) for p in files}, 'means': means, 'curves': curves,
                  'test_evaluated': False, 'production_jobs_submitted': 0}
    (ROOT / 'provenance/readout-review-20261001.json').write_text(json.dumps(provenance, indent=2) + '\n')
    print('Wrote', output, 'and validation plots.')


if __name__ == '__main__':
    main()
