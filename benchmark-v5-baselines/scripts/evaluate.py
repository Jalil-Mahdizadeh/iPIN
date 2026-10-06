"""Evaluate the same fixed baselines fitted on v5 TRAIN, without test tuning."""
import csv
import gzip
import time
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from threadpoolctl import threadpool_limits
from common import *

def write_csv(path, rows):
    with Path(path).open('w', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def metrics(labels, scores):
    assert len(scores) == len(labels) and np.isfinite(scores).all()
    ap, auc = float(average_precision_score(labels, scores)), float(roc_auc_score(labels, scores))
    independent = Ranking(labels, scores).compute(np.ones(len(labels)))
    assert np.allclose([ap, auc], independent, atol=1e-12, rtol=1e-12)
    return {'ap': ap, 'auroc': auc, 'unique_scores': len(np.unique(scores)),
            'minimum': float(scores.min()), 'maximum': float(scores.max())}

def bootstrap(dataset, pairs, labels, score, estimates):
    names = list(score)
    ranks = [Ranking(labels, score[n]) for n in names]
    check = np.random.default_rng(177).integers(0, 4, len(labels)).astype(float)
    for name, rank in zip(names, ranks):
        expected = [average_precision_score(labels, score[name], sample_weight=check),
                    roc_auc_score(labels, score[name], sample_weight=check)]
        assert np.allclose(rank.compute(check), expected, atol=1e-12, rtol=1e-12)
    unique, inverse = np.unique(pairs, return_inverse=True)
    endpoints = inverse.reshape(-1, 2)
    selfpair = endpoints[:, 0] == endpoints[:, 1]
    n = len(unique)
    cfg = CONFIG['bootstrap']
    rng = np.random.default_rng(cfg['seed'])
    samples = np.empty((cfg['replicates'], len(names), 2))
    for rep in range(cfg['replicates']):
        multiplicity = np.bincount(rng.integers(0, n, n), minlength=n)
        weights = multiplicity[endpoints[:, 0]].astype(float)*multiplicity[endpoints[:, 1]]
        weights[selfpair] = multiplicity[endpoints[selfpair, 0]]
        for j, rank in enumerate(ranks):
            samples[rep, j] = rank.compute(weights)
        if rep % 250 == 0:
            print({'dataset': dataset, 'bootstrap': rep}, flush=True)
    assert np.isfinite(samples).all()
    npz(ROOT / 'results' / (dataset + '-bootstrap.npz'), samples=samples,
        names=np.array(names), metrics=np.array(['ap', 'auroc']))
    intervals, differences = [], []
    for j, name in enumerate(names):
        for k, metric in enumerate(['ap', 'auroc']):
            low, high = np.quantile(samples[:, j, k], [.025, .975])
            intervals.append({'dataset': dataset, 'model': name, 'metric': metric,
                              'estimate': estimates[name][metric], 'low': float(low), 'high': float(high)})
            for ref in REFERENCES:
                if ref not in names or name not in BASELINES:
                    continue
                rj = names.index(ref)
                low, high = np.quantile(samples[:, j, k]-samples[:, rj, k], [.025, .975])
                differences.append({'dataset': dataset, 'model': name, 'reference': ref, 'metric': metric,
                                    'difference': estimates[name][metric]-estimates[ref][metric],
                                    'low': float(low), 'high': float(high)})
    return intervals, differences


def plots(summary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, metric in zip(axes, ['ap', 'auroc']):
        matrix = np.array([[summary['datasets'][d]['models'].get(n, {}).get(metric, np.nan)
                            for d in DATASETS] for n in NAMES])
        cmap = plt.get_cmap('YlGnBu').copy()
        cmap.set_bad('#eeeeee')
        im = ax.imshow(matrix, vmin=0, vmax=1, aspect='auto', cmap=cmap)
        for i in range(len(NAMES)):
            for j in range(len(DATASETS)):
                value = matrix[i, j]
                ax.text(j, i, 'N/A' if np.isnan(value) else f'{value:.3f}', ha='center', va='center',
                        color='white' if value > .65 else 'black', fontsize=11)
        ax.set_xticks(range(3), ['V5\nvalidation', 'Original\nBernett test', 'V5 ILP\ntest'])
        ax.set_yticks(range(len(NAMES)), [LABELS[n] for n in NAMES])
        ax.axhline(3.5, color='white', lw=2)
        ax.set_title('Average precision' if metric == 'ap' else 'AUROC')
        fig.colorbar(im, ax=ax, shrink=.6)
    fig.suptitle('Same fixed baseline methods, fitted exclusively on v5 TRAIN\nFrozen iPIN and native predictions reused; every evaluation row retained', fontsize=13)
    fig.text(.06, .01, 'All splits: 1:1 positives/negatives; reference AP and AUROC = 0.500. Native Bernett was not evaluated on v5 validation.\nThe two tests share every positive. iPIN validation scores were used for checkpoint selection.', fontsize=9)
    fig.tight_layout(rect=(0, .075, 1, .9))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(ROOT / 'results' / ('comparison.' + ext), dpi=190, bbox_inches='tight')
    plt.close(fig)


def table(summary, metric):
    result = ['| Method | V5 validation | Original Bernett test | V5 ILP test |', '|---|---:|---:|---:|']
    for name in NAMES:
        values = []
        for dataset in DATASETS:
            m = summary['datasets'][dataset]['models'].get(name)
            values.append('—' if m is None else f"{m[metric]:.4f}")
        result.append('| ' + ' | '.join([LABELS[name]] + values) + ' |')
    return result


def report(summary):
    train = summary['training_degree_diagnostic']
    fitted = read(ROOT / 'provenance/predictions-frozen.json')
    search = read(ROOT / 'provenance/search.json')
    all_constant = train['proteins_with_equal_positive_negative_degrees'] == train['proteins']
    lines = ['# The three fixed V11 baseline methods on the v5 data', '', 'Completed: ' + summary['at_utc'], '',
             'The three requested baseline methods were fitted afresh on **v5 TRAIN only**, with the exact configuration used in the previous V11 study. Full v5 validation, original Bernett test and custom ILP-negative test were scored. The previous V11 fitted regressor and graph were not reused. Existing v5 iPIN and native Bernett predictions were reused; no neural model was retrained or rerun.', '',
             '## Findings', '',
             '**All three fixed baselines are near the 0.5 reference on v5 validation and both tests, while the reused iPIN models score substantially higher. This contrasts with their V11 results and is consistent with the different degree balance and protein partitioning. It is evidence about these tested mechanisms, not proof that v5 contains no shortcuts.**', '']
    if all_constant:
        lines += ['**Every TRAIN protein has exactly equal positive and negative degree. Consequently the degree-ratio target is exactly zero for all TRAIN proteins. Homology-transferred degree and sequence-only propensity have a constant target by construction, so this experiment is an explicit check of the removal of that target signal. Their collapse does not establish that every possible sequence-composition classifier would fail.**', '']
    else:
        fraction = train['proteins_with_equal_positive_negative_degrees']/train['proteins']
        lines += [f"TRAIN positive and negative degrees are equal for {train['proteins_with_equal_positive_negative_degrees']:,}/{train['proteins']:,} proteins ({100*fraction:.2f}%). Only {train['proteins']-train['proteins_with_equal_positive_negative_degrees']} proteins retain a degree difference. The log-degree-ratio target has standard deviation {train['target_std']:.8g} and range [{train['target_min']:.8g}, {train['target_max']:.8g}]. These measured targets were retained without modification. Both propensity methods predict this nearly constant target; they are not general-purpose sequence pair classifiers.", '']
    for d in TESTS:
        m = summary['datasets'][d]['models']
        largest_ap = max(m[n]['ap'] for n in BASELINES[1:])
        largest_auc = max(m[n]['auroc'] for n in BASELINES[1:])
        lines.append(f"On {SET_LABELS[d]}, the highest AP among the three requested methods is {largest_ap:.4f} and the highest AUROC is {largest_auc:.4f}; these may belong to different methods and are descriptive, not a selected ensemble. iPIN ESM2 scores {m['ipin-esm2']['ap']:.4f}/{m['ipin-esm2']['auroc']:.4f} (AP/AUROC), and ESMC scores {m['ipin-esmc']['ap']:.4f}/{m['ipin-esmc']['auroc']:.4f}.")
    lines += ['', 'The main metric tables report every method separately. The exact TRAIN-degree lookup is the same reference control used previously and is constant on all three evaluations because their proteins are absent from TRAIN.', '',
              '## Data', '',
              '| Split | Pairs | Positives | Negatives | Distinct sequences |', '|---|---:|---:|---:|---:|']
    prepared = read(ROOT / 'provenance/prepared.json')
    for d in ['train'] + DATASETS:
        c = prepared['counts'][d]
        lines.append(f"| {d} | {c['rows']:,} | {c['positives']:,} | {c['negatives']:,} | {c['sequences']:,} |")
    lines += ['', 'The three protein groups TRAIN, validation and test are exactly sequence-disjoint. All splits contain unique, non-self unordered sequence pairs. No sequences were truncated and no rows were dropped or resampled. Both tests share all 26,024 positives and 1,154 negatives. All 27,178 shared pairs were verified to have identical scores under every compared predictor.', '',
              '## Average precision', '', *table(summary, 'ap'), '', '## AUROC', '', *table(summary, 'auroc'), '',
              'All datasets are balanced, so reference AP and AUROC are 0.5. Raw AP cannot be compared directly with V11 or its five-species tests, which use about 1:10 positives/negatives. Native Bernett has no cached predictions on this v5 validation set; its historical validation set was not substituted. iPIN validation metrics were used to select their checkpoints and are not an independent estimate of test performance.', '',
              '## Methods held fixed', '',
              'See the [frozen protocol](PROTOCOL.md) and [inherited V11 protocol](provenance/original-v11-protocol.md). The source implementation was checked against the completed V11 artifact manifest. The full configuration, degree-counting routine, sequence features, gradient boosting model, homology transfer rule, interolog rule and bootstrap implementation are unchanged. Paths and data adapters were updated for this study.', '',
              '- **Homology-transferred degree:** add the two proteins\' positive/negative TRAIN log-degree propensities, transferred through up to five TRAIN sequence matches; exact matches use stored propensity and no-hit queries use the mean TRAIN propensity.',
              '- **Sequence-only propensity:** the same 422 log-length, amino-acid, ordered-dipeptide and unknown-residue features; 300 iterations of a 15-leaf histogram gradient boosting regressor; predict the same TRAIN log-degree propensity and add the endpoint predictions.',
              '- **Interolog lookup:** maximum product of sequence-match weights over retained homologs connected by a v5 TRAIN-positive edge. No supporting edge scores zero; all such pairs remain in evaluation.', '',
              'MMseqs2 thresholds remain 25% identity, 50% query and target coverage, E-value 1e-5, sensitivity 7.5, and up to five retained hits. This search can find weaker or partial similarities that survived the v5 exclusion criterion of 40% identity over 80% of both sequences. No claim of complete absence of all homology is made. No validation/test label or test-specific annotation was used to fit these methods.', '',
              '## TRAIN degree diagnostic', '',
              f"There are {train['proteins']:,} TRAIN proteins, of which {train['proteins_with_equal_positive_negative_degrees']:,} have equal positive/negative row-occurrence counts. The sum of absolute degree differences is {train['sum_absolute_degree_difference']:,}; maximum absolute difference is {train['max_absolute_degree_difference']:,}. The smoothed log ratio uses +1 in both counts. Its mean is {train['target_mean']:.8g}, standard deviation {train['target_std']:.8g}, and it has {train['unique_targets']:,} unique values. These counts come exclusively from the final actual TRAIN pairs, not the solver objective or requested tolerances.", '',
              '[training-degrees.csv.gz](results/training-degrees.csv.gz) contains every TRAIN protein\'s counts and target. Uniform protein weighting, seed 47 and all other training parameters were retained even if the target carries little variation.', '',
              '## Homolog and interolog coverage', '',
              '| Dataset | Proteins with retained TRAIN match | Pairs with both endpoints matched | Positive pairs with interolog support | Negative pairs with interolog support |',
              '|---|---:|---:|---:|---:|']
    for d in DATASETS:
        c, cov = summary['datasets'][d], summary['datasets'][d]['coverage']
        lines.append(f"| {SET_LABELS[d]} | {cov['proteins_with_homolog']:,}/{cov['proteins']:,} | {cov['pairs_with_both_homologs']:,}/{c['rows']:,} | {cov['positive_pairs_with_interolog_support']:,}/{c['positives']:,} | {cov['negative_pairs_with_interolog_support']:,}/{c['negatives']:,} |")
    lines += ['', 'Coverage is descriptive and computed after scores were frozen. Low interolog coverage can make many scores tie; this limitation is visible rather than removed by evaluating only covered pairs.', '',
              'Interolog lookup retains a small detectable residual: its AUROC intervals are just above 0.5, although the absolute improvement is tiny. Only 200 of the 26,024 shared test positives have supporting retained TRAIN-homolog edges (about 0.77%). Thus near chance does not mean exactly zero signal. The degree and sequence-propensity AUROC intervals include 0.5 on every split.', '',
              '## Uncertainty and interpretation', '',
              'The [95% intervals](results/confidence-intervals.csv) and [paired differences](results/paired-differences.csv) use 500 protein-endpoint bootstrap replicates and seed 20261006, identical to the V11 baseline protocol. Every model receives the same sampled endpoint multiplicities within each dataset. Self pairs would receive one multiplicity; other pairs receive their product. Intervals are descriptive and unadjusted for multiple comparisons. They do not cover training-seed, phylogenetic-family or data-construction uncertainty. No across-test significance calculation treats the shared positives as independent.', '',
              'For all three requested baselines versus each iPIN model, the paired 95% intervals for baseline-minus-iPIN performance lie below zero in both AP and AUROC on every evaluated split. Validation comparisons concern the already selected iPIN checkpoints; the two test comparisons share positives.', '',
              'A low score here shows that these particular baseline mechanisms do not recover the strong V11 signal under this preparation. It does not establish that v5 is unbiased or that iPIN learned binding-site compatibility. In particular, degree-target regression is not a general test of every sequence-only classifier. The V11/v5 comparison changes the training graph, proteins, partitioning, negative construction and prevalence simultaneously, so it cannot isolate the causal contribution of any one change.', '',
              '## Computation and verification', '',
              f"All new computation was CPU-only inside the current interactive allocation, using at most 16 threads. Search wall time: {search['wall_seconds']:.2f} s. Regressor fitting: {fitted['regressor_fit_seconds']:.2f} s; feature construction: {fitted['feature_seconds']:.2f} s; prediction for all proteins: {fitted['regressor_prediction_seconds']:.2f} s. Both homology methods share one search. No production SLURM job or new neural inference was submitted.", '',
              'Evaluation was performed twice after eliminating repeated decompression during CSV export. The first run completed normally; neither the models nor frozen predictions changed. All first-run point metrics, retained in its log, were exactly reproduced. Both runs\' resource records and the export-optimization provenance are retained; this was an output-speed fix, not another scientific configuration.', '',
              'Source bytes and both backbone TRAIN/validation arrays were verified. Cached neural predictions were checked against exact sequence-pair identity, labels and source row order, then their existing metrics were reproduced. Baseline checks include independent TRAIN counts, finite complete predictions, exact AB/BA symmetry, a scalar interolog reference, independent AP/AUROC calculations including weighted bootstrap checks, and shared-test-pair concordance. The container and MMseqs2 binary are pinned by SHA256.', '',
              '## Files and reproduction', '',
              '- [Comparison figure](results/comparison.png), with PDF/SVG copies; [metrics CSV](results/metrics.csv); [summary JSON](results/summary.json).',
              '- results/*-predictions.csv.gz contains source row numbers, sequence hashes, labels, all scores and support diagnostics. Machine-readable NPZ scores are also saved.',
              '- models/ contains the fitted regressor, the v5 TRAIN-positive graph, degree/propensity arrays and retained homologs. work/alignments.tsv stores the filtered search output.',
              '- inputs/ archives the exact source files; provenance/ records source checksums, inherited methods, configuration and the pre-evaluation prediction freeze.',
              '- logs/*.resources.txt records elapsed time, CPU time and peak memory. results/COMPLETE.json is the final artifact manifest.', '',
              'From the project root:', '', '```bash', 'bash benchmark-v5-baselines/scripts/run.sh', '```', '',
              'Completed preparation, search and fitting stages verify and reuse their artifacts. Evaluation reads the frozen predictions. The earlier V11 study and existing v5 benchmarks are not modified.', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines))


def main():
    start = time.monotonic()
    threadpool_limits(1)
    verify_protocol()
    frozen = read(ROOT / 'provenance/predictions-frozen.json')
    for item in frozen['artifacts']:
        verify(item)
    meta = read(ROOT / 'data/sequences.json')
    saved_proteins = np.load(ROOT / 'models/protein-values.npz', allow_pickle=False)
    train_ids = np.array(meta['train_indices'])
    pos, neg = saved_proteins['positive_degree'][train_ids], saved_proteins['negative_degree'][train_ids]
    target = saved_proteins['exact_degree'][train_ids]
    degree_info = {'proteins': len(train_ids), 'proteins_with_equal_positive_negative_degrees': int((pos == neg).sum()),
                  'sum_absolute_degree_difference': int(np.abs(pos-neg).sum()),
                  'max_absolute_degree_difference': int(np.abs(pos-neg).max()),
                  'target_min': float(target.min()), 'target_max': float(target.max()),
                  'target_mean': float(target.mean()), 'target_std': float(target.std()),
                  'unique_targets': len(np.unique(target))}
    with gzip.open(ROOT / 'results/training-degrees.csv.gz', 'wt', newline='') as out:
        writer = csv.writer(out)
        writer.writerow(['sequence_sha256', 'positive_train_degree', 'negative_train_degree', 'log_ratio'])
        for i, p, n, r in zip(train_ids, pos, neg, target):
            writer.writerow([meta['sha256'][i], int(p), int(n), repr(float(r))])
    hits = saved_proteins['homolog_indices']
    has_hit = hits[:, 0] >= 0
    nhits = (hits >= 0).sum(1)
    train_mask = saved_proteins['train_mask']
    previous = read(ROOT / 'provenance/benchmark-summary.json')
    reused = read(ROOT / 'provenance/reused-neural.json')
    allmetrics, intervals, differences, datasets, test_values = [], [], [], {}, {}
    print({'training_degree_diagnostic': degree_info}, flush=True)
    for dataset in DATASETS:
        pairs = np.load(ROOT / 'data' / (dataset + '-pairs.npy'), allow_pickle=False)
        y = np.load(ROOT / 'data' / (dataset + '-labels.npy'), allow_pickle=False)
        pair_ids = read(ROOT / 'data' / (dataset + '-pair-ids.json'))
        baseline = np.load(ROOT / 'results' / (dataset + '-baseline-scores.npz'), allow_pickle=False)
        scores = {n: baseline[n] for n in BASELINES}
        support_counts = baseline['interolog_support_count']
        neural = np.load(ROOT / 'data' / (dataset + '-neural-reused.npz'), allow_pickle=False)
        scores.update({n: neural[n] for n in neural.files})
        assert not train_mask[pairs].any()
        assert np.all(scores['exact-degree'] == 0)
        estimates = {n: metrics(y, z) for n, z in scores.items()}
        assert estimates['exact-degree']['ap'] == .5 and estimates['exact-degree']['auroc'] == .5
        for n in neural.files:
            if dataset in TESTS:
                for k in ['ap', 'auroc']:
                    assert abs(estimates[n][k] - previous['tests'][dataset]['models'][n][k]) < 1e-12
            else:
                assert abs(estimates[n]['ap'] - reused[n + '-validation']['selection_validation_ap']) < 1e-12
        support = scores['interolog'] > 0
        cov = {'proteins': len(np.unique(pairs)), 'proteins_with_homolog': int(has_hit[np.unique(pairs)].sum()),
               'pairs_with_exact_train_endpoint': int(train_mask[pairs].any(1).sum()),
               'pairs_with_both_homologs': int(has_hit[pairs].all(1).sum()),
               'positive_pairs_with_both_homologs': int((has_hit[pairs].all(1) & (y == 1)).sum()),
               'negative_pairs_with_both_homologs': int((has_hit[pairs].all(1) & (y == 0)).sum()),
               'positive_pairs_with_interolog_support': int((support & (y == 1)).sum()),
               'negative_pairs_with_interolog_support': int((support & (y == 0)).sum())}
        counts = {'rows': len(y), 'positives': int(y.sum()), 'negatives': int((1-y).sum())}
        datasets[dataset] = {**counts, 'prevalence': float(y.mean()), 'coverage': cov, 'models': estimates}
        for n, m in estimates.items():
            allmetrics.append({'dataset': dataset, 'model': n, **counts, **m})
        names = list(scores)
        with gzip.open(ROOT / 'results' / (dataset + '-predictions.csv.gz'), 'wt', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['source_row_1based', 'pair_id', 'protein_a_sha256', 'protein_b_sha256', 'label',
                             *names, 'a_retained_hits', 'b_retained_hits', 'interolog_support_count'])
            for i, ((a, b), label) in enumerate(zip(pairs, y)):
                writer.writerow([i+1, pair_ids[i], meta['sha256'][a], meta['sha256'][b], int(label),
                                 *[repr(float(scores[n][i])) for n in names], int(nhits[a]), int(nhits[b]),
                                 int(support_counts[i])])
        print({'dataset': dataset, 'metrics': {n: {k: m[k] for k in ['ap', 'auroc']} for n, m in estimates.items()}}, flush=True)
        ci, di = bootstrap(dataset, pairs, y, scores, estimates)
        intervals.extend(ci)
        differences.extend(di)
        if dataset in TESTS:
            test_values[dataset] = (pairs, y, scores)
    orig_p, orig_y, orig_s = test_values['original']
    ilp_p, ilp_y, ilp_s = test_values['ilp']
    lookup = {tuple(sorted(p)): i for i, p in enumerate(orig_p)}
    oi, ii = [], []
    for j, p in enumerate(ilp_p):
        key = tuple(sorted(p))
        if key in lookup:
            oi.append(lookup[key])
            ii.append(j)
    assert len(oi) == 27178 and np.array_equal(orig_y[oi], ilp_y[ii]) and int(orig_y[oi].sum()) == 26024
    for name in orig_s:
        assert np.array_equal(orig_s[name][oi], ilp_s[name][ii]), name
    summary = {'at_utc': now(), 'complete': True, 'config': CONFIG, 'datasets': datasets,
               'training_degree_diagnostic': degree_info,
               'shared_test_pairs': {'positives': 26024, 'negatives': 1154, 'identical_scores_for_all_models': True},
               'checks': {'all_frozen_prediction_hashes_verified': True, 'independent_metrics_agree': True,
                          'all_reused_neural_metrics_reproduced': True, 'shared_test_pair_scores_identical': True,
                          'exact_degree_control_constant_chance_on_all_evaluations': True,
                          'configuration_identical_to_v11': True, 'no_gpu_computation': True},
               'resource_use': stage_stat(start), 'native_v5_validation': 'not available; historical validation not substituted'}
    write_csv(ROOT / 'results/metrics.csv', allmetrics)
    write_csv(ROOT / 'results/confidence-intervals.csv', intervals)
    write_csv(ROOT / 'results/paired-differences.csv', differences)
    atomic(ROOT / 'results/summary.json', summary)
    plots(summary)
    report(summary)
    print({'evaluation_complete': True, 'wall_seconds': time.monotonic()-start}, flush=True)


if __name__ == '__main__':
    main()
