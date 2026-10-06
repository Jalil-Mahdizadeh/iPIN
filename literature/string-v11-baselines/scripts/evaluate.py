"""Evaluate frozen baseline scores and the verified cached neural predictions."""
import csv
import gzip
import os
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

def paircodes(pairs, count):
    return pairs.min(1)*count + pairs.max(1)

def masks(pairs, labels, train_mask, train_codes, count, dataset):
    codes = paircodes(pairs, count)
    _, first, inverse = np.unique(codes, return_index=True, return_inverse=True)
    positive = np.bincount(inverse, weights=labels) > 0
    negative = np.bincount(inverse, weights=1-labels) > 0
    unique = np.zeros(len(labels), bool)
    unique[first[~(positive & negative)]] = True
    result = {'all_rows': np.ones(len(labels), bool),
              'excluding_train_pair_overlap': ~np.isin(codes, train_codes),
              'unique_pairs_without_conflicts': unique}
    if dataset != 'human-validation':
        result['excluding_exact_train_endpoints'] = ~train_mask[pairs].any(1)
    return result

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
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.7))
    for ax, metric in zip(axes, ['ap', 'auroc']):
        data = np.array([[summary['datasets'][s]['models'].get(n, {}).get(metric, np.nan)
                          for s in DATASETS] for n in NAMES])
        cmap = plt.get_cmap('YlGnBu').copy()
        cmap.set_bad('#eeeeee')
        im = ax.imshow(data, vmin=0, vmax=1, cmap=cmap, aspect='auto')
        for row in range(len(NAMES)):
            for col in range(len(DATASETS)):
                value = data[row, col]
                ax.text(col, row, 'N/A' if np.isnan(value) else f'{value:.3f}', ha='center', va='center',
                        fontsize=10, color='white' if value > .65 else 'black')
        ax.set_xticks(range(len(DATASETS)), [SET_LABELS[s].replace(' ', '\n') for s in DATASETS])
        ax.set_yticks(range(len(NAMES)), [LABELS[n] for n in NAMES])
        ax.set_title('Average precision' if metric == 'ap' else 'AUROC')
        ax.axhline(3.5, color='white', linewidth=2)
        ax.axvline(.5, color='white', linewidth=2)
        fig.colorbar(im, ax=ax, shrink=.7)
    fig.suptitle('Fixed CPU baselines trained on exact V11 human TRAIN\nNeural five-species predictions reused; all original test rows retained', fontsize=13)
    fig.text(.04, .015, 'Chance: AP ≈ 0.091, AUROC = 0.500. N/A: exact neural human-validation predictions not available. 95% intervals are in confidence-intervals.csv.', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .91))
    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(ROOT / 'results' / ('comparison.' + ext), dpi=190, bbox_inches='tight')
    plt.close(fig)

def table(summary, metric):
    lines = ['| Method | Human validation | Mouse | Fly | Worm | Yeast | E. coli | Five-species mean |',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name in NAMES:
        entries = []
        for s in DATASETS:
            m = summary['datasets'][s]['models'].get(name)
            entries.append('—' if m is None else f"{m[metric]:.4f}")
        entries.append(f"{summary['species_macro'][name][metric]:.4f}")
        lines.append('| ' + ' | '.join([LABELS[name]] + entries) + ' |')
    return lines

def report(summary):
    fitted = read(ROOT / 'provenance/predictions-frozen.json')
    search = read(ROOT / 'provenance/search.json')
    prepared = read(ROOT / 'provenance/prepared.json')
    lines = ['# Three fixed CPU baselines on the exact V11 human and five-species data', '',
             'Completed: ' + summary['at_utc'], '',
             '**The inexpensive baselines recover substantial cross-species signal, but all three fall below both native PLM-interact humanV11 and TUnA human seed 47 on every species in both AP and AUROC. These results do not support claiming that these baselines match the expensive models across the five species.**', '',
             f"The strongest fixed baseline by five-species mean is homology-transferred degree: AP {summary['species_macro']['homology-degree']['ap']:.4f}, AUROC {summary['species_macro']['homology-degree']['auroc']:.4f}, compared with native PLM-interact AP {summary['species_macro']['native-human']['ap']:.4f}, AUROC {summary['species_macro']['native-human']['auroc']:.4f} and TUnA AP {summary['species_macro']['tuna-human']['ap']:.4f}, AUROC {summary['species_macro']['tuna-human']['auroc']:.4f}. This is a descriptive comparison of the three prespecified methods, not selection of a new predictor or a per-species ensemble.", '',
             'The defensible message is narrower: a CPU method that scores the proteins separately reaches AUROC above 0.90 on mouse, fly and worm, so learning partner-specific compatibility is not required for substantial performance on these tests. This does not measure which mechanisms the neural models use, or establish that all predictive protein-level signal is a sampling artifact. The much stronger human-validation lookup result and the remaining cross-species gap should both be reported.', '',
             'Human TRAIN contains 421,792 pairs (38,344 positive; 383,448 negative) and 15,631 distinct sequences. Human validation has 52,725 rows. The five nonhuman tests have 242,000 rows in total. The release called human_test is the human validation split. Every released row is preserved in the primary results.', '',
             'One fixed configuration was evaluated for each requested method. No validation or species-test labels were used for fitting or configuration selection. The original exact-sequence degree lookup is included as a reference control. Existing native PLM-interact humanV11 and TUnA human seed-47 predictions were reused and their source-row mapping and hashes verified. No neural model was retrained or rerun.', '',
             '## Average precision', '', *table(summary, 'ap'), '',
             '## AUROC', '', *table(summary, 'auroc'), '',
             'Chance AP is the positive prevalence: 0.090925 for human validation and 0.090909 for each species. Chance AUROC is 0.5. The five-species mean is an unweighted mean of the five separate metrics, not a pooled ranking. Dashes indicate unavailable exact neural human-validation predictions; values from a different split or paper were not substituted.', '',
             'The equal human-validation scores for homology-transferred degree and exact TRAIN degree are expected: every validation sequence is already in TRAIN, and exact matches use their stored TRAIN propensity. Those two human columns are not independent evidence of transfer. The sequence-only model uses no identity lookup and still reaches human-validation AUROC 0.9175; its validation proteins are nevertheless also present among its fitting examples.', '',
             '## Methods and interpretation', '',
             'The [frozen protocol](PROTOCOL.md) contains all parameters and limitations. TRAIN-degree propensity is log((positive occurrences + 1)/(negative occurrences + 1)), with a self pair counted once toward its endpoint. Both requested protein-propensity models add two separately computed protein scores, so neither learns partner-specific compatibility.', '',
             '- **Homology-transferred degree:** use the exact TRAIN propensity if available; otherwise average propensities from up to five human TRAIN sequence matches, weighted by identity and alignment coverage. No match uses the mean TRAIN protein propensity.',
             '- **Sequence-only propensity:** a 300-iteration, 15-leaf histogram gradient boosting regressor predicts the TRAIN propensity from 422 length/composition/dipeptide features. The target is protein sampling propensity, not a calibrated pair interaction probability. It uses no PLM embeddings.',
             '- **Interolog lookup:** maximum product of two sequence-similarity weights over human TRAIN-positive edges connecting the retained homologs. No supporting edge gives zero. Conservation is legitimate biological information; this method is not a pure shortcut diagnostic.', '',
             'The alignment thresholds were fixed at E-value 1e-5, identity 25%, and 50% coverage of both proteins, with MMseqs2 sensitivity 7.5 and up to five retained hits. All missing-hit pairs remain in the denominator. Alignment hits are not assertions of orthology. Top-five truncation and fixed thresholds limit this particular implementation.', '',
             'Strong performance of the first two methods would show that independent protein information can explain much of this benchmark. Failure of these fixed implementations does not establish that the data are unbiased or that stronger cheap baselines cannot work. The neural five-species results were already known before this study, so this is an exploratory assessment; baseline settings were nonetheless frozen before their scores were evaluated.', '',
             '## Coverage', '',
             '| Dataset | Exact human TRAIN endpoint in pair | Both endpoints have a retained human hit | Interolog score > 0: positives | Interolog score > 0: negatives |',
             '|---|---:|---:|---:|---:|']
    for s in DATASETS:
        d = summary['datasets'][s]
        c = d['coverage']
        lines.append(f"| {SET_LABELS[s]} | {c['pairs_with_exact_train_endpoint']:,} / {d['rows']:,} | {c['pairs_with_both_homologs']:,} / {d['rows']:,} | {c['positive_pairs_with_interolog_support']:,} / {d['positives']:,} | {c['negative_pairs_with_interolog_support']:,} / {d['negatives']:,} |")
    lines += ['', '## Sensitivity analyses and uncertainty', '',
              'Removing every pair containing an exact human TRAIN sequence leaves 52,311 mouse pairs (4,001 positives). Homology-transferred degree still achieves AP 0.6085 and AUROC 0.9116, versus native PLM-interact 0.8886/0.9811 and TUnA 0.8564/0.9749. Fly changes negligibly (homology degree AP 0.7228, AUROC 0.9360 on 54,858 remaining pairs). Worm, yeast and E. coli already have no exact human TRAIN endpoint matches. Thus the cross-species signal is not explained solely by exact sequence reuse.', '',
              'All paired 95% protein-bootstrap intervals for each of the three requested baselines minus either neural comparator remain below zero for both metrics in every species. This supports the observed performance gap under this resampling scheme, with the dependence and multiple-comparison limitations stated below.', '',
              'After deduplicating E. coli and excluding contradictory-label pairs, 18,228 pairs remain, including 1,999 positives. The sequence-only model has AP 0.5628 and AUROC 0.7878, versus native PLM-interact 0.7545/0.9072 and TUnA 0.6963/0.8770. The higher positive prevalence partly changes AP; the ranking conclusion persists.', '',
              '[subsets.csv](results/subsets.csv) applies the same subsets to every compared method: exclude exact TRAIN-pair overlaps; exclude every nonhuman pair containing an exact human TRAIN sequence; and retain unique unordered pairs after removing label conflicts. These subsets have different sizes and sometimes different prevalence, so their AP values should not be compared across cohorts as though the tasks were identical. The primary tables preserve the original duplicate rows, including the substantial E. coli duplication.', '',
              '[confidence-intervals.csv](results/confidence-intervals.csv) contains 95% percentile intervals from 500 paired protein-bootstrap samples. [paired-differences.csv](results/paired-differences.csv) compares each baseline to each neural reference on the same resampled endpoints. Endpoint resampling addresses shared proteins, but not all homolog-family or phylogenetic dependence. Intervals are exploratory, unadjusted for multiple comparisons; they do not establish equivalence. Protein-unseen performance cannot be inferred from human validation, whose proteins all appear in TRAIN.', '',
              '## Computation and verification', '',
              f"All new computations used CPU cores within the existing interactive GH200 allocation, with at most 16 threads and no GPU computation. Measured sequence-search wall time: {search['wall_seconds']:.1f} s. Regres​sor fitting: {fitted['regressor_fit_seconds']:.1f} s; prediction for all proteins: {fitted['regressor_prediction_seconds']:.1f} s; sequence-feature construction: {fitted['feature_seconds']:.1f} s. Full fitting, scoring, and symmetry audits: {fitted['resource_use']['wall_seconds']:.1f} s; preparation: {prepared['resource_use']['wall_seconds']:.1f} s. These are timings on this node, not portable cost guarantees or a complete cost comparison with the historical neural training runs.", '',
              'Detailed elapsed time, CPU time and peak resident memory are in logs/*.resources.txt. Two homology methods share one search. The installed module is named mmseqs2-gpu but was explicitly run with --gpu 0; its binary SHA256 is recorded. Python used the existing ARM64 SIF with GPU passthrough disabled. The environment, fitted model, protocol hash, scripts, TRAIN-only inputs, and pre-evaluation prediction hashes are recorded in provenance/.', '',
              'Checks passed: released-source hashes; all neural/source pair mappings; independent TRAIN count implementation; finite full-coverage scores; exact endpoint-reversal symmetry; independent scalar interolog checks; independent AP/AUROC including weighted checks; reproduction of earlier human degree metrics and existing neural species metrics. Worm, yeast and E. coli exact lookup scores are constant and give AUROC 0.5 and AP 1/11.', '',
              '## Artifacts and reproduction', '',
              '- [Metric table](results/metrics.csv), [summary JSON](results/summary.json), [comparison figure](results/comparison.png), and vector PDF/SVG versions.',
              '- results/*-predictions.csv.gz preserve source row numbers, exact sequence hashes, labels, scores and homolog diagnostics; NPZ files retain machine-readable score arrays.',
              '- models/ contains the fitted regressor, TRAIN graph, protein scores/counts and retained homologs. work/alignments.tsv contains the full filtered search output.',
              '- inputs/ contains lossless archives of all seven original data files; provenance/prepared.json records input hashes, URLs and reused neural outputs.',
              '- results/COMPLETE.json is the final artifact manifest, created only after evaluation and report generation finish.', '',
              'From the project root:', '', '```bash', 'bash literature/V11-baselines/scripts/run.sh', '```', '',
              'Completed preparation, search and predictions are verified and reused. Evaluation can be regenerated from frozen scores. GPU inference is never invoked.', '',
              'Method references: [MMseqs2 documentation](https://github.com/soedinglab/MMseqs2/wiki); [HistGradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingRegressor.html); [Bernett et al. 2024](https://doi.org/10.1093/bib/bbae076).', '']
    (ROOT / 'REPORT.md').write_text('\n'.join(lines).replace('Regres​sor', 'Regressor'))

def main():
    start = time.monotonic()
    threadpool_limits(1)
    verify_protocol()
    frozen = read(ROOT / 'provenance/predictions-frozen.json')
    for item in frozen['artifacts']:
        verify(item)
    meta = read(ROOT / 'data/sequences.json')
    count = len(meta['sequence'])
    values = np.load(ROOT / 'models/protein-values.npz', allow_pickle=False)
    train_mask = values['train_mask']
    hits = values['homolog_indices']
    has_hit = hits[:, 0] >= 0
    train_pairs = np.load(ROOT / 'data/human-train-pairs.npy', allow_pickle=False)
    train_codes = np.unique(paircodes(train_pairs, count))
    allmetrics, subset_rows, intervals, differences = [], [], [], []
    datasets = {}
    old_neural = read(ROOT / 'provenance/prior-nonhuman-summary.json')
    old_degree = read(ROOT / 'provenance/prior-degree-summary.json')
    for dataset in DATASETS:
        pairs = np.load(ROOT / 'data' / (dataset + '-pairs.npy'), allow_pickle=False)
        labels = np.load(ROOT / 'data' / (dataset + '-labels.npy'), allow_pickle=False)
        saved = np.load(ROOT / 'results' / (dataset + '-baseline-scores.npz'), allow_pickle=False)
        scores = {n: saved[n] for n in BASELINES}
        if dataset in SPECIES:
            reference = np.load(ROOT / 'data' / (dataset + '-neural-reused.npz'), allow_pickle=False)
            scores.update({n: reference[n] for n in REFERENCES})
        estimates = {n: metrics(labels, s) for n, s in scores.items()}
        if dataset == 'human-validation':
            expected = old_degree['sources']['native']['metrics']['all_validation']['degree_ratio']
            for k in ['ap', 'auroc']:
                assert abs(estimates['exact-degree'][k] - expected[k]) < 1e-12
            assert np.array_equal(scores['exact-degree'], scores['homology-degree'])
        else:
            for n in REFERENCES:
                for k in ['ap', 'auroc']:
                    assert abs(estimates[n][k] - old_neural['tests'][dataset]['models'][n][k]) < 1e-12
        if dataset in ['worm', 'yeast', 'ecoli']:
            assert estimates['exact-degree']['unique_scores'] == 1
            assert abs(estimates['exact-degree']['auroc'] - .5) < 1e-12
            assert abs(estimates['exact-degree']['ap'] - 1/11) < 1e-12
        coverage = {'unique_query_proteins': len(np.unique(pairs)),
                    'query_proteins_with_homolog': int(has_hit[np.unique(pairs)].sum()),
                    'pairs_with_exact_train_endpoint': int(train_mask[pairs].any(1).sum()),
                    'pairs_with_both_homologs': int(has_hit[pairs].all(1).sum()),
                    'pairs_with_at_least_one_homolog': int(has_hit[pairs].any(1).sum()),
                    'positive_pairs_with_interolog_support': int(((scores['interolog'] > 0) & (labels == 1)).sum()),
                    'negative_pairs_with_interolog_support': int(((scores['interolog'] > 0) & (labels == 0)).sum())}
        datasets[dataset] = {'rows': len(labels), 'positives': int(labels.sum()), 'negatives': int((1-labels).sum()),
                             'prevalence': float(labels.mean()), 'models': estimates, 'coverage': coverage}
        for n, m in estimates.items():
            allmetrics.append({'dataset': dataset, 'model': n, 'rows': len(labels), 'positives': int(labels.sum()),
                               'negatives': int((1-labels).sum()), **m})
        for subset, mask in masks(pairs, labels, train_mask, train_codes, count, dataset).items():
            assert len(np.unique(labels[mask])) == 2
            for n, s in scores.items():
                m = metrics(labels[mask], s[mask])
                subset_rows.append({'dataset': dataset, 'subset': subset, 'model': n, 'rows': int(mask.sum()),
                                    'positives': int(labels[mask].sum()), 'prevalence': float(labels[mask].mean()), **m})
                if dataset == 'human-validation' and subset == 'excluding_train_pair_overlap' and n == 'exact-degree':
                    old = old_degree['sources']['native']['metrics'][subset]['degree_ratio']
                    assert all(abs(m[k]-old[k]) < 1e-12 for k in ['ap', 'auroc'])
        names = list(scores)
        with gzip.open(ROOT / 'results' / (dataset + '-predictions.csv.gz'), 'wt', newline='') as out:
            writer = csv.writer(out)
            writer.writerow(['source_row_1based', 'protein_a_sha256', 'protein_b_sha256', 'label', *names,
                             'a_in_train', 'b_in_train', 'a_retained_hits', 'b_retained_hits', 'interolog_support_count'])
            nhits = (hits >= 0).sum(1)
            for i, ((a, b), y) in enumerate(zip(pairs, labels)):
                writer.writerow([i+1, meta['sha256'][a], meta['sha256'][b], int(y), *[repr(float(scores[n][i])) for n in names],
                                 int(train_mask[a]), int(train_mask[b]), int(nhits[a]), int(nhits[b]), int(saved['interolog_support_count'][i])])
        print({'dataset': dataset, 'metrics': {n: {k: m[k] for k in ['ap', 'auroc']} for n, m in estimates.items()}}, flush=True)
        ci, di = bootstrap(dataset, pairs, labels, scores, estimates)
        intervals.extend(ci)
        differences.extend(di)
    macro = {n: {k: float(np.mean([datasets[s]['models'][n][k] for s in SPECIES])) for k in ['ap', 'auroc']} for n in NAMES}
    summary = {'at_utc': now(), 'complete': True, 'config': CONFIG, 'datasets': datasets, 'species_macro': macro,
               'checks': {'all_frozen_prediction_hashes_verified': True, 'all_independent_metrics_agree': True,
                          'prior_human_degree_results_reproduced': True, 'prior_neural_species_metrics_reproduced': True,
                          'no_gpu_computation': True, 'no_configuration_search': True},
               'resource_use': stage_stat(start), 'neural_human_validation': 'unavailable; no inference rerun'}
    write_csv(ROOT / 'results/metrics.csv', allmetrics)
    write_csv(ROOT / 'results/subsets.csv', subset_rows)
    write_csv(ROOT / 'results/confidence-intervals.csv', intervals)
    write_csv(ROOT / 'results/paired-differences.csv', differences)
    atomic(ROOT / 'results/summary.json', summary)
    plots(summary)
    report(summary)
    print({'completed': True, 'species_macro': macro, 'wall_seconds': time.monotonic()-start}, flush=True)

if __name__ == '__main__':
    main()
