"""Apply the preregistered three-fold v3 gates; never read historical test scores."""
import argparse
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from data import PairData
from metrics import evaluate, promotion
from state import atomic_json, sha256

ROOT = Path('/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v3')


def load_selected(name):
    run = ROOT / 'runs' / name
    if not (run / 'completed.json').exists():
        raise RuntimeError(f'{name}: all prespecified updates must finish before the stage decision')
    contract = json.loads((run / 'contract.json').read_text())
    completed = json.loads((run / 'completed.json').read_text())
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    assert contract['configuration'] == cfg and not contract['qualification']
    assert cfg['partition'] in ['fold-0', 'fold-1', 'fold-2']
    assert cfg['initialization'] == 'esm2_pretrained' and not cfg['classification_corruption']
    assert cfg['mlm_weight'] == 0. and cfg['total_updates'] == 6000
    assert contract['code'] == {n: sha256(ROOT / 'scripts' / n) for n in contract['code']}
    assert contract['data_manifest_sha256'] == sha256(ROOT / 'data/prepared/manifest.json')
    assert completed['fingerprint'] == contract['fingerprint'] and not completed['qualification']
    assert len(set(completed['model_hashes_by_rank'])) == 1
    state = completed['state']
    assert state['update'] == cfg['total_updates'] == state['last_validation_update']
    assert state['examples_seen'] == cfg['total_updates'] * cfg['global_pairs_per_update']
    assert not state['pending_validation']
    best = json.loads((run / 'best.json').read_text())
    assert best['fingerprint'] == contract['fingerprint'] and best['update'] == state['best_update']
    data = PairData(ROOT / 'data/prepared', cfg['partition'], 'val')
    # Reconstruct the actual strict-maximum checkpoint selection, including ties.
    trajectory, first_best, max_ap = [], None, -np.inf
    evidence = {str((run / 'contract.json').relative_to(ROOT)): sha256(run / 'contract.json'),
                str((run / 'completed.json').relative_to(ROOT)): sha256(run / 'completed.json'),
                str((run / 'best.json').relative_to(ROOT)): sha256(run / 'best.json')}
    for update in range(cfg['validate_every_updates'], cfg['total_updates'] + 1, cfg['validate_every_updates']):
        file = run / 'validation' / f'update-{update:09d}.npz'
        meta_file = file.with_suffix('.json')
        meta = json.loads(meta_file.read_text())
        assert meta['fingerprint'] == contract['fingerprint'] and meta['update'] == update
        assert meta['rows'] == len(data) and sha256(file) == meta['sha256']
        with np.load(file) as saved:
            preds = saved['predictions']
        assert preds.shape == (len(data), 4) and np.isfinite(preds).all()
        assert np.array_equal(preds[:, 0], np.arange(len(data)))
        assert np.array_equal(preds[:, 1], data.rows[:, 2])
        ap = float(average_precision_score(preds[:, 1], preds[:, 2:4].mean(1)))
        assert abs(ap - meta['metrics']['pooled_ap']) < 1e-12
        trajectory.append({'update': update, 'ap': ap})
        if ap > max_ap:
            max_ap, first_best, selected = ap, update, preds
        evidence[str(file.relative_to(ROOT))] = meta['sha256']
        evidence[str(meta_file.relative_to(ROOT))] = sha256(meta_file)
    assert first_best == best['update'] and abs(max_ap - best['best_ap']) < 1e-12
    return {'name': name, 'config': cfg, 'rows': data.rows, 'lengths': data.lengths,
            'scores': selected[:, 2:4].mean(1), 'best_update': first_best,
            'metrics': evaluate(data.rows, selected[:, 2:4].mean(1)), 'trajectory': trajectory,
            'best_at_horizon': first_best == cfg['total_updates'],
            'last_evaluation_ap_gain': trajectory[-1]['ap'] - trajectory[-2]['ap'], 'evidence': evidence}


def paired_comparison(control, candidate, bootstrap):
    assert np.array_equal(control['rows'], candidate['rows'])
    different = {k for k in control['config'] if control['config'][k] != candidate['config'][k]}
    assert different <= {'name', 'arm', 'learning_rate', 'readout'}, different
    assert not {'learning_rate', 'readout'} <= different, 'Do not confound adaptation and readout comparisons'
    ca, cb = control['metrics'], candidate['metrics']
    assert ca['macro_eligible_proteins'] == cb['macro_eligible_proteins'] > 0
    deltas = {k: cb[k] - ca[k] for k in ['ap', 'auroc', 'macro_ap', 'brier']}
    rows, a, b = control['rows'], control['scores'], candidate['scores']
    y = rows[:, 2]
    proteins, endpoints = np.unique(rows[:, :2], return_inverse=True)
    endpoints = endpoints.reshape(-1, 2)
    rng = np.random.default_rng(20260930)
    boot = []
    for _ in range(bootstrap):
        mult = rng.multinomial(len(proteins), np.full(len(proteins), 1 / len(proteins)))
        weights = mult[endpoints[:, 0]] * mult[endpoints[:, 1]]
        if len(np.unique(y[weights > 0])) == 2:
            boot.append(average_precision_score(y, b, sample_weight=weights) -
                        average_precision_score(y, a, sample_weight=weights))
    assert len(boot) >= .95 * bootstrap
    strata = {}
    for name, mask in [('at-most-2193-residues', control['lengths'] <= 2196),
                       ('over-2193-residues', control['lengths'] > 2196)]:
        if len(np.unique(y[mask])) == 2:
            strata[name] = {'control': evaluate(rows[mask], a[mask]), 'candidate': evaluate(rows[mask], b[mask])}
    return {'fold': control['config']['partition'], 'control': control['name'], 'candidate': candidate['name'],
            'control_best_update': control['best_update'], 'candidate_best_update': candidate['best_update'],
            'control_metrics': ca, 'candidate_metrics': cb, 'deltas': deltas, 'length_strata': strata,
            'conditional_node_bootstrap_ap_delta_95_ci': np.quantile(boot, [.025, .975]).tolist(),
            'bootstrap_replicates': len(boot), 'best_at_horizon': candidate['best_at_horizon'],
            'last_evaluation_ap_gain': candidate['last_evaluation_ap_gain']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['adaptation', 'readout'], required=True)
    parser.add_argument('--output')
    parser.add_argument('--bootstrap', type=int, default=1000)
    args = parser.parse_args()
    assert 200 <= args.bootstrap <= 10000
    output = Path(args.output).resolve() if args.output else ROOT / 'decisions' / f'{args.stage}.json'
    assert output.is_relative_to(ROOT / 'decisions') and not output.exists(), 'Keep every recorded decision'
    campaign_path = ROOT / 'configs/campaign.json'
    campaign = json.loads(campaign_path.read_text())
    evidence = {'configs/campaign.json': sha256(campaign_path)}
    cache = {}

    def load(head, tag, fold):
        name = f'clean-{head}-{tag}-fold{fold}-seed2'
        if name not in cache:
            cache[name] = load_selected(name)
            evidence.update(cache[name]['evidence'])
        return cache[name]

    def compare(c_head, c_tag, h, tag):
        comparisons = [paired_comparison(load(c_head, c_tag, f), load(h, tag, f), args.bootstrap) for f in range(3)]
        return {'folds': comparisons, **promotion([x['deltas'] for x in comparisons], campaign['development_promotion'])}

    if args.stage == 'adaptation':
        result = compare('cls-linear', 'lr2e-5', 'cls-linear', 'lr5e-6')
        tag = 'lr5e-6' if result['promoted'] else 'lr2e-5'
        decision = {'comparison': result, 'selected_learning_rate_tag': tag,
                    'recommended_next_templates': campaign['stage2_templates_by_learning_rate'][tag],
                    'horizon_caution': any(x['best_at_horizon'] or x['last_evaluation_ap_gain'] > 0 for x in result['folds']),
                    'non_promotion_meaning': 'Retain the control for Stage 2; a still-improving low-LR trajectory is inconclusive. No automatic extension or restart with a changed schedule.'}
    else:
        prior_path = ROOT / 'decisions/adaptation.json'
        prior = json.loads(prior_path.read_text())
        assert prior['stage'] == 'adaptation' and prior['test_evaluated'] is False
        for name, digest in prior['evidence_sha256'].items():
            assert sha256(ROOT / name) == digest, ('Prior adaptation decision input changed', name)
        tag = prior['selected_learning_rate_tag']
        capacity = compare('cls-linear', tag, 'cls-mlp', tag)
        residue = compare('cls-linear', tag, 'residue-mean', tag)
        mechanism = compare('cls-mlp', tag, 'residue-mean', tag)
        options = [(capacity['mean_deltas']['ap'], 'cls_mlp')] if capacity['promoted'] else []
        if residue['promoted'] and mechanism['promoted']:
            options.append((residue['mean_deltas']['ap'], 'residue_mean'))
        # Stable iteration puts the simpler CLS-only control first on exact ties.
        chosen = max(options, key=lambda item: item[0])[1] if options else None
        decision = {'cls_capacity_vs_linear': capacity, 'residue_vs_linear': residue,
                    'residue_vs_capacity': mechanism, 'selected_learning_rate_tag': tag,
                    'selected_readout_candidate': chosen,
                    'confirmation_recommended': bool(chosen or prior['comparison']['promoted']),
                    'confirmation_candidate': chosen or ('cls_linear' if prior['comparison']['promoted'] else None),
                    'confirmation_blocker': 'New external development/test panel has not passed provenance/exposure audit; a separate frozen release is required.'}
        evidence[str(prior_path.relative_to(ROOT))] = sha256(prior_path)
    evidence.update({f'scripts/{n}': sha256(ROOT / 'scripts' / n) for n in ['compare_development.py', 'metrics.py']})
    atomic_json(output, {'stage': args.stage, **decision, 'test_evaluated': False, 'jobs_submitted': 0,
                        'evidence_sha256': evidence,
                        'uncertainty_scope': 'Paired multinomial endpoint bootstrap, conditional on this graph/checkpoints. No training-seed or model-selection uncertainty; folds share training data.',
                        'next_stage_launch_policy': 'Decision report does not submit jobs or enable templates. Review and qualify a separate release.'})
    print(json.dumps(decision, indent=2))


if __name__ == '__main__':
    main()
