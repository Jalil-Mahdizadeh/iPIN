"""Compare all readout arms in the same five-checkpoint development window."""
import argparse
import json
from pathlib import Path
from compare_development import paired_comparison
from metrics import promotion
from selection_window import load_window
from state import atomic_json, sha256

ROOT = Path('/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v3')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--bootstrap', type=int, default=1000)
    args = p.parse_args()
    assert 200 <= args.bootstrap <= 10000
    target = ROOT / 'decisions/readout-window5000.json'
    assert not target.exists(), 'Preserve recorded decisions'
    prior_path = ROOT / 'decisions/adaptation-early.json'
    prior = json.loads(prior_path.read_text())
    assert prior['selected_learning_rate'] == 2e-5
    assert prior['common_validation_updates'] == [1000, 2000, 3000, 4000, 5000]
    for name, digest in prior['evidence_sha256'].items():
        assert sha256(ROOT / name) == digest, ('Adaptation evidence changed', name)
    campaign = json.loads((ROOT / 'configs/campaign.json').read_text())
    cache, evidence = {}, {str(prior_path.relative_to(ROOT)): sha256(prior_path)}
    def load(head, fold):
        name = f'clean-{head}-lr2e-5-fold{fold}-seed2'
        if name not in cache:
            cache[name] = load_window(name, require_screen=head != 'cls-linear')
            evidence.update(cache[name]['evidence'])
        return cache[name]
    def compare(left, right):
        folds = [paired_comparison(load(left, f), load(right, f), args.bootstrap) for f in range(3)]
        return {'folds': folds, **promotion([x['deltas'] for x in folds], campaign['development_promotion'])}
    capacity = compare('cls-linear', 'cls-mlp')
    residue = compare('cls-linear', 'residue-mean')
    mechanism = compare('cls-mlp', 'residue-mean')
    options = [(capacity['mean_deltas']['ap'], 'cls_mlp')] if capacity['promoted'] else []
    if residue['promoted'] and mechanism['promoted']:
        options.append((residue['mean_deltas']['ap'], 'residue_mean'))
    chosen = max(options, key=lambda x: x[0])[1] if options else None
    report = {'stage': 'readout', 'learning_rate': 2e-5, 'selection_window_end': 5000,
              'checkpoint_opportunities_per_run': 5, 'optimizer_schedule_updates': 6000,
              'cls_capacity_vs_linear': capacity, 'residue_vs_linear': residue,
              'residue_vs_capacity': mechanism, 'selected_candidate': chosen,
              'confirmation_recommended': chosen is not None,
              'confirmation_blocker': 'External provenance/exposure audit and a separate seed-confirmation release are still required.',
              'evidence_sha256': evidence, 'test_evaluated': False, 'jobs_submitted': 0,
              'limitation': 'Development window chosen after the LR screen and frozen before readout training; conditional bootstrap excludes training-seed/model-selection uncertainty.'}
    atomic_json(target, report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
