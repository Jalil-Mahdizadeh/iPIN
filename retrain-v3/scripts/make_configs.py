"""Generate the staged v3 campaign. No scheduler calls; never overwrite a release."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def write(name, value):
    (ROOT / 'configs' / name).write_text(json.dumps(value, indent=2) + '\n')


def main():
    assert not (ROOT / 'releases/CURRENT').exists(), 'Configuration generation is closed after freezing'
    base = json.loads((ROOT.parent / 'retrain-v2/configs/clean-bce-fold0-seed2.json').read_text())
    base.update(root=str(ROOT), classification_corruption=False, mlm_weight=0.,
                positive_weight=1., readout='cls_linear', attention_mode='standard')
    stage1, stage2 = [], {}
    for tag, lr in [('lr2e-5', 2e-5), ('lr5e-6', 5e-6)]:
        for head in ['cls_linear', 'cls_mlp', 'residue_mean']:
            arm = f'clean-{head.replace("_", "-")}-{tag}'
            names = []
            for fold in range(3):
                name = f'{arm}-fold{fold}-seed2'
                cfg = {**base, 'name': name, 'arm': arm, 'learning_rate': lr,
                       'readout': head, 'partition': f'fold-{fold}'}
                write(name + '.json', cfg)
                names.append(name)
            if head == 'cls_linear':
                stage1.extend(names)
            else:
                stage2.setdefault(tag, []).extend(names)
    optional = {**base, 'name': 'clean-capped-official-seed2', 'arm': 'clean-capped',
                'partition': 'official', 'total_updates': 12745, 'warmup_updates': 2000,
                'train_cap_residues': 2193, 'learning_rate': 2e-5}
    write(optional['name'] + '.json', optional)
    qual = {**base, 'name': 'qualification', 'arm': 'qualification', 'learning_rate': 5e-6,
            'global_pairs_per_update': 64, 'warmup_updates': 1, 'validate_every_updates': 2,
            'checkpoint_every_updates': 1, 'log_every_updates': 1}
    write('qualification.json', qual)
    write('campaign.json', {
        'version': 3, 'initial_group': 'adaptation', 'initial_runs': stage1,
        'stage2_templates_by_learning_rate': stage2,
        'optional_disabled_runs': [optional['name']],
        'stage3': {'status': 'not configured: requires development decision and exposure-audited panel',
                   'training_seeds': [2, 17, 42], 'candidate_and_matched_control_runs': 6},
        'production_submission_authorized_now': False,
        'staging': 'Only six adaptation runs can enter the initial release. No automatic later-stage submission.',
        'development_promotion': {
            'minimum_mean_ap_delta': .005, 'minimum_positive_folds': 2,
            'maximum_single_fold_ap_drop': .01, 'maximum_mean_auroc_drop': .005,
            'maximum_mean_macro_ap_drop': .005,
            'macro_minimum_positives_per_protein': 2, 'macro_minimum_negatives_per_protein': 2},
        'adaptation_selection': 'Choose 5e-6 only if all gates pass against 2e-5; otherwise retain 2e-5 for the readout screen. Inspect horizon trends before interpreting a non-promotion.',
        'readout_selection': 'Require gates against CLS-linear; a residue-mechanism claim additionally requires gates against matched CLS-MLP. Select largest mean AP gain among passing candidates, ties prefer CLS-MLP.',
        'checkpoint_selection': 'Maximum pooled fold-validation AP, strict greater-than, earlier update on exact ties; evaluate every 1000 updates and final 6000.',
        'historical_test_policy': 'Exploratory only; absent from this package and unavailable to development selection.',
        'external_confirmation': {'minimum_ap_delta': .01, 'paired_ci_lower_above_zero': True,
                                  'status': 'not ready; external provenance/exposure audit required'},
        'stage1_budget_gpu_hours': [150, 270], 'core_budget_with_contingency_gpu_hours': [1000, 1500]
    })
    print(f'Prepared {len(stage1)} initial runs, 12 gated readout templates and 1 disabled optional arm.')


if __name__ == '__main__':
    main()
