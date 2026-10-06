"""Check preserved controls, unchanged optimization, and common-window analysis."""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import compare_readout as comparator
from metrics import evaluate
from selection_window import load_window
from state import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    policy = json.loads((ROOT / 'configs/readout-stage.json').read_text())
    prior = json.loads((ROOT / 'decisions/adaptation-early.json').read_text())
    assert sha256(ROOT / 'decisions/adaptation-early.json') == policy['adaptation_decision_sha256']
    assert prior['selected_learning_rate'] == policy['learning_rate'] == 2e-5
    assert prior['completed_prespecified_6000_update_comparison'] is False
    assert policy['common_validation_updates'] == [1000, 2000, 3000, 4000, 5000]
    assert policy['screen_stop_update'] == 5000 and policy['optimizer_schedule_updates'] == 6000
    assert policy['pair_exposures_at_comparison_window'] == 320000
    for fold in range(3):
        base_name = f'clean-cls-linear-lr2e-5-fold{fold}-seed2'
        baseline = json.loads((ROOT / 'configs' / (base_name + '.json')).read_text())
        loaded = load_window(base_name)
        assert loaded['actual_stopped_update'] >= 5000 and len(loaded['trajectory']) == 5
        for head, arm in [('cls_mlp', 'cls-mlp'), ('residue_mean', 'residue-mean')]:
            name = f'clean-{arm}-lr2e-5-fold{fold}-seed2'
            cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
            expected = {**baseline, 'name': name, 'arm': f'clean-{arm}-lr2e-5', 'readout': head}
            assert cfg == expected and name in policy['enabled_runs'], name
    assert len(policy['enabled_runs']) == 6
    try:
        load_window('clean-cls-linear-lr2e-5-fold0-seed2', require_screen=True)
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('Readout completion marker was not required')
    try:
        load_window('clean-cls-linear-lr2e-5-fold0-seed2', end=4000)
    except AssertionError:
        pass
    else:
        raise AssertionError('Wrong completed comparison window was accepted')
    fixture = ROOT / 'qualification/readout-aligned-decision-fixture'
    assert not fixture.exists()
    (fixture / 'decisions').mkdir(parents=True); (fixture / 'configs').mkdir()
    (fixture / 'configs/campaign.json').write_bytes((ROOT / 'configs/campaign.json').read_bytes())
    atomic_json(fixture / 'decisions/adaptation-early.json', {
        'selected_learning_rate': 2e-5, 'common_validation_updates': [1000, 2000, 3000, 4000, 5000],
        'evidence_sha256': {}})
    rows = np.array([[10*g, 10*g+i+1, i % 2, 8*g+i] for g in range(12) for i in range(8)])
    scores = np.tile(np.arange(8, dtype=float), 12)
    def fake(name, require_screen=False):
        cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
        assert require_screen == (cfg['readout'] != 'cls_linear')
        return {'name': name, 'config': cfg, 'rows': rows, 'lengths': np.full(len(rows), 100),
                'scores': scores, 'metrics': evaluate(rows, scores), 'best_update': 3000,
                'best_at_horizon': False, 'last_evaluation_ap_gain': -.001, 'evidence': {}}
    with patch.object(comparator, 'ROOT', fixture), patch.object(comparator, 'load_window', side_effect=fake), \
         patch.object(sys, 'argv', ['compare_readout.py', '--bootstrap', '200']):
        comparator.main()
    result = json.loads((fixture / 'decisions/readout-window5000.json').read_text())
    assert result['selected_candidate'] is None and not result['confirmation_recommended']
    assert result['checkpoint_opportunities_per_run'] == 5 and not result['test_evaluated']
    files = ['scripts/selection_window.py', 'scripts/record_adaptation_decision.py',
             'scripts/compare_readout.py', 'scripts/qualify_readout_aligned_protocol.py',
             'configs/readout-stage.json', 'decisions/adaptation-early.json']
    files += [f'configs/{name}.json' for name in policy['enabled_runs']]
    atomic_json(ROOT / 'qualification/readout-aligned-protocol.json', {
        'passed': True, 'controlled_configurations': 6, 'unchanged_schedule_updates': 6000,
        'common_selection_window_end': 5000, 'equal_checkpoint_opportunities': 5,
        'all_three_preserved_controls_verified': True, 'configuration_differences_only_readout': True,
        'engine_amendment': 'Readout trainer uses independent gradient storage; controls retain their immutable original release.',
        'readout_completion_required': True, 'wrong_window_rejected': True,
        'synthetic_equal_predictions_do_not_promote': True,
        'scope': 'Real stopped controls and all six configs; synthetic end-to-end readout decision plumbing.',
        'source_sha256': {n: sha256(ROOT / n) for n in files}})
    print('Readout protocol and decision qualification passed.')


if __name__ == '__main__':
    main()
