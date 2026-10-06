"""Exercise scientific decision plumbing with synthetic outputs, never real test data."""
import copy
import json
import sys
from pathlib import Path
from unittest.mock import patch
import numpy as np
import compare_development as decision
from metrics import evaluate
from state import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    fixture = ROOT / 'qualification/decision-fixture-v2'
    assert not fixture.exists()
    (fixture / 'decisions').mkdir(parents=True)
    (fixture / 'configs').mkdir()
    (fixture / 'scripts').mkdir()
    for name in ['compare_development.py', 'metrics.py']:
        (fixture / 'scripts' / name).write_bytes((ROOT / 'scripts' / name).read_bytes())
    (fixture / 'configs/campaign.json').write_bytes((ROOT / 'configs/campaign.json').read_bytes())
    try:
        decision.load_selected('not-a-completed-run')
    except RuntimeError as e:
        assert 'must finish' in str(e)
    else:
        raise AssertionError('An incomplete run was accepted')
    # Multiple components avoid degeneracy when a bootstrap omits a star's hub.
    # The first fixture intentionally exposed the <95% usable-replicates guard;
    # preserve its failed log/fixture and exercise orchestration on this graph.
    rows = np.array([[10 * g, 10 * g + i + 1, i % 2, 8 * g + i]
                     for g in range(12) for i in range(8)])
    scores = np.tile(np.arange(8, dtype=float), 12)

    def fake_load(name):
        cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
        return {'name': name, 'config': cfg, 'rows': rows, 'lengths': np.full(len(rows), 100),
                'scores': scores, 'metrics': evaluate(rows, scores), 'best_update': 1000,
                'best_at_horizon': False, 'last_evaluation_ap_gain': -.001, 'evidence': {}}

    # Complete orchestration: identical predictions must not promote a candidate.
    with patch.object(decision, 'ROOT', fixture), patch.object(decision, 'load_selected', side_effect=fake_load), \
         patch.object(sys, 'argv', ['compare_development.py', '--stage', 'adaptation', '--bootstrap', '200']):
        decision.main()
    first = json.loads((fixture / 'decisions/adaptation.json').read_text())
    assert not first['comparison']['promoted'] and first['selected_learning_rate_tag'] == 'lr2e-5'
    assert len(first['recommended_next_templates']) == 6 and first['jobs_submitted'] == 0
    with patch.object(decision, 'ROOT', fixture), patch.object(decision, 'load_selected', side_effect=fake_load), \
         patch.object(sys, 'argv', ['compare_development.py', '--stage', 'readout', '--bootstrap', '200']):
        decision.main()
    second = json.loads((fixture / 'decisions/readout.json').read_text())
    assert not second['confirmation_recommended'] and second['confirmation_candidate'] is None
    a = fake_load('clean-cls-linear-lr2e-5-fold0-seed2')
    b = copy.deepcopy(a)
    b['config'].update(learning_rate=5e-6, readout='residue_mean')
    try:
        decision.paired_comparison(a, b, 200)
    except AssertionError as e:
        assert 'confound' in str(e)
    else:
        raise AssertionError('A confounded comparison was accepted')
    assert not (ROOT / 'runs/not-a-completed-run').exists()
    atomic_json(ROOT / 'qualification/decisions.json', {
        'passed': True, 'scope': 'Synthetic scores and mocked completed loaders; real bootstrap, metric, promotion and stage orchestration code.',
        'incomplete_production_runs_rejected': True, 'zero_gain_not_promoted': True,
        'all_failed_arms_stop_confirmation': True, 'confounded_lr_and_readout_comparison_rejected': True,
        'source_sha256': {f'scripts/{n}': sha256(ROOT / 'scripts' / n) for n in
                          ['compare_development.py', 'metrics.py', 'qualify_decisions.py']}})
    print('Decision qualification passed; synthetic results are not campaign outcomes.')


if __name__ == '__main__':
    main()
