"""Record the user-authorized early LR decision without claiming full completion."""
import datetime
import json
from pathlib import Path
from metrics import promotion
from selection_window import load_window
from state import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    target = ROOT / 'decisions/adaptation-early.json'
    assert not target.exists(), 'Preserve the original decision'
    campaign = json.loads((ROOT / 'configs/campaign.json').read_text())
    request = ROOT / 'provenance/stop-adaptation-request.json'
    assert json.loads(request.read_text())['selected_learning_rate'] == 2e-5
    results, deltas, evidence, payloads = [], [], {}, {}
    for fold in range(3):
        pair = [load_window(f'clean-cls-linear-{tag}-fold{fold}-seed2') for tag in ['lr2e-5', 'lr5e-6']]
        a, b = pair
        delta = {k: b['metrics'][k] - a['metrics'][k] for k in ['ap', 'auroc', 'macro_ap']}
        deltas.append(delta)
        results.append({'fold': fold, 'control': {k: a[k] for k in ['name', 'best_update', 'metrics', 'actual_stopped_update']},
                        'lower_lr': {k: b[k] for k in ['name', 'best_update', 'metrics', 'actual_stopped_update']},
                        'lower_minus_control': delta})
        for item in pair:
            evidence.update(item['evidence'])
            run = ROOT / 'runs' / item['name']
            for pointer in ['best', 'latest']:
                meta = json.loads((run / f'{pointer}.json').read_text())
                path = run / 'checkpoints' / meta['file']
                assert path.stat().st_size == meta['bytes']
                digest = payloads.get(str(path)) or sha256(path)
                assert digest == meta['sha256'], (item['name'], pointer, 'checkpoint changed')
                payloads[str(path)] = digest
            print(item['name'], 'best/latest payloads verified', flush=True)
    gate = promotion(deltas, campaign['development_promotion'])
    assert not gate['promoted'] and all(d['ap'] < 0 for d in deltas)
    evidence[str(request.relative_to(ROOT))] = sha256(request)
    evidence['configs/campaign.json'] = sha256(ROOT / 'configs/campaign.json')
    atomic_json(target, {
        'time_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'stage': 'adaptation', 'decision_type': 'user-authorized early development selection',
        'selected_learning_rate': 2e-5, 'selected_learning_rate_tag': 'lr2e-5',
        'completed_prespecified_6000_update_comparison': False,
        'common_validation_updates': [1000, 2000, 3000, 4000, 5000],
        'evaluation_selection_window_end': 5000, 'training_schedule_updates': 6000,
        'decision_basis': results, 'lower_lr_promotion_rule_on_observed_window': gate,
        'next_runs': campaign['stage2_templates_by_learning_rate']['lr2e-5'],
        'next_stage_budget': 'Keep the 6000-update learning-rate schedule, stop after committed validation 5000; compare exactly the same five checkpoint opportunities.',
        'test_evaluated': False, 'checkpoint_payloads_verified': payloads,
        'evidence_sha256': evidence,
        'limitation': 'Window and early stop were chosen after seeing adaptation development outcomes; freeze this window before readout training. No external superiority claim.'})
    print('Selected 2e-5; early decision and preserved checkpoints recorded.', flush=True)


if __name__ == '__main__':
    main()
