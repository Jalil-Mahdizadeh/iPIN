"""Freeze the completed C0 run's validation-selected checkpoint before testing."""
import datetime
import json
import os
import shutil
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score
from common import ROOT, PairData, atomic_json, sha256


def check_array(path, digest, data):
    assert sha256(path) == digest, str(path)
    with np.load(path, allow_pickle=False) as saved:
        a = saved['predictions']
    assert a.shape == (len(data), 4) and np.isfinite(a).all()
    assert np.array_equal(a[:, 0], np.arange(len(data)))
    assert np.array_equal(a[:, 1], data.rows[:, 2])
    return a


def snapshot(source, destination, digest, link=False):
    assert sha256(source) == digest
    if not destination.exists():
        temporary = destination.with_name(destination.name + f'.tmp-{os.getpid()}')
        if temporary.exists(): temporary.unlink()
        if link: os.link(source, temporary)
        else: shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    assert sha256(destination) == digest


def validate_endpoint(contract, completed, best):
    cfg, state = contract['configuration'], completed['state']
    assert cfg['arm'] == 'C0' and cfg['backbone'] == 'esmc'
    assert cfg['attention_mode'] == 'standard' and cfg['partition'] == 'official'
    assert not contract['qualification'] and not contract['tiny']
    assert not completed['qualification'] and not completed['test_evaluated']
    assert state['update'] == state['total_steps'] == cfg['total_updates'] == 12745
    assert state['last_validation_update'] == 12745 and not state['pending_validation']
    assert state['examples_seen'] == 12745 * 64
    assert len(completed['model_hashes_by_rank']) == 4
    assert len(set(completed['model_hashes_by_rank'])) == 1
    assert best['fingerprint'] == completed['fingerprint'] == contract['fingerprint']
    assert best['update'] == state['best_update'] and best['best_ap'] == state['best_ap']


def main():
    plan = json.loads((ROOT / 'provenance/plan.json').read_text())
    run = Path(plan['run_path'])
    for name, key in [('contract.json', 'training_contract_sha256'),
                      ('completed.json', 'training_completed_sha256'),
                      ('best.json', 'best_manifest_sha256')]:
        assert sha256(run / name) == plan[key]
    contract = json.loads((run / 'contract.json').read_text())
    completed = json.loads((run / 'completed.json').read_text())
    best = json.loads((run / 'best.json').read_text())
    cfg = contract['configuration']
    assert cfg == plan['training_configuration']
    validate_endpoint(contract, completed, best)
    for source, name in [('model.py', 'frozen_model.py'), ('model_esm2.py', 'model_esm2.py'),
                         ('model_esmc.py', 'model_esmc.py'), ('data.py', 'training_data.py')]:
        assert sha256(ROOT / 'scripts' / name) == contract['code'][source]
    for name, digest in plan['data']['sha256'].items():
        assert sha256(ROOT / 'data' / name) == digest
    val, test = PairData(ROOT / 'data', 'val'), PairData(ROOT / 'data', 'test')
    assert len(val) == 59258 and len(test) == 52048
    for sources in plan['reused_predictions'].values():
        for split, data in [('val', val), ('test', test)]:
            item = sources[split]
            check_array(ROOT / item['path'], item['sha256'], data)
    records = []
    updates = list(range(1000, 12745, 1000)) + [12745]
    assert sorted(int(p.stem.split('-')[1]) for p in (run / 'validation').glob('update-*.json')) == updates
    for step in updates:
        path = run / 'validation' / f'update-{step:09d}.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['update'] == step and meta['fingerprint'] == contract['fingerprint']
        a = check_array(path, meta['sha256'], val)
        ap = float(average_precision_score(a[:, 1], a[:, 2:4].mean(1)))
        assert abs(ap - meta['metrics']['pooled_ap']) < 1e-12
        records.append({'update': step, 'ap': ap, 'sha256': meta['sha256']})
    winner = max(records, key=lambda row: row['ap'])
    assert best['update'] == winner['update'] == 7000
    assert best['best_ap'] == winner['ap']
    source = run / 'checkpoints' / best['file']
    checkpoint = ROOT / 'checkpoints' / f'v4-esmc-standard-{best["file"]}'
    snapshot(source, checkpoint, best['sha256'], link=True)
    val_source = run / 'validation' / f'update-{best["update"]:09d}.npz'
    val_target = ROOT / 'predictions/v4-esmc-standard-selected-validation.npz'
    snapshot(val_source, val_target, winner['sha256'])
    selection = {k: plan[k] for k in ['reused_predictions', 'data', 'container', 'base_model_path',
        'training_prepared_path', 'historical_test_interpretation']}
    selection['models'] = {name: dict(entry, test_predictions_reused=True,
        checkpoint_identity_source='provenance/benchmark-v3-selection.json', path=entry['source_path'])
        for name, entry in plan['models'].items()}
    selection['models']['v4-esmc-standard'] = {'kind': 'local_training_checkpoint',
        'path': str(checkpoint.relative_to(ROOT)), 'source_path': str(source),
        'source_manifest': best, 'source_run': plan['run_name'], 'sha256': best['sha256'],
        'update': best['update'], 'configuration': cfg, 'validation_primary_ap': winner['ap'],
        'test_predictions_reused': False, 'snapshot_method': 'hard link to immutable committed checkpoint',
        'selected_validation': {'path': str(val_target.relative_to(ROOT)),
            'source_path': str(val_source), 'sha256': winner['sha256']}}
    selection.update(selection_basis='Maximum pooled official-validation AP over all 13 planned validations; earliest exact tie. No test-based selection.',
        fresh_inference_models=['v4-esmc-standard'], validation_selection_records=records,
        training_completed_sha256=plan['training_completed_sha256'],
        training_contract_sha256=plan['training_contract_sha256'],
        training_endpoint={'training_horizon_completed': True, 'updates': 12745,
                           'validation_opportunities': 13})
    target = ROOT / 'provenance/selection.json'
    if target.exists():
        prior = json.loads(target.read_text())
        assert {k: v for k, v in prior.items() if k != 'selected_at_utc'} == selection
    else:
        selection['selected_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(target, selection)
    print(json.dumps({'event': 'checkpoint_frozen', 'update': best['update'],
        'validation_ap': winner['ap'], 'selection_opportunities': len(records),
        'reused_baseline_models': len(plan['reused_predictions'])}), flush=True)


if __name__ == '__main__':
    main()
