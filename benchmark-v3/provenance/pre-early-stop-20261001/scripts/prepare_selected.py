"""Freeze exactly one completed full-data run's validation-selected checkpoint."""
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


def main():
    plan = json.loads((ROOT / 'provenance/plan.json').read_text())
    run = Path(plan['run_path'])
    cfg = plan['training_configuration']
    contract = json.loads((run / 'contract.json').read_text())
    assert contract['configuration'] == cfg and not contract['qualification']
    for source, name in [('model.py', 'frozen_model.py'), ('data.py', 'training_data.py')]:
        assert sha256(ROOT / 'scripts' / name) == contract['code'][source]
    completed = json.loads((run / 'completed.json').read_text())
    state = completed['state']
    assert not completed['qualification'] and not completed['test_evaluated']
    assert state['update'] == state['total_steps'] == cfg['total_updates'] == 12745
    assert state['last_validation_update'] == 12745 and not state['pending_validation']
    assert state['examples_seen'] == 12745 * 64
    assert len(completed['model_hashes_by_rank']) == 4
    assert len(set(completed['model_hashes_by_rank'])) == 1
    best = json.loads((run / 'best.json').read_text())
    assert best['fingerprint'] == completed['fingerprint']
    val = PairData(ROOT / 'data', 'val')
    records = []
    # Require every planned validation; independently verify the strict-max rule.
    updates = list(range(1000, 12745, 1000)) + [12745]
    for step in updates:
        path = run / 'validation' / f'update-{step:09d}.npz'
        meta = json.loads(path.with_suffix('.json').read_text())
        assert meta['update'] == step and meta['fingerprint'] == completed['fingerprint']
        a = check_array(path, meta['sha256'], val)
        ap = float(average_precision_score(a[:, 1], a[:, 2:4].mean(1)))
        assert abs(ap - meta['metrics']['pooled_ap']) < 1e-12
        records.append({'update': step, 'ap': ap, 'sha256': meta['sha256']})
    winner = max(records, key=lambda row: row['ap'])  # Earlier wins exact ties.
    assert best['update'] == state['best_update'] == winner['update']
    assert best['best_ap'] == state['best_ap'] == winner['ap']
    source = run / 'checkpoints' / best['file']
    checkpoint = ROOT / 'checkpoints' / f'v3-residue-mlp-{best["file"]}'
    snapshot(source, checkpoint, best['sha256'], link=True)
    val_source = run / 'validation' / f'update-{best["update"]:09d}.npz'
    val_target = ROOT / 'predictions/v3-residue-mlp-selected-validation.npz'
    snapshot(val_source, val_target, winner['sha256'])
    entry = {'kind': 'local_training_checkpoint', 'path': str(checkpoint.relative_to(ROOT)),
        'sha256': best['sha256'], 'update': best['update'], 'source_manifest': best,
        'source_path': str(source), 'source_run': plan['run_name'], 'configuration': cfg,
        'validation_primary_ap': winner['ap'], 'test_predictions_reused': False,
        'snapshot_method': 'hard link to immutable atomically committed checkpoint',
        'selected_validation': {'path': str(val_target.relative_to(ROOT)),
                                'source_path': str(val_source), 'sha256': winner['sha256']}}
    selection = {k: plan[k] for k in ['models', 'reused_predictions', 'data', 'container',
        'base_model_path', 'training_prepared_path', 'historical_test_interpretation']}
    selection['models'] = dict(selection['models'], **{'v3-residue-mlp': entry})
    selection.update(selection_basis='Maximum official validation pooled AP over 13 fixed opportunities; earliest exact tie. No test-based selection.',
        fresh_inference_models=['v3-residue-mlp'], training_completed_sha256=sha256(run / 'completed.json'),
        training_contract_sha256=sha256(run / 'contract.json'), validation_selection_records=records)
    target = ROOT / 'provenance/selection.json'
    if target.exists():
        prior = json.loads(target.read_text())
        assert {k: v for k, v in prior.items() if k != 'selected_at_utc'} == selection
    else:
        selection['selected_at_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        atomic_json(target, selection)
    print(json.dumps({'event': 'final_checkpoint_frozen', 'update': best['update'],
                      'validation_ap': winner['ap'], 'selection_opportunities': len(records)}), flush=True)


if __name__ == '__main__': main()
