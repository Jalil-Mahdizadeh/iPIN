"""Load a verified, equal-checkpoint development window after an authorized stop."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
from data import PairData
from metrics import evaluate
from state import sha256

ROOT = Path('/nobackup/proj/disk/theo-storage/personal/jalil/iPIN/retrain-v3')


def load_window(name, end=5000, require_screen=False):
    run = ROOT / 'runs' / name
    contract = json.loads((run / 'contract.json').read_text())
    cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
    assert cfg == contract['configuration'] and not contract['qualification']
    assert cfg['total_updates'] == 6000 and cfg['warmup_updates'] == 1000
    assert cfg['partition'].startswith('fold-') and cfg['initialization'] == 'esm2_pretrained'
    assert cfg['mlm_weight'] == 0 and cfg['classification_corruption'] is False
    assert contract['code'] == {n: sha256(ROOT / 'scripts' / n) for n in contract['code']}
    assert contract['data_manifest_sha256'] == sha256(ROOT / 'data/prepared/manifest.json')
    stop = json.loads((run / 'stopped.json').read_text())
    state = stop['state']
    assert stop['checkpoint_committed'] and not state['pending_validation']
    assert state['last_validation_update'] == end and state['update'] >= end
    assert state['examples_seen'] == state['update'] * cfg['global_pairs_per_update']
    latest = json.loads((run / 'latest.json').read_text())
    best = json.loads((run / 'best.json').read_text())
    assert latest['update'] == state['update'] and best['update'] == state['best_update']
    assert latest['fingerprint'] == best['fingerprint'] == contract['fingerprint']
    proof = ['contract.json', 'stopped.json', 'latest.json', 'best.json']
    if require_screen:
        screen = json.loads((run / 'SCREEN_COMPLETE.json').read_text())
        assert screen['screen_completed'] and screen['comparison_window_end'] == end
        assert screen['fingerprint'] == contract['fingerprint']
        assert screen['actual_stopped_update'] == state['update'] <= end + 5
        assert screen['best_checkpoint_sha256'] == best['sha256']
        proof.append('SCREEN_COMPLETE.json')
    else:
        request = json.loads((ROOT / 'provenance/stop-adaptation-request.json').read_text())
        assert name in [r['run'] for r in request['runs']]
        assert (run / 'REQUEST_STOP').exists()
    evidence = {str((run / n).relative_to(ROOT)): sha256(run / n) for n in proof}
    evidence['data/prepared/manifest.json'] = sha256(ROOT / 'data/prepared/manifest.json')
    evidence[f'configs/{name}.json'] = sha256(ROOT / 'configs' / (name + '.json'))
    data = PairData(ROOT / 'data/prepared', cfg['partition'], 'val')
    maximum, first_best, selected, trajectory = -np.inf, None, None, []
    for update in range(1000, end + 1, 1000):
        file = run / 'validation' / f'update-{update:09d}.npz'
        meta = json.loads(file.with_suffix('.json').read_text())
        assert sha256(file) == meta['sha256'] and meta['fingerprint'] == contract['fingerprint']
        assert meta['update'] == update and meta['rows'] == len(data)
        with np.load(file) as f:
            preds = f['predictions']
        assert preds.shape == (len(data), 4) and np.isfinite(preds).all()
        assert np.array_equal(preds[:, 0], np.arange(len(data)))
        assert np.array_equal(preds[:, 1], data.rows[:, 2])
        ap = float(average_precision_score(preds[:, 1], preds[:, 2:4].mean(1)))
        assert abs(ap - meta['metrics']['pooled_ap']) < 1e-12
        trajectory.append({'update': update, 'ap': ap})
        if ap > maximum:
            maximum, first_best, selected = ap, update, preds[:, 2:4].mean(1)
        evidence[str(file.relative_to(ROOT))] = meta['sha256']
        evidence[str(file.with_suffix('.json').relative_to(ROOT))] = sha256(file.with_suffix('.json'))
    assert first_best == best['update'] and abs(maximum - best['best_ap']) < 1e-12
    return {'name': name, 'config': cfg, 'rows': data.rows, 'lengths': data.lengths,
            'scores': selected, 'best_update': first_best, 'metrics': evaluate(data.rows, selected),
            'trajectory': trajectory, 'best_at_horizon': first_best == end,
            'last_evaluation_ap_gain': trajectory[-1]['ap'] - trajectory[-2]['ap'],
            'actual_stopped_update': state['update'], 'evidence': evidence}
