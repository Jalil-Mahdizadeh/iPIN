"""CPU-only selection guards and eight-model report fixture; no new inference."""
import copy
import json
import os
import shutil
import tempfile
from pathlib import Path
import numpy as np
import analyze
import prepare_selected
import write_report
from common import ROOT, atomic_json, sha256
from qualify import validation_statistics


def selection_fixture(base):
    b, run = base / 'selection-benchmark', base / 'selection-run'
    for path in [b / d for d in ['provenance', 'scripts', 'data', 'checkpoints', 'predictions']] + [run / 'validation', run / 'checkpoints']:
        path.mkdir(parents=True)
    cfg = json.loads((ROOT / 'provenance/plan.json').read_text())['training_configuration']
    for name in ['frozen_model.py', 'training_data.py']: shutil.copy2(ROOT / 'scripts' / name, b / 'scripts' / name)
    data = type('FixtureData', (), {'__len__': lambda self: 4})()
    data.rows = np.array([[0, 1, 0, 0], [2, 3, 0, 1], [4, 5, 1, 2], [6, 7, 1, 3]])
    records = {}
    for step in list(range(1000, 12745, 1000)) + [12745]:
        scores = [-1., -1., 1., 1.] if step in [1000, 2000] else [1., 1., -1., -1.]
        a = np.column_stack((np.arange(4), data.rows[:, 2], scores, scores))
        path = run / 'validation' / f'update-{step:09d}.npz'
        np.savez_compressed(path, predictions=a)
        metric = float(prepare_selected.average_precision_score(a[:, 1], a[:, 2:4].mean(1)))
        atomic_json(path.with_suffix('.json'), {'update': step, 'fingerprint': 'fixture-only',
            'sha256': sha256(path), 'metrics': {'pooled_ap': metric}})
        records[step] = metric
    payload = run / 'checkpoints/update-000001000.pt'
    payload.write_bytes(b'FIXTURE ONLY; not a model or training artifact')
    best = {'update': 1000, 'best_ap': records[1000], 'file': payload.name,
            'sha256': sha256(payload), 'fingerprint': 'fixture-only'}
    atomic_json(run / 'best.json', best)
    atomic_json(run / 'contract.json', {'configuration': cfg, 'qualification': False,
        'code': {'model.py': sha256(b / 'scripts/frozen_model.py'), 'data.py': sha256(b / 'scripts/training_data.py')}})
    completed = {'state': {'update': 12745, 'total_steps': 12745, 'last_validation_update': 12745,
        'pending_validation': False, 'examples_seen': 12745 * 64, 'best_update': 1000, 'best_ap': records[1000]},
        'model_hashes_by_rank': ['fixture-only'] * 4, 'qualification': False, 'test_evaluated': False, 'fingerprint': 'fixture-only'}
    atomic_json(run / 'completed.json', completed)
    plan = json.loads((ROOT / 'provenance/plan.json').read_text())
    plan.update(run_path=str(run))
    atomic_json(b / 'provenance/plan.json', plan)
    old_root, old_data = prepare_selected.ROOT, prepare_selected.PairData
    prepare_selected.ROOT, prepare_selected.PairData = b, lambda *args: data
    try:
        prepare_selected.main()
        first = sha256(b / 'provenance/selection.json')
        prepare_selected.main()
        assert sha256(b / 'provenance/selection.json') == first
        bad = copy.deepcopy(completed); bad['state']['update'] = 12744
        atomic_json(run / 'completed.json', bad)
        try: prepare_selected.main()
        except AssertionError: pass
        else: raise AssertionError('Unfinished training was accepted')
        atomic_json(run / 'completed.json', completed)
        bad_best = dict(best, update=2000)
        atomic_json(run / 'best.json', bad_best)
        try: prepare_selected.main()
        except AssertionError: pass
        else: raise AssertionError('Later exact-tie checkpoint was accepted')
    finally:
        prepare_selected.ROOT, prepare_selected.PairData = old_root, old_data
    return {'all_13_validations_required': True, 'earlier_tie_selected': True,
            'selection_idempotent': True, 'unfinished_run_rejected': True, 'wrong_best_rejected': True}


def analysis_fixture(base):
    b = base / 'benchmark-v3'
    for d in ['provenance', 'data', 'reused', 'results']: (b / d).mkdir(parents=True)
    (base / 'retrain-v3').mkdir()
    (base / 'retrain-v3/README.md').write_text('# PLM-interact retraining v3\n\nFixture only.\n')
    for d in ['data', 'reused']:
        for path in (ROOT / d).iterdir():
            if path.is_file(): os.link(path, b / d / path.name)
    for name in ['reuse-verification.json', 'benchmark-v2-summary.json']:
        shutil.copy2(ROOT / 'provenance' / name, b / 'provenance' / name)
    cfg = json.loads((ROOT / 'config.json').read_text())
    plan = json.loads((ROOT / 'provenance/plan.json').read_text())
    # Stand in for the future candidate with an existing prediction set. This
    # exercises all comparisons, coverage, bootstrap verification and reporting.
    plan['models']['v3-residue-mlp'] = copy.deepcopy(plan['models']['v2-clean-bce'])
    plan['reused_predictions']['v3-residue-mlp'] = copy.deepcopy(plan['reused_predictions']['v2-clean-bce'])
    atomic_json(b / 'config.json', cfg)
    atomic_json(b / 'provenance/selection.json', plan)
    atomic_json(b / 'provenance/qualification.json', {'passed': True, 'prediction_fingerprint': 'fixture-only'})
    validation = {}
    for name, sources in plan['reused_predictions'].items():
        with np.load(b / sources['val']['path']) as f: validation[name] = validation_statistics(f['predictions'])
    atomic_json(b / 'provenance/validation-before-test.json', {'metrics_and_thresholds': validation})
    old_root, old_fingerprint, old_report_root = analyze.ROOT, analyze.prediction_fingerprint, write_report.ROOT
    analyze.ROOT, analyze.prediction_fingerprint, write_report.ROOT = b, lambda: 'fixture-only', b
    try:
        analyze.main()
        write_report.main()
        completed = json.loads((b / 'completed.json').read_text())
        assert completed['v3_closed'] and (base / 'retrain-v3/FINALIZED.json').exists()
        assert 'superiority is not established' in completed['conclusion']
        write_report.finish_closeout(completed)
    finally:
        analyze.ROOT, analyze.prediction_fingerprint, write_report.ROOT = old_root, old_fingerprint, old_report_root
    return {'eight_model_report_completed': True, 'all_seven_baselines_reproduced_to_1e_12': True,
            'bootstrap_replicates': cfg['bootstrap_replicates'], 'finalization_idempotent': True,
            'new_model_predictions': False, 'fixture_only': True}


def main():
    with tempfile.TemporaryDirectory(prefix='final-report-fixture-', dir=ROOT.parent / 'retrain-v3/qualification') as tmp:
        base = Path(tmp)
        selected = selection_fixture(base)
        analysis = analysis_fixture(base)
    report = {'passed': True, 'selection': selected, 'analysis': analysis,
        'source_sha256': {name: sha256(ROOT / 'scripts' / name) for name in ['prepare_selected.py', 'analyze.py', 'write_report.py']}}
    atomic_json(ROOT / 'provenance/finalization-check.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
