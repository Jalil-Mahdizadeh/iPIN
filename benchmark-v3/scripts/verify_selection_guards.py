"""Check early-stop authorization and checkpoint-selection failure cases on copies."""
import copy
import json
import shutil
import tempfile
from pathlib import Path

import prepare_selected
from check_finalize import selection_fixture
from common import ROOT, atomic_json, sha256, stop_requested


def main():
    plan = json.loads((ROOT / 'provenance/plan.json').read_text())
    source = Path(plan['run_path'])
    cfg = plan['training_configuration']
    contract = json.loads((source / 'contract.json').read_text())
    _, updates, endpoint = prepare_selected.training_endpoint(source, cfg, contract)
    assert updates == list(range(1000, 8001, 1000))
    assert endpoint['stopped_update'] == 8446
    assert (source / 'REQUEST_STOP').exists() and not stop_requested()
    rejected = []
    with tempfile.TemporaryDirectory(prefix='selection-guards-', dir=ROOT / 'provenance') as temp:
        base = Path(temp)
        full_checks = selection_fixture(base)
        benchmark, run = base / 'early-benchmark', base / 'early-run'
        (benchmark / 'provenance').mkdir(parents=True)
        (run / 'validation').mkdir(parents=True)
        for name in ['contract.json', 'stopped.json', 'latest.json', 'best.json', 'REQUEST_STOP']:
            shutil.copy2(source / name, run / name)
        for p in (source / 'validation').glob('update-*.json'):
            shutil.copy2(p, run / 'validation' / p.name)
        amendment = json.loads((ROOT / 'provenance/early-stop-amendment.json').read_text())
        amendment['source_run'] = str(run)
        path = benchmark / 'provenance/early-stop-amendment.json'
        atomic_json(path, amendment)
        old_root = prepare_selected.ROOT
        prepare_selected.ROOT = benchmark
        try:
            prepare_selected.training_endpoint(run, cfg, contract)
            def rejects(name):
                try:
                    prepare_selected.training_endpoint(run, cfg, contract)
                except (AssertionError, FileNotFoundError):
                    rejected.append(name)
                else:
                    raise AssertionError(f'Invalid endpoint accepted: {name}')
            path.unlink()
            rejects('early_stop_without_amendment')
            atomic_json(path, dict(amendment, approved_early_stop=False))
            rejects('unauthorized_early_stop')
            atomic_json(path, amendment)
            (run / 'REQUEST_STOP').unlink()
            rejects('training_restart_marker_removed')
            shutil.copy2(source / 'REQUEST_STOP', run / 'REQUEST_STOP')
            missing = run / 'validation/update-000005000.json'
            missing.unlink()
            rejects('missing_intermediate_validation')
            shutil.copy2(source / 'validation' / missing.name, missing)
            bad = json.loads((run / 'best.json').read_text())
            bad['update'] = 8000
            atomic_json(run / 'best.json', bad)
            rejects('changed_best_checkpoint_manifest')
            shutil.copy2(source / 'best.json', run / 'best.json')
            bad_stop = json.loads((run / 'stopped.json').read_text())
            bad_stop['state']['pending_validation'] = True
            atomic_json(run / 'stopped.json', bad_stop)
            modified = copy.deepcopy(amendment)
            modified['stopped_manifest_sha256'] = sha256(run / 'stopped.json')
            atomic_json(path, modified)
            rejects('pending_validation_at_stop')
        finally:
            prepare_selected.ROOT = old_root
    result = {'passed': True, 'full_horizon_selection_checks': full_checks,
        'accepted_verified_early_stop': True, 'completed_validation_updates': updates,
        'invalid_cases_rejected': rejected, 'training_stop_does_not_stop_benchmark': True,
        'source_sha256': {name: sha256(ROOT / 'scripts' / name) for name in
            ['prepare_selected.py', 'common.py', 'verify_selection_guards.py']}}
    atomic_json(ROOT / 'provenance/selection-guard-checks.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
