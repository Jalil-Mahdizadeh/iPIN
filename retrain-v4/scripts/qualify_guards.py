"""Ensure shortened/changed or unfrozen training cannot enter production."""
import json
import subprocess
import sys
from pathlib import Path
from contracts import core_hashes, production_config, sha256
from release import verify_release

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = ROOT / 'configs/esm2-chain-aware-official-seed2.json'
    cfg = json.loads(config.read_text())
    production_config(cfg)
    rejected = []
    for name, value in dict(total_updates=4, warmup_updates=1, learning_rate=5e-6,
        global_pairs_per_update=9, world_size=1, seed=3, train_cap_residues=2193,
        partition='fold-0', readout='cls_mlp', classification_corruption=True,
        mlm_weight=.1, positive_weight=2., attention_backend='math',
        attention_mode='standard', initialization='native_supervised').items():
        try: production_config({**cfg, name: value})
        except AssertionError: rejected.append(name)
        else: raise AssertionError(('Production accepted a changed protocol', name))
    try: verify_release(ROOT, config, ROOT / 'scripts', verify_assets=False)
    except (AssertionError, FileNotFoundError): pass
    else: raise AssertionError('Unfrozen source tree enabled production')
    out = ROOT / 'runs/guard-must-not-be-created'
    assert not out.exists()
    cases = []
    for args in [
        ['--production', '--tiny'], ['--production', '--max-updates', '4'],
        ['--production', '--train-limit', '11'], ['--production', '--stop-after-updates', '2'],
        ['--qualification', '--max-updates', '4', '--train-limit', '11', '--val-limit', '8']]:
        result = subprocess.run([sys.executable, '-B', str(ROOT / 'scripts/train.py'),
            '--config', str(config), '--output', str(out), *args], capture_output=True, text=True)
        assert result.returncode != 0 and 'AssertionError' in result.stderr, (args, result.stdout, result.stderr)
        assert not out.exists(), 'Rejected command created a production directory'
        cases.append({'arguments': args, 'rejected_before_output_creation': True})
    report = {'passed': True, 'changed_production_fields_rejected': rejected,
              'unfrozen_source_rejected': True, 'cli_cases': cases,
              'production_output_created': False, 'code': core_hashes(ROOT / 'scripts'),
              'source_sha256': {'scripts/qualify_guards.py': sha256(Path(__file__))}}
    (ROOT / 'qualification/production-guards.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
