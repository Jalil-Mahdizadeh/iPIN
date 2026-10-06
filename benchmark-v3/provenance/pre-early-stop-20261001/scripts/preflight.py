"""One engineering forward check with an existing residue head; no training."""
import json
from pathlib import Path
import numpy as np
import torch
from common import ROOT, PairData, TrainingData, atomic_json, configure, load_checkpoint, sha256
from prepare_selected import check_array
from qualify import forward_checks, validation_data


def main():
    plan = json.loads((ROOT / 'provenance/plan.json').read_text())
    data, original = validation_data(plan)
    train = TrainingData(Path(plan['training_prepared_path']), 'official', 'train')
    assert len(train) == 163085 and int(train.rows[:, 2].sum()) == 81550
    test = PairData(ROOT / 'data', 'test')
    assert len(test) == 52048 and int(test.rows[:, 2].sum()) == 26024 and test.lengths.max() == 7426
    for name, sources in plan['reused_predictions'].items():
        for split, ds in [('test', test), ('val', data)]:
            check_array(ROOT / sources[split]['path'], sources[split]['sha256'], ds)
    run = ROOT.parent / 'retrain-v3/runs/clean-residue-mean-lr2e-5-fold0-seed2'
    best = json.loads((run / 'best.json').read_text())
    contract = json.loads((run / 'contract.json').read_text())
    for src, dst in [('model.py', 'frozen_model.py'), ('data.py', 'training_data.py')]:
        assert sha256(ROOT / 'scripts' / dst) == contract['code'][src]
    entry = {'path': str(run / 'checkpoints' / best['file']), 'sha256': best['sha256'],
             'update': best['update'], 'source_manifest': best, 'configuration': contract['configuration']}
    device = configure()
    model = load_checkpoint(entry, plan['base_model_path'], device)
    checks = forward_checks(model, data, original, device)
    report = {'passed': True, 'checks': checks, 'train_pairs': len(train), 'val_pairs': len(data),
        'test_pairs': len(test), 'verified_cached_prediction_arrays': 14,
        'scope': 'Engineering only: existing fold checkpoint, full official validation adapter and longest pair; no training or new test predictions',
        'checkpoint': entry, 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'source_sha256': {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / 'scripts').glob('*.py'))}}
    atomic_json(ROOT / 'provenance/preflight.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
