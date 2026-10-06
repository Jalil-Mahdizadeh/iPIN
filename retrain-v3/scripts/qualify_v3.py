"""Audit the v3 data/configuration contract, inherited engine and decision rules."""
import json
from pathlib import Path
import numpy as np
import torch
from data import PairData
from model import PairModel
from metrics import evaluate, promotion
from state import atomic_json, sha256
from train import configuration_checks

ROOT = Path(__file__).resolve().parents[1]
CORE = ['train.py', 'model.py', 'data.py', 'state.py', 'release.py']


def main():
    torch.set_num_threads(8)
    core = {n: sha256(ROOT / 'scripts' / n) for n in CORE}
    previous = ROOT.parent / 'retrain-v2'
    old_release = previous / 'releases' / (previous / 'releases/CURRENT').read_text().strip()
    assert core == {n: sha256(old_release / 'code' / n) for n in CORE}
    source_reports = {}
    for name in ['science.json', 'ddp.json', 'signal-resume.json', 'checkpoint-faults.json',
                 'scheduler-requeue.json', 'full-model-profile.json']:
        path = ROOT / 'provenance/v2-qualification' / name
        report = json.loads(path.read_text())
        assert report['passed']
        assert sha256(path) == sha256(previous / 'qualification' / name)
        if 'code' in report:
            assert report['code'] == core
        source_reports[str(path.relative_to(ROOT))] = sha256(path)

    path = ROOT / 'data/prepared'
    manifest = json.loads((path / 'manifest.json').read_text())
    assert not manifest['test_imported']
    assert sha256(path / 'manifest.json') == sha256(previous / 'data/prepared/manifest.json')
    for name, expected in manifest['files'].items():
        assert sha256(path / name) == expected
    assert not list((ROOT / 'data').rglob('*test*'))
    official = PairData(path, 'official', 'train')
    official_val = PairData(path, 'official', 'val')
    active = np.unique(np.concatenate([official.rows[:, :2].ravel(), official_val.rows[:, :2].ravel()]))
    assert np.array_equal(np.flatnonzero(np.diff(official.offsets)), active)
    assert not set(official.rows[:, :2].ravel()) & set(official_val.rows[:, :2].ravel())
    groups = json.loads((path / 'fold-groups.json').read_text())
    assert sha256(ROOT / 'data/development-homology.tsv') == groups['homology_search_sha256']
    assignments = {p: f for f, ids in enumerate(groups['fold_proteins']) for p in ids}
    for line in (ROOT / 'data/development-homology.tsv').read_text().splitlines():
        a, b, identity, qc, tc, _ = line.split('\t')
        if float(identity) >= .4 and min(float(qc), float(tc)) >= .8:
            assert assignments[int(a)] == assignments[int(b)]
    folds = []
    for fold in range(3):
        tr, va = [PairData(path, f'fold-{fold}', s) for s in ['train', 'val']]
        cross = np.load(path / f'fold-{fold}/crossing-unused.npy')
        rebuilt = np.concatenate([tr.rows, va.rows, cross])
        assert np.array_equal(rebuilt[np.argsort(rebuilt[:, 3])], official.rows[np.argsort(official.rows[:, 3])])
        assert not set(tr.rows[:, :2].ravel()) & set(va.rows[:, :2].ravel())
        assert all(assignments[int(p)] == fold for p in np.unique(va.rows[:, :2]))
        assert all(assignments[int(p)] != fold for p in np.unique(tr.rows[:, :2]))
        cursor, seen = {'cycle': 0, 'offset': 0}, []
        for _ in range((len(tr) + 63) // 64):
            ids, cycles, cursor = tr.next_batch(cursor, 2, 64)
            seen.extend(ids[cycles == 0].tolist())
        assert np.array_equal(np.sort(seen), np.arange(len(tr)))
        clone = PairData(path, f'fold-{fold}', 'train')
        for a, b in zip(tr.next_batch(cursor, 2, 64)[:2], clone.next_batch(cursor, 2, 64)[:2]):
            assert np.array_equal(a, b)
        folds.append({'fold': fold, 'train_rows': len(tr), 'val_rows': len(va),
                      'crossing_unused': len(cross), 'pair_exposures': 6000 * 64,
                      'effective_epochs': 6000 * 64 / len(tr),
                      'maximum_train_tokens': int(tr.lengths.max()), 'maximum_val_tokens': int(va.lengths.max())})

    campaign = json.loads((ROOT / 'configs/campaign.json').read_text())
    reference = json.loads((previous / 'configs/clean-bce-fold0-seed2.json').read_text())
    initial = []
    for tag, lr in [('lr2e-5', 2e-5), ('lr5e-6', 5e-6)]:
        for head in ['cls_linear', 'cls_mlp', 'residue_mean']:
            arm = f'clean-{head.replace("_", "-")}-{tag}'
            for fold in range(3):
                name = f'{arm}-fold{fold}-seed2'
                cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
                expected = {**reference, 'root': str(ROOT), 'name': name, 'arm': arm,
                            'readout': head, 'learning_rate': lr, 'partition': f'fold-{fold}'}
                assert cfg == expected, name
                configuration_checks(cfg)
                if head == 'cls_linear':
                    initial.append(name)
    assert initial == campaign['initial_runs'] and len(initial) == 6
    assert campaign['production_submission_authorized_now'] is False
    optional = json.loads((ROOT / 'configs/clean-capped-official-seed2.json').read_text())
    v2cap = json.loads((previous / 'configs/capped-official-seed2.json').read_text())
    assert optional == {**v2cap, 'root': str(ROOT), 'name': 'clean-capped-official-seed2',
                        'arm': 'clean-capped', 'classification_corruption': False, 'mlm_weight': 0.}

    # Frozen macro definition: protein 0 has 2+ / 2-; protein 1 has 1+ / 1- and is excluded.
    rows = np.array([[0, 2, 1], [0, 3, 1], [0, 4, 0], [0, 5, 0], [1, 6, 1], [1, 7, 0]])
    measured = evaluate(rows, np.array([3., 2., 1., 0., -2., 4.]))
    assert measured['macro_eligible_proteins'] == 1 and measured['macro_ap'] == 1.
    rules = campaign['development_promotion']
    good = [{'ap': .006, 'auroc': 0., 'macro_ap': 0.} for _ in range(3)]
    assert promotion(good, rules)['promoted']
    # A high mean must not conceal a single damaging fold or macro-AUPR deterioration.
    assert not promotion([{'ap': x, 'auroc': 0., 'macro_ap': 0.} for x in [.03, .03, -.011]], rules)['promoted']
    assert not promotion([{**d, 'macro_ap': -.006} for d in good], rules)['promoted']
    assert not promotion([{**d, 'auroc': -.006} for d in good], rules)['promoted']
    assert not promotion([{'ap': x, 'auroc': 0., 'macro_ap': 0.} for x in [.03, 0., 0.]], rules)['promoted']
    assert not promotion([{**d, 'ap': .004} for d in good], rules)['promoted']

    # Isolate the proposed clean readout's initial behavior and padding exclusion.
    hidden = torch.randn(4, 9, 64)
    chains = torch.tensor([[0, 0, 0, 0, 1, 1, 1, 1, -1]] * 4)
    residues = torch.tensor([[False, True, True, False, True, True, True, False, False]] * 4)
    outputs, counts = {}, {}
    for head in ['cls_linear', 'cls_mlp', 'residue_mean']:
        torch.manual_seed(2)
        model = PairModel(None, {**reference, 'readout': head}, tiny=True)
        outputs[head] = model.classify(hidden, chains, residues).detach()
        counts[head] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        if head == 'residue_mean':
            torch.nn.init.normal_(model.readout_output.weight)
            masked = hidden.clone()
            masked[:, [3, 7, 8]] += 1000
            torch.testing.assert_close(model.classify(hidden, chains, residues),
                                       model.classify(masked, chains, residues), rtol=0, atol=0)
    assert torch.equal(outputs['cls_linear'], outputs['cls_mlp'])
    assert torch.equal(outputs['cls_linear'], outputs['residue_mean'])
    assert counts['cls_mlp'] == counts['residue_mean'] > counts['cls_linear']
    sources = {str(p.relative_to(ROOT)): sha256(p) for p in (ROOT / 'configs').glob('*.json')}
    sources.update({f'scripts/{n}': sha256(ROOT / 'scripts' / n) for n in ['qualify_v3.py', 'metrics.py', 'make_configs.py']})
    sources.update(source_reports)
    atomic_json(ROOT / 'qualification/v3-contract.json', {
        'passed': True, 'code': core, 'data_manifest_sha256': sha256(path / 'manifest.json'),
        'source_sha256': sources, 'initial_runs': initial, 'folds': folds,
        'active_sequences': len(active), 'no_test_tokens': True, 'unchanged_v2_data': True,
        'core_byte_identical_to_qualified_v2_release': True,
        'inherited_qualification_scope': 'Exact unchanged training/model/data/checkpoint engine; v3 launcher is separately tested. No claim of a new v3 real scheduler requeue.',
        'objectives_explicitly_clean': True, 'controlled_configurations_checked': 19,
        'folds_disjoint_and_no_dropped_sampling_rows': True, 'macro_eligibility_and_promotion_guards_verified': True,
        'residual_head_counts_tiny': counts, 'zero_residual_preserves_initial_logits': True,
        'pooling_excludes_special_and_padding_tokens': True})
    print('V3 data/configuration/decision qualification passed; unchanged core inherits pinned v2 evidence.')


if __name__ == '__main__':
    main()
