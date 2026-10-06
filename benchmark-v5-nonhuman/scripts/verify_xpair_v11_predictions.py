"""Independently reproduce saved V11 scores from raw features in every species."""
import numpy as np
import torch

import xpair_v11 as runner
from bench_utils import ROOT, atomic, cuda, load_npz, now, read, record, sha


@torch.inference_mode()
def main():
    done_path = ROOT / 'predictions/xpair-v11/done.json'
    done = read(done_path)
    assert done['signature'] == runner.signature()
    assert sha(done['file']['path']) == done['file']['sha256']
    scores = load_npz(done['file']['path'])['scores']
    meta = read(ROOT / 'data/sequences.json')
    lengths = np.asarray(meta['length'])
    mapping = load_npz(ROOT / 'data/pair-mapping.npz')
    device = cuda(0)
    model = runner.load_model(device)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    cases = []
    for test in ['mouse', 'fly', 'worm', 'yeast', 'ecoli']:
        rows = np.load(ROOT / 'data' / f'{test}.npy')
        total_length = lengths[rows[:, :2]].sum(1)
        order = np.argsort(total_length, kind='stable')
        for i in [int(order[0]), int(order[len(order)//2]), int(order[-1])]:
            a, b = map(int, rows[i, :2])
            features = [torch.load(runner.base.verified_feature(k, meta),
                                   map_location=device, weights_only=True) for k in [a, b]]
            masks = [torch.ones(1, len(feature), dtype=torch.bool, device=device)
                     for feature in features]
            raw = model({'input1': (features[0][None], masks[0]),
                         'input2': (features[1][None], masks[1])}, task='interaction')[0]
            runner.base.clear_attention(model)
            value = float(raw.item())
            expected = float(scores[mapping[test][i]])
            error = abs(value-expected)
            assert np.isfinite(value) and error < 2e-4, (test, i, error)
            cases.append({'test': test, 'source_row_id': i, 'union_id': int(mapping[test][i]),
                          'length_a': int(lengths[a]), 'length_b': int(lengths[b]),
                          'native_logit': value, 'saved_logit': expected, 'absolute_error': error})
    assert all(torch.equal(value, model.state_dict()[key]) for key, value in before.items())
    atomic(ROOT / 'qualification/xpair-v11-production-check.json', {
        'at_utc': now(), 'passed': True, 'script': record(__file__),
        'completed_inference': record(done_path), 'signature': done['signature'],
        'cases': cases, 'max_absolute_logit_error': max(case['absolute_error'] for case in cases),
        'unmodified_native_full_forward': True, 'raw_features_reloaded_independently': True,
        'state_unchanged': True, 'fixture_selection': 'Shortest, median and longest combined length per species; labels unused',
        'test_performance_metrics_read': False,
    })
    print({'production_scores_checked_against_native': len(cases),
           'max_absolute_logit_error': max(case['absolute_error'] for case in cases)}, flush=True)


if __name__ == '__main__':
    main()
