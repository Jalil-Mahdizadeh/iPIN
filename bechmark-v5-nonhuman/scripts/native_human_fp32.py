"""Full-precision humanV11 inference, independently checked against archived native logits.

The BF16 pilot failed its predeclared numeric tolerance. No test metric was
calculated or used to choose precision. Keep the active BF16 implementations
for the other checkpoints immutable.
"""
import argparse
import hashlib
import json
import os
import numpy as np
import torch
from scipy.special import expit
import pair_infer as pair
from bench_utils import ROOT, atomic, cuda, load_npz, now, sha

base_predict = pair.predict
base_signature = pair.signature

def signature(name):
    assert name == 'native-human'
    return hashlib.sha256(json.dumps({'base_signature': base_signature(name),
        'wrapper_sha256': sha(__file__), 'precision': 'FP32 parameters and computation; TF32 disabled'},
        sort_keys=True).encode()).hexdigest()

def predict(model, name, data, ids, device, fp32=True):
    return base_predict(model, name, data, ids, device, fp32=True)

pair.signature = signature
pair.predict = predict

def qualify(model, device):
    gold = load_npz(ROOT/'data/native-human-qualification.npz')
    data = pair.test_data()
    cases = []
    singleton = {}
    for row, col, ref in zip(gold['indices'], gold['columns'], gold['logits']):
        z = predict(model, 'native-human', data, np.array([int(row)]), device)[0]
        le, pe = float(abs(z[int(col)]-ref)), float(abs(expit(z[int(col)])-expit(ref)))
        assert le <= .002 and pe <= .0001, (int(row), le, pe)
        singleton[int(row)] = z
        cases.append({'union_row': int(row), 'orientation': int(col), 'reference_logit': float(ref),
            'fresh_fp32_logit': float(z[int(col)]), 'logit_error': le, 'probability_error': pe})
    ids = np.array(sorted(singleton, key=lambda i: data.lengths[i]))
    padded_errors = []
    for pos in data.microbatches(ids, 16384, 8):
        z = predict(model, 'native-human', data, ids[pos], device)
        expected = np.array([singleton[int(i)] for i in ids[pos]])
        le, pe = float(abs(z-expected).max()), float(abs(expit(z)-expit(expected)).max())
        assert le <= .002 and pe <= .0001, (le, pe)
        padded_errors.append({'indices': ids[pos].tolist(), 'logit_error': le, 'probability_error': pe})
    result = {'passed': True, 'at_utc': now(), 'fingerprint': signature('native-human'),
        'precision': 'FP32, TF32 disabled', 'source': 'Independent original-runtime FP32 archived logits; full supplied sequences',
        'cases': cases, 'padding_checks': padded_errors, 'test_metrics_read': False,
        'torch': torch.__version__, 'device': torch.cuda.get_device_name(device)}
    atomic(ROOT/'qualification/native-human.json', result)
    print(json.dumps({'event': 'qualification_passed', **result}), flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--qualify', action='store_true')
    args = parser.parse_args()
    device = cuda(int(os.environ.get('LOCAL_RANK', '0')))
    model = pair.load_model('native-human', device)
    if args.qualify:
        qualify(model, device)
    else:
        pair.infer('native-human', model, device, int(os.environ.get('RANK','0')), int(os.environ.get('WORLD_SIZE','1')))

if __name__ == '__main__':
    main()
