"""Numerical and input qualification using validation pairs only."""
import gc
import json
import time

import numpy as np
import torch
from transformers import AutoTokenizer

from common import ROOT, PairData, atomic_json, configure, load_model, predict, prediction_fingerprint


def main():
    device = configure()
    data = PairData(ROOT / 'data', 'val')
    ordering = np.argsort(data.lengths, kind='stable')
    eligible = ordering[data.lengths[ordering] <= 3000]
    sample = eligible[np.linspace(0, len(eligible) - 1, 24, dtype=int)]
    selection = json.loads((ROOT / 'provenance/selection.json').read_text())
    tok = AutoTokenizer.from_pretrained(selection['base_model_path'], local_files_only=True)
    input_checks = []
    for index in sample:
        left, right = map(int, data.rows[index, :2])
        a = ''.join(tok.convert_ids_to_tokens(data.sequence(left).tolist()))
        b = ''.join(tok.convert_ids_to_tokens(data.sequence(right).tolist()))
        expected = tok(a, b, truncation=False)['input_ids']
        actual = data.batch([int(index)], 0, 2, False, device)['input_ids'][0].cpu().tolist()
        assert expected == actual, ('tokenization mismatch', int(index))
        input_checks.append(int(index))
    checks = {}
    for name in ('reference-seed2', 'symmetric-seed2', 'native-bernett'):
        model = load_model(name, device)
        fresh = np.concatenate([predict(model, data, [int(i)], device) for i in sample])
        item = {'sample_rows': sample.tolist(), 'finite_predictions': True}
        if name != 'native-bernett':
            with np.load(ROOT / 'predictions' / f'{name}-selected-validation.npz', allow_pickle=False) as saved:
                original = saved['predictions'][sample]
            assert np.array_equal(fresh[:, :2], original[:, :2])
            delta = np.abs(fresh[:, 2:] - original[:, 2:])
            item['max_logit_difference_from_saved_validation'] = float(delta.max())
            # BF16 batched GEMM can differ slightly from singleton evaluation.
            assert delta.max() <= 0.125, item
        else:
            fp32 = np.concatenate([predict(model, data, [int(i)], device, bf16=False) for i in sample])
            delta = np.abs(fresh[:, 2:] - fp32[:, 2:])
            prob_delta = np.abs(1 / (1 + np.exp(-fresh[:, 2:])) - 1 / (1 + np.exp(-fp32[:, 2:])))
            item['bf16_vs_fp32_max_logit_difference'] = float(delta.max())
            item['bf16_vs_fp32_max_probability_difference'] = float(prob_delta.max())
            assert prob_delta.max() <= 0.03, item
            # Compare efficient attention with the original HF eager attention, FP32 for both.
            eager = load_model(name, device, backend='eager')
            ordinary = np.concatenate([predict(eager, data, [int(i)], device, bf16=False) for i in sample])
            gap = np.abs(ordinary[:, 2:] - fp32[:, 2:])
            item['efficient_vs_eager_fp32_max_logit_difference'] = float(gap.max())
            assert np.allclose(ordinary[:, 2:], fp32[:, 2:], atol=2e-4, rtol=2e-4), item
            del eager
            gc.collect()
            torch.cuda.empty_cache()
            longest = int(np.argmax(data.lengths))
            start = time.monotonic()
            long_prediction = predict(model, data, [longest], device)
            item['longest_validation'] = {'row': longest, 'tokens': int(data.lengths[longest]),
                'logits': long_prediction[:, 2:].tolist(), 'elapsed_seconds': time.monotonic() - start}
        checks[name] = item
        print(json.dumps({'model': name, 'qualification': item}), flush=True)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    report = {'passed': True, 'prediction_fingerprint': prediction_fingerprint(),
              'selection_only_uses_validation': True, 'tokenizer_checked_rows': input_checks,
              'checks': checks, 'device': torch.cuda.get_device_name(), 'torch': torch.__version__}
    atomic_json(ROOT / 'provenance/qualification.json', report)


if __name__ == '__main__':
    main()
