"""Verify reuse integrity and qualify the new weights on validation data only."""
import datetime
import gc
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from transformers import AutoTokenizer
from common import ROOT, PairData, atomic_json, configure, load_model, predict, prediction_fingerprint, sha256
from frozen_v2_data import PairData as V2Data
from frozen_v2_model import PairModel as V2Model

def check_array(path, digest, data):
    assert sha256(path) == digest, str(path)
    with np.load(path, allow_pickle=False) as payload:
        a = payload['predictions']
    assert a.shape == (len(data), 4) and np.isfinite(a).all()
    assert np.array_equal(a[:, 0], np.arange(len(data))) and np.array_equal(a[:, 1], data.rows[:, 2])
    return a

def check_source_chunks(old_name, split, expected, fingerprint):
    folder = ROOT.parent / 'benchmark-v1/predictions' / f'{old_name}-{split}'
    arrays = []
    for rank in range(4):
        manifest = json.loads((folder / f'rank-{rank:02d}.done.json').read_text())
        assert manifest['fingerprint'] == fingerprint and manifest['rank'] == rank and manifest['world_size'] == 4
        count = 0
        for chunk in manifest['chunks']:
            path = folder / chunk['file']
            assert sha256(path) == chunk['sha256']
            with np.load(path, allow_pickle=False) as payload:
                a = payload['predictions']
            assert a.shape == (chunk['rows'], 4)
            arrays.append(a)
            count += len(a)
        assert count == manifest['rows']
    merged = np.concatenate(arrays)
    merged = merged[np.argsort(merged[:, 0])]
    assert np.array_equal(merged, expected), (old_name, split)
    return len(arrays)

def validation_statistics(a):
    y, scores = a[:, 1], a[:, 2:4].mean(1)
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-30)
    best = np.flatnonzero(f1 == f1.max())[-1]
    return {'ap': float(average_precision_score(y, scores)), 'auroc': float(roc_auc_score(y, scores)),
            'brier': float(np.mean((expit(scores)-y)**2)), 'max_f1_logit_threshold': float(thresholds[best]),
            'max_f1_probability_threshold': float(expit(thresholds[best])),
            'rule': 'maximum validation F1, highest threshold on exact ties'}

@torch.inference_mode()
def main():
    selection = json.loads((ROOT / 'provenance/selection.json').read_text())
    cfg = json.loads((ROOT / 'config.json').read_text())
    prior = json.loads((ROOT / 'provenance/benchmark-v1-qualification.json').read_text())
    assert prior['passed']
    for name, digest in selection['data']['sha256'].items():
        assert sha256(ROOT / 'data' / name) == digest
    data, test = PairData(ROOT / 'data', 'val'), PairData(ROOT / 'data', 'test')
    assert len(data) == 59258 and int(data.rows[:, 2].sum()) == 29628
    assert len(test) == 52048 and int(test.rows[:, 2].sum()) == 26024 and test.lengths.max() == 7426
    training_data = V2Data(Path(selection['v2_prepared_path']), 'official', 'val')
    assert np.array_equal(training_data.rows, data.rows) and np.array_equal(training_data.lengths, data.lengths)
    proteins = np.unique(data.rows[:, :2])
    assert all(np.array_equal(data.sequence(i), training_data.sequence(i)) for i in proteins)
    validation, reuse = {}, {}
    for name in cfg['comparison_models']:
        item = selection['models'][name]
        if name in selection['reused_predictions']:
            sources = selection['reused_predictions'][name]
            a = check_array(ROOT / sources['test']['path'], sources['test']['sha256'], test)
            chunks = check_source_chunks(item['reused_from_model'], 'test', a, prior['prediction_fingerprint'])
            a = check_array(ROOT / sources['val']['path'], sources['val']['sha256'], data)
            if name == 'native-bernett':
                chunks += check_source_chunks('native-bernett', 'val', a, prior['prediction_fingerprint'])
            reuse[name] = {'test_rows': len(test), 'val_rows': len(data), 'source_chunks_verified': chunks,
                           'merged_predictions_match_source_chunks': True}
        else:
            source = item['selected_validation']
            a = check_array(ROOT / source['path'], source['sha256'], data)
        validation[name] = validation_statistics(a)
        if 'validation_primary_ap' in item and name.startswith('v2-'):
            assert abs(validation[name]['ap'] - item['validation_primary_ap']) < 1e-12
    atomic_json(ROOT / 'provenance/validation-before-test.json', {
        'fixed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'metrics_and_thresholds': validation, 'test_used_for_threshold_selection': False})
    print(json.dumps({'event': 'reuse_and_data_verified', 'models': reuse,
                      'validation_sequences_matched': len(proteins)}), flush=True)

    device = configure()
    lengths = data.lengths
    eligible = np.argsort(lengths, kind='stable')
    eligible = eligible[lengths[eligible] <= 3000]
    sample = set(eligible[np.linspace(0, len(eligible)-1, 24, dtype=int)].tolist())
    batches = []
    for rank in range(4):
        rows = np.arange(rank, len(data), 4)
        rows = rows[np.argsort(lengths[rows], kind='stable')]
        for batch in data.microbatches(rows, cfg['token_budget'], cfg['max_pairs']):
            if sample.intersection(batch): batches.append(batch)
    tokenizer = AutoTokenizer.from_pretrained(selection['base_model_path'], local_files_only=True)
    for i in sorted(sample):
        a, b = data.rows[i, :2]
        seqs = [''.join(tokenizer.convert_ids_to_tokens(data.sequence(x).tolist())) for x in [a, b]]
        expected = tokenizer(*seqs, truncation=False)['input_ids']
        actual = data.batch([i], 0, 2, False, device)['input_ids'][0].cpu().tolist()
        assert expected == actual
    checks = {}
    for name in cfg['models']:
        entry = selection['models'][name]
        contract = json.loads((ROOT / 'provenance' / f'{name}-contract.json').read_text())
        assert sha256(ROOT / 'scripts/frozen_v2_model.py') == contract['code']['model.py']
        assert sha256(ROOT / 'scripts/frozen_v2_data.py') == contract['code']['data.py']
        model = load_model(name, device)
        saved = torch.load(ROOT / entry['path'], map_location='cpu', mmap=True, weights_only=False)
        original = V2Model(selection['base_model_path'], entry['configuration'])
        original.load_state_dict(saved['model'], strict=True)
        del saved
        original.esm_mask.gradient_checkpointing_disable()
        original.eval().requires_grad_(False).to(device)
        source = entry['selected_validation']
        old = check_array(ROOT / source['path'], source['sha256'], data)
        maximum_gap, implementation_gap, count = 0., 0., 0
        for batch in batches:
            fresh = predict(model, data, batch, device)
            features = training_data.batch(np.array(batch), np.zeros(len(batch), dtype=np.int64), 2, False, device)
            common_features = data.batch(batch, 0, 2, False, device)
            assert torch.equal(features['clean_ids'], common_features['input_ids'])
            assert torch.equal(features['attention_mask'], common_features['attention_mask'])
            with torch.autocast('cuda', dtype=torch.bfloat16):
                original_scores = original(**features, compute_loss=False).float().cpu().numpy()
            gap = float(np.abs(fresh[:, 2:4] - original_scores).max())
            implementation_gap = max(implementation_gap, gap)
            assert gap == 0., (name, 'BF16 forward implementation mismatch', gap)
            delta = float(np.abs(fresh[:, 2:4] - old[batch, 2:4]).max())
            maximum_gap = max(maximum_gap, delta)
            assert delta <= 0.03125, (name, 'saved validation mismatch', delta)
            count += len(batch)
        batch = batches[len(batches)//2]
        fresh_fp32 = predict(model, data, batch, device, bf16=False)
        features = training_data.batch(np.array(batch), np.zeros(len(batch), dtype=np.int64), 2, False, device)
        original_fp32 = original(**features, compute_loss=False).float().cpu().numpy()
        fp32_gap = float(np.abs(fresh_fp32[:, 2:4] - original_fp32).max())
        assert fp32_gap <= 2e-5
        if name == cfg['models'][0]:
            longest = int(np.argmax(data.lengths))
            long_prediction = predict(model, data, [longest], device)
            long_check = {'row': longest, 'tokens': int(lengths[longest]), 'finite': bool(np.isfinite(long_prediction).all())}
        checks[name] = {'validated_pairs': count, 'batches': len(batches),
            'max_gap_from_saved_validation': maximum_gap, 'max_bf16_gap_between_implementations': implementation_gap,
            'max_fp32_gap_between_implementations': fp32_gap, 'strict_weight_loading': True}
        print(json.dumps({'event': 'model_qualified', 'model': name, 'checks': checks[name]}), flush=True)
        del original, model
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(ROOT / 'provenance/qualification.json', {'passed': True,
        'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'prediction_fingerprint': prediction_fingerprint(), 'reuse': reuse, 'checks': checks,
        'longest_validation': long_check, 'tokenizer_sample_rows': sorted(sample),
        'all_validation_sequences_match_training': True,
        'new_gpu_inference_uses_validation_only': True, 'device': torch.cuda.get_device_name(),
        'torch': torch.__version__, 'cuda': torch.version.cuda,
        'prior_native_precision_qualification': prior,
        'test_metrics_read_for_selection': False})
    print('QUALIFICATION PASSED', flush=True)

if __name__ == '__main__': main()
