"""Check the selected checkpoint's inference using official validation only."""
import datetime
import gc
import importlib.metadata
import os
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from common import (ROOT, PairData, TrainingData, atomic_json, batch_features, configure,
                    load_checkpoint, load_model, predict, prediction_fingerprint, require_continue, sha256)
from prepare_selected import check_array


def validation_statistics(a):
    y, scores = a[:, 1], a[:, 2:4].mean(1)
    p, r, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * p[:-1] * r[:-1] / np.maximum(p[:-1] + r[:-1], 1e-30)
    idx = np.flatnonzero(f1 == f1.max())[-1]
    return {'ap': float(average_precision_score(y, scores)), 'auroc': float(roc_auc_score(y, scores)),
        'brier': float(np.mean((expit(scores) - y)**2)), 'max_f1_logit_threshold': float(thresholds[idx]),
        'max_f1_probability_threshold': float(expit(thresholds[idx])),
        'rule': 'maximum validation F1, highest threshold on exact ties'}


def validation_data(plan):
    data = PairData(ROOT / 'data', 'val')
    original = TrainingData(Path(plan['training_prepared_path']), 'official', 'val')
    assert len(data) == 59258 and int(data.rows[:, 2].sum()) == 29628
    assert np.array_equal(data.rows, original.rows) and np.array_equal(data.lengths, original.lengths)
    assert all(np.array_equal(data.sequence(i), original.sequence(i)) for i in np.unique(data.rows[:, :2]))
    return data, original


@torch.inference_mode()
def forward_checks(model, data, original, device, reference=None):
    cfg = json.loads((ROOT / 'config.json').read_text())
    ids = np.argsort(data.lengths, kind='stable')
    ids = ids[data.lengths[ids] <= 3000]
    sample = set(ids[np.linspace(0, len(ids) - 1, 24, dtype=int)].tolist())
    batches = []
    for rank in range(4):
        rows = np.arange(rank, len(data), 4)
        rows = rows[np.argsort(data.lengths[rows], kind='stable')]
        for batch in data.microbatches(rows, cfg['token_budget'], cfg['max_pairs']):
            if sample.intersection(batch): batches.append(batch)
    saved_gap, adapter_gap, count = 0., 0., 0
    for batch in batches:
        require_continue()
        features = batch_features(data, batch, device)
        expected = original.batch(np.array(batch), np.zeros(len(batch), dtype=np.int64), 2, False, device)
        assert all(torch.equal(features[k], expected[k]) for k in expected)
        old = data.batch(batch, 0, 2, False, device)
        assert torch.equal(features['clean_ids'], old['input_ids'])
        assert torch.equal(features['attention_mask'], old['attention_mask'])
        fresh = predict(model, data, batch, device)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            score = model(**expected, compute_loss=False).float().cpu().numpy()
        gap = float(np.abs(fresh[:, 2:4] - score).max())
        adapter_gap = max(adapter_gap, gap)
        assert gap == 0., 'Training/benchmark adapter must give identical BF16 logits'
        if reference is not None:
            delta = float(np.abs(fresh[:, 2:4] - reference[batch, 2:4]).max())
            saved_gap = max(saved_gap, delta)
            assert delta <= 0.03125, ('Saved validation mismatch', delta)
        count += len(batch)
    longest = int(np.argmax(data.lengths))
    require_continue()
    torch.cuda.reset_peak_memory_stats()
    z = predict(model, data, [longest], device)
    assert data.lengths[longest] == 39391
    return {'validated_pairs': count, 'batches': len(batches), 'sample_rows': sorted(sample),
        'max_bf16_gap_between_adapters': adapter_gap,
        'max_gap_from_saved_validation': saved_gap if reference is not None else None,
        'saved_validation_comparison_performed': reference is not None,
        'saved_validation_tolerance': 0.03125,
        'strict_weight_loading': True, 'all_official_validation_sequences_match_training': True,
        'longest_validation': {'row': longest, 'tokens': int(data.lengths[longest]),
            'finite': bool(np.isfinite(z).all()), 'peak_gpu_bytes': torch.cuda.max_memory_allocated()}}


def main():
    selection = json.loads((ROOT / 'provenance/selection.json').read_text())
    cfg = json.loads((ROOT / 'config.json').read_text())
    for name, digest in selection['data']['sha256'].items():
        assert sha256(ROOT / 'data' / name) == digest
    contract = json.loads((ROOT / 'provenance/esmc-standard-contract.json').read_text())
    assert torch.__version__ == contract['torch'] and torch.version.cuda == contract['cuda']
    assert os.environ.get('PLMI_V4_IMAGE_SHA256') == contract['image_sha256']
    assert all(importlib.metadata.version(name) == version for name, version in contract['packages'].items())
    from esm.tokenization import EsmSequenceTokenizer
    vocabulary = (ROOT / 'provenance/input-vocabulary.txt').read_text().splitlines()
    tokenizer = EsmSequenceTokenizer()
    tokens = np.load(ROOT / 'data/tokens.npy', mmap_mode='r')
    for token in np.unique(tokens):
        symbol = vocabulary[int(token)]
        assert len(symbol) == 1 and tokenizer.encode(symbol, add_special_tokens=False) == [int(token)]
    assert (tokenizer.cls_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id) == (0, 1, 2)
    data, original = validation_data(selection)
    validation = {}
    for name in cfg['comparison_models']:
        item = (selection['reused_predictions'][name]['val'] if name in selection['reused_predictions']
                else selection['models'][name]['selected_validation'])
        a = check_array(ROOT / item['path'], item['sha256'], data)
        validation[name] = validation_statistics(a)
    entry = selection['models']['v4-esmc-standard']
    assert abs(validation['v4-esmc-standard']['ap'] - entry['validation_primary_ap']) < 1e-12
    thresholds_path = ROOT / 'provenance/validation-before-test.json'
    if thresholds_path.exists():
        frozen = json.loads(thresholds_path.read_text())
        assert frozen['metrics_and_thresholds'] == validation
        assert frozen['test_used_for_threshold_selection'] is False
    else:
        atomic_json(thresholds_path, {
            'fixed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'metrics_and_thresholds': validation, 'test_used_for_threshold_selection': False})
    qpath = ROOT / 'provenance/qualification.json'
    if qpath.exists():
        old = json.loads(qpath.read_text())
        assert old['passed'] and old['prediction_fingerprint'] == prediction_fingerprint()
        print('Selected-checkpoint forward checks already passed; resuming.', flush=True)
        return
    device = configure()
    model = load_model('v4-esmc-standard', device)
    source = entry['selected_validation']
    reference = check_array(ROOT / source['path'], source['sha256'], data)
    checks = forward_checks(model, data, original, device, reference)
    atomic_json(qpath, {'passed': True, 'prediction_fingerprint': prediction_fingerprint(),
        'completed_at_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'checks': checks, 'new_gpu_inference_uses_validation_only': True, 'all_cached_sequence_tokens_match_esmc_vocabulary': True,
        'training_runtime_versions_match': True,
        'torch': torch.__version__, 'cuda': torch.version.cuda, 'device': torch.cuda.get_device_name(),
        'historical_native_precision_evidence': 'provenance/benchmark-v2-qualification.json'})
    del model
    gc.collect(); torch.cuda.empty_cache()
    print(json.dumps({'event': 'selected_checkpoint_forward_passed', 'checks': checks}), flush=True)


if __name__ == '__main__': main()
