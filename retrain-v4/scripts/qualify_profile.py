"""One longest train step and full longest validation inference for each v4 arm."""
import argparse
import gc
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from contracts import core_hashes, RUNS
from data import PairData
from model import PairModel
from state import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(); p.add_argument('--backbone', choices=['esm2', 'esmc'], required=True)
    args = p.parse_args()
    torch.set_num_threads(8); torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    train, val = [PairData(ROOT / 'data/prepared', 'official', split) for split in ['train', 'val']]
    records = []
    esm2_head_match = None
    for name in RUNS:
        cfg = json.loads((ROOT / 'configs' / (name + '.json')).read_text())
        if cfg['backbone'] != args.backbone: continue
        torch.manual_seed(2)
        model = PairModel(ROOT / 'assets' / cfg['backbone'], cfg).cuda()
        if cfg['backbone'] == 'esm2':
            def head_hash(m):
                h = hashlib.sha256()
                for n, value in m.named_parameters():
                    if n.startswith(('classifier.', 'readout_')):
                        h.update(n.encode()); h.update(value.detach().cpu().numpy().tobytes())
                return h.hexdigest()
            actual = head_hash(model)
            torch.manual_seed(2)
            control = PairModel(ROOT / 'assets/esm2', {**cfg, 'attention_mode': 'standard'})
            assert actual == head_hash(control)
            esm2_head_match = actual
            del control; gc.collect()
        groups = [dict(params=[p for p in model.parameters() if p.requires_grad and p.ndim >= 2], weight_decay=cfg['weight_decay']),
                  dict(params=[p for p in model.parameters() if p.requires_grad and p.ndim < 2], weight_decay=0.)]
        optimizer = torch.optim.AdamW(groups, lr=cfg['learning_rate'], foreach=False)
        for stage, data, training in [('longest-training', train, True), ('longest-validation', val, False)]:
            index = np.array([np.argmax(data.lengths)])
            batch = data.batch(index, np.zeros(1, dtype=int), 2, False, 'cuda')
            assert batch['clean_ids'].shape == (2, int(data.lengths[index[0]]))
            model.train(training); torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
            start = time.monotonic()
            if training:
                with torch.autocast('cuda', dtype=torch.bfloat16): loss, *_ = model(**batch)
                assert torch.isfinite(loss); loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['clip_grad_norm'], error_if_nonfinite=True)
                assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
                optimizer.step(); optimizer.zero_grad(set_to_none=True)
                assert all(torch.isfinite(p).all() for p in model.parameters())
                value = float(loss.detach()); del loss
            else:
                with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                    logits = model(**batch, compute_loss=False)
                assert torch.isfinite(logits).all(); value = logits.float().cpu().tolist(); del logits
            torch.cuda.synchronize()
            item = {'run': name, 'stage': stage, 'tokens': int(data.lengths[index[0]]),
                'orientations': 2, 'source_row': int(data.rows[index[0], 3]),
                'seconds': time.monotonic()-start, 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
                'peak_reserved_gib': torch.cuda.max_memory_reserved()/2**30, 'finite_output': value,
                'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)}
            records.append(item); print(json.dumps(item), flush=True); del batch
        del model, optimizer, groups; gc.collect(); torch.cuda.empty_cache()
    report = {'passed': True, 'profiles': records, 'backbone': args.backbone,
              'test_data_used': False, 'candidate_checkpoints_saved': False,
              'disposable_optimizer_steps': len(records)//2,
              'esm2_control_and_chain_initial_head_hash': esm2_head_match,
              'device': torch.cuda.get_device_name(), 'device_memory_gib': torch.cuda.get_device_properties(0).total_memory/2**30,
              'code': core_hashes(ROOT / 'scripts'),
              'source_sha256': {'scripts/qualify_profile.py': sha256(Path(__file__))}}
    atomic_json(ROOT / 'qualification' / ('full-length-' + args.backbone + '.json'), report)


if __name__ == '__main__': main()
