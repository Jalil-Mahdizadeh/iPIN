"""Synthetic resource profile only: no optimizer, parameter update, or biological data.

Run in the existing ARM64 SIF. The model checkpoint is used solely to instantiate
the exact architecture/dtype. Results cannot establish learning or model quality.
"""
import argparse
import gc
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import transformers
from transformers import AutoConfig, AutoModelForMaskedLM

OUT = Path(__file__).resolve().parent
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--extended-lengths', action='store_true')
args = parser.parse_args()
result_path = OUT / ('synthetic-long-profile.json' if args.extended_lengths else 'synthetic-profile.json')
ASSETS = Path('/opt/plm_interact/assets')
torch.set_num_threads(8)
torch.manual_seed(20260928)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

class ProfileModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.esm_mask = AutoModelForMaskedLM.from_config(
            AutoConfig.from_pretrained(ASSETS / 'esm2_650m', local_files_only=True))
        self.classifier = torch.nn.Linear(1280, 1)

model = ProfileModel()
checkpoint = ASSETS / 'humanV11/pytorch_model.bin'
state = torch.load(checkpoint, mmap=True, weights_only=True, map_location='cpu')
model.load_state_dict(state, strict=True)
model.to('cuda').train()
params = sum(p.numel() for p in model.parameters())
assert model.esm_mask.config.hidden_dropout_prob == 0
assert model.esm_mask.config.attention_probs_dropout_prob == 0

def synthetic(batch, length):
    gen = torch.Generator(device='cuda').manual_seed(734)
    ids = torch.randint(4, 24, (batch, length), generator=gen, device='cuda')
    ids[:, 0] = 0
    ids[:, length // 2] = 2
    ids[:, -1] = 2
    masked = torch.rand((batch, length), generator=gen, device='cuda') < .15
    masked[:, [0, length // 2, length - 1]] = False
    targets = ids.clone()
    targets[~masked] = -100
    ids[masked] = 32
    return dict(input_ids=ids, attention_mask=torch.ones_like(ids)), targets, (torch.arange(batch, device='cuda') % 2).float()

def objective(mode, features, targets, labels):
    if mode == 'native_two_pass':
        mlm = model.esm_mask(**features, labels=targets).loss
        hidden = model.esm_mask.base_model(**features).last_hidden_state
    else:
        hidden = model.esm_mask.base_model(**features).last_hidden_state
        scores = model.esm_mask.lm_head(hidden)
        mlm = F.cross_entropy(scores.reshape(-1, 33).float(), targets.reshape(-1))
    logits = model.classifier(F.relu(hidden[:, 0, :])).reshape(-1)
    classification = F.binary_cross_entropy_with_logits(logits.float(), labels, pos_weight=torch.tensor([10.], device='cuda'))
    return mlm + 10 * classification

result = {
    'timestamp_utc': datetime.now(timezone.utc).isoformat(),
    'purpose': 'synthetic no-update forward/backward feasibility profile',
    'optimizer_created': False, 'optimizer_steps': 0, 'biological_sequences_used': False,
    'python': platform.python_version(), 'machine': platform.machine(),
    'torch': torch.__version__, 'transformers': transformers.__version__,
    'cuda': torch.version.cuda, 'nccl': list(torch.cuda.nccl.version()),
    'device': torch.cuda.get_device_name(),
    'cuda_total_memory_bytes': torch.cuda.get_device_properties(0).total_memory,
    'bf16_supported': torch.cuda.is_bf16_supported(),
    'native_esm_supports_sdpa': model.esm_mask._supports_sdpa,
    'parameters': params, 'parameter_dtype': 'float32',
    'checkpoint_used_for_architecture_profile': str(checkpoint),
    'extra_fp32_adam_moments_bytes': params * 8,
    'possible_extra_fp32_ddp_bucket_bytes': params * 4,
    'measurements': [],
}

# Qualify the algebraic sharing change on a small deterministic FP32 fixture.
features, targets, labels = synthetic(2, 65)
loss = objective('native_two_pass', features, targets, labels)
loss.backward()
ref_loss = loss.item()
ref_grad = {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters() if p.grad is not None}
model.zero_grad(set_to_none=True)
del loss
shared_loss = objective('shared_pass', features, targets, labels)
shared_loss.backward()
max_abs = 0.
delta_sq, ref_sq = 0., 0.
for n, p in model.named_parameters():
    assert (p.grad is None) == (n not in ref_grad)
    if p.grad is not None:
        g = p.grad.detach().cpu()
        d = g - ref_grad[n]
        max_abs = max(max_abs, float(d.abs().max()))
        delta_sq += float(d.double().square().sum())
        ref_sq += float(ref_grad[n].double().square().sum())
result['shared_pass_fp32_qualification'] = {
    'batch': 2, 'tokens': 65, 'native_loss': ref_loss,
    'shared_loss': shared_loss.item(), 'loss_abs_diff': abs(ref_loss - shared_loss.item()),
    'gradient_max_abs_diff': max_abs,
    'gradient_relative_l2_diff': (delta_sq / ref_sq) ** .5,
    'gradient_tensors_compared': len(ref_grad),
    'dropout_probabilities': [0., 0.],
}
del features, targets, labels, ref_grad, shared_loss, g, d
model.zero_grad(set_to_none=True)
gc.collect()
torch.cuda.empty_cache()
print(json.dumps({'qualification': result['shared_pass_fp32_qualification']}), flush=True)

cases = [
    ('native_two_pass', False, 1, 1024),
    ('shared_pass', False, 1, 1024),
    ('shared_pass', False, 4, 1024),
    ('shared_pass', False, 1, 1603),
    ('shared_pass', True, 1, 1603),
    ('shared_pass', True, 2, 1603),
    ('shared_pass', True, 1, 2204),
    ('shared_pass', True, 1, 3833),
]
if args.extended_lengths:
    cases = [
        ('shared_pass', False, 4, 1603),
        ('shared_pass', True, 4, 2204),
        ('shared_pass', True, 2, 3833),
        ('shared_pass', True, 1, 7426),
    ]
for mode, checkpointing, batch, length in cases:
    if checkpointing:
        model.esm_mask.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
    else:
        model.esm_mask.gradient_checkpointing_disable()
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    features, targets, labels = synthetic(batch, length)
    times, losses = [], []
    row = {'mode': mode, 'gradient_checkpointing': checkpointing, 'batch': batch, 'tokens': length,
           'autocast': 'bfloat16', 'attention': 'native eager', 'warmup_passes': 1, 'measured_passes': 3}
    try:
        for step in range(4):
            model.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            if step == 1:
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss = objective(mode, features, targets, labels)
            loss.backward()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            assert torch.isfinite(loss)
            assert all(torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None)
            if step:
                times.append(elapsed)
                losses.append(loss.item())
            del loss
        row.update(status='ok', seconds=times, median_seconds=statistics.median(times),
                   examples_per_second=batch / statistics.median(times),
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                   peak_reserved_bytes=torch.cuda.max_memory_reserved(), losses=losses)
    except torch.cuda.OutOfMemoryError as e:
        row.update(status='out_of_memory', error=str(e))
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()
    result['measurements'].append(row)
    print(json.dumps(row), flush=True)
    result_path.write_text(json.dumps(result, indent=2) + '\n')
    del features, targets, labels

model.zero_grad(set_to_none=True)
unchanged = all(torch.equal(value.detach().cpu(), state[name]) for name, value in model.state_dict().items())
assert unchanged
result['all_state_tensors_unchanged'] = unchanged
result['finished_utc'] = datetime.now(timezone.utc).isoformat()
result_path.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps({'all_state_tensors_unchanged': unchanged, 'optimizer_steps': 0}), flush=True)
