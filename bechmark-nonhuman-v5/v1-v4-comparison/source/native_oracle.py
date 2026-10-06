"""Frozen inference equivalent to the authors' PLMinteract.forward_test.

The base configuration is constructed locally; strict loading supplies EVERY
parameter, including the unused MLM head. No pretrained downloads or training.
"""
import hashlib
import platform
import time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
import transformers
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path('/opt/plm_interact/assets')
CHECKPOINTS = {
    'human': ASSETS / 'humanV11/pytorch_model.bin',
    'bernett': ROOT / 'checkpoints/PLM-interact-650M-Leakage-Free-Dataset/pytorch_model.bin',
    'mutation': ROOT / 'checkpoints/PLM-interact-650M-Mutation/pytorch_model.bin',
}
EXPECTED_SHA256 = {
    'human': '68c50e1dc84ee3cb6c08a7c83eefb382a29a3c1237fd577986854f746c63d665',
    'bernett': '207f1bf02e9bebc393fe510a98c0f9af04039d0a9258047ac107c6d24221c7e2',
    'mutation': '8a2116369bb706a5cde10029db1ee3b0e841d739b8aefe72d215c5a0b9d9931b',
}

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(16 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def configure():
    torch.set_num_threads(8)
    torch.manual_seed(20260928)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)

class NativePLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = AutoConfig.from_pretrained(ASSETS / 'esm2_650m', local_files_only=True)
        self.esm_mask = AutoModelForMaskedLM.from_config(self.config)
        self.classifier = nn.Linear(1280, 1)

    def forward(self, features):
        hidden = self.esm_mask.base_model(**features, return_dict=True).last_hidden_state[:, 0, :]
        return self.classifier(F.relu(hidden)).view(-1)

def load_model(name, device='cuda', verify_hash=True):
    start = time.monotonic()
    path = CHECKPOINTS[name]
    digest = sha256(path) if verify_hash else EXPECTED_SHA256[name]
    assert digest == EXPECTED_SHA256[name], (name, digest)
    state = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    original_prefix = ''
    if name == 'mutation':
        assert all(k.startswith('PLM.') for k in state)
        state = {k.removeprefix('PLM.'): v for k, v in state.items()}
        original_prefix = 'PLM.'
    model = NativePLM()
    model.load_state_dict(state, strict=True)
    assert all(torch.equal(v, state[k]) for k, v in model.state_dict().items())
    assert all(v.dtype == torch.float32 for v in model.parameters())
    model.eval().requires_grad_(False).to(device)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    tokenizer = AutoTokenizer.from_pretrained(ASSETS / 'esm2_650m', local_files_only=True)
    tokenizer.pad_token = tokenizer.eos_token  # Explicitly follows upstream.
    return model, tokenizer, {
        'checkpoint': name, 'path': str(path), 'sha256': digest,
        'strict_load': True, 'all_loaded_tensors_equal': True,
        'state_keys': len(state), 'removed_prefix': original_prefix,
        'parameters': sum(p.numel() for p in model.parameters()),
        'requires_grad': False, 'precision': 'float32', 'amp': False, 'tf32': False,
        'python': platform.python_version(), 'torch': torch.__version__,
        'transformers': transformers.__version__, 'cuda': torch.version.cuda,
        'device': torch.cuda.get_device_name(), 'load_seconds': time.monotonic()-start,
    }

def tokenize(tokenizer, pairs, max_length=None, padding=True):
    a, b = zip(*pairs)
    args = dict(padding=padding, return_tensors='pt')
    if max_length is not None:
        args.update(truncation='longest_first', max_length=max_length)
    else:
        args['truncation'] = False
    features = tokenizer([s.strip() for s in a], [s.strip() for s in b], **args)
    return features
