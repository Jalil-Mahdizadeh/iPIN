"""Unmodified authors' classes and strict original-checkpoint loading."""
import hashlib
from pathlib import Path
import numpy as np
import sentencepiece as sp
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from train import LSTMAWD
from data import RapppidDataset2
from common import ORIGINAL_SHA, read, sha

UPSTREAM = Path('/opt/rapppid/upstream')
RELEASE = '1690837077.519848_red-dreamy'
ASSETS = UPSTREAM / 'data/pretrained_weights' / RELEASE
CHECKPOINT = ASSETS / f'{RELEASE}.ckpt'
TOKENIZER_SHA = 'b60afba79a5f2e9e561f616cd1c18212b1e8d8bc57f0d6c4d5de9715c2496ba8'
MAX_RESIDUES = 1500


def learned_digest(model):
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        digest.update(name.encode() + b'\0')
        digest.update(str(value.dtype).encode() + str(tuple(value.shape)).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def load_model(device):
    document = read('/opt/rapppid/downloads.json')
    assert document['upstream_commit'] == 'c3a28be56fb3bd96ccf8cc44ea9145cfcadeaf6c'
    for item in document['files']:
        path = UPSTREAM / item['path']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256']
    assert sha(CHECKPOINT) == ORIGINAL_SHA
    # The original Lightning checkpoint names its callback class as a dictionary
    # key. Allow precisely that class, without general unrestricted unpickling.
    with torch.serialization.safe_globals([ModelCheckpoint]):
        checkpoint = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
    model = LSTMAWD(**checkpoint['hyper_parameters'])
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    assert all(torch.equal(value, checkpoint['state_dict'][name]) for name, value in model.state_dict().items())
    assert model.class_head_name == 'mult' and model.trunc_len == MAX_RESIDUES
    assert model.bi_reduce == 'last' and model.embedding_size == 64
    assert all(not getattr(module, 'variational', False) for module in model.modules())
    model.eval().requires_grad_(False).to(device)
    return model


def tokenizer():
    path = ASSETS / 'spm.model'
    assert sha(path) == TOKENIZER_SHA
    result = sp.SentencePieceProcessor(model_file=str(path))
    assert result.get_piece_size() == 250
    return result


def encode(spp, sequence):
    # Native validation/test tokenizer: deterministic SentencePiece, leading
    # 1500 residues retained, zero padding. No random replacement of residues.
    values = np.asarray(RapppidDataset2.static_encode(MAX_RESIDUES, spp, sequence,
                                                    sp=True, pad=True, sampling=False), dtype=np.int64)
    assert values.shape == (MAX_RESIDUES,) and ((values >= 0) & (values < 250)).all()
    assert np.count_nonzero(values) > 0, 'Native encoder cannot process an empty effective sequence'
    return values


@torch.inference_mode()
def singleton_embedding(model, tokens, device):
    return model(torch.as_tensor(tokens[None, :], device=device)).reshape(1, 64)


@torch.inference_mode()
def native_probability(model, a_tokens, b_tokens, device):
    # The authors' test_step, evaluated for one pair: separate endpoint
    # encodings, then one unmodified native head call. No cross-pair moments.
    a = singleton_embedding(model, a_tokens, device)
    b = singleton_embedding(model, b_tokens, device)
    return torch.sigmoid(model.class_head(a, b).float()).reshape(-1)


@torch.inference_mode()
def independent_head(model, a, b):
    # Algebraic batching of the same singleton MultClassHead. Upstream .mean()
    # and .std() span the entire input tensor; direct multi-pair calls would
    # change each score. These row reductions preserve the one-pair definition.
    a = (a - a.mean(dim=1, keepdim=True)) / a.std(dim=1, keepdim=True, correction=1)
    b = (b - b.mean(dim=1, keepdim=True)) / b.std(dim=1, keepdim=True, correction=1)
    return torch.sigmoid(model.class_head.fc(model.class_head.nl(a * b)).float()).reshape(-1)
