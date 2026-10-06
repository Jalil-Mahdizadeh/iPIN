"""Finite mathematical, SDK-equivalence and unchanged-data checks for v4."""
import copy
import importlib.util
import json
import types
from pathlib import Path
import numpy as np
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from esm.layers.attention import MultiHeadAttention
from esm.tokenization import EsmSequenceTokenizer
from contracts import CORE, RUNS, core_hashes, production_config, verify_inputs
from data import PairData
from model import PairModel
from model_esm2 import CHAIN_CONTEXT, chain_context, expand_chain_qk
from model_esmc import esmc_attention_forward
from state import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]


def assert_close(a, b, atol=3e-6, rtol=3e-5):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def dense_attention(attn, x, valid, chains, mode):
    q, k, v = attn.layernorm_qkv(x).chunk(3, -1)
    q, k = attn.q_ln(q).to(q.dtype), attn.k_ln(k).to(k.dtype)
    qr, kr = attn._apply_rotary(q, k)
    reshape = lambda t: t.unflatten(-1, (attn.n_heads, attn.d_head)).transpose(1, 2)
    q, k, v, qr, kr = map(reshape, (q, k, v, qr, kr))
    within = qr @ kr.transpose(-1, -2)
    if mode == 'chain_aware':
        across = q @ k.transpose(-1, -2)
        same = (chains[:, :, None] == chains[:, None, :])[:, None]
        scores = torch.where(same, within, across)
        eq, ek = expand_chain_qk(q, k, qr, kr, chains)
        assert_close(eq @ ek.transpose(-1, -2), scores, atol=2e-5)
    else:
        scores = within
    probabilities = (scores * attn.d_head ** -.5).masked_fill(~valid[:, None, None, :], -torch.inf).softmax(-1)
    return attn.out_proj((probabilities @ v).transpose(1, 2).contiguous().flatten(-2))


def attention_checks():
    details = []
    valid = torch.tensor([[1] * 11, [1] * 9 + [0] * 2], device='cuda', dtype=torch.bool)
    # Padding receives a valid dummy chain here, so the dense all-query reference
    # can also compare padded-query outputs. Production metadata uses -1 there.
    chains = torch.tensor([[0] * 5 + [1] * 6, [0] * 4 + [1] * 7], device='cuda')
    for mode in ['standard', 'chain_aware']:
        layer = MultiHeadAttention(64, 4).cuda()
        layer.pair_attention, layer.pair_backend = mode, 'math'
        layer.forward = types.MethodType(esmc_attention_forward, layer)
        x = torch.randn(2, 11, 64, device='cuda', requires_grad=True)
        with chain_context(chains):
            actual = layer(x, valid)
        reference = dense_attention(layer, x, valid, chains, mode)
        assert_close(actual, reference)
        probe = torch.randn_like(actual)
        parameters = (x, *layer.parameters())
        ga = torch.autograd.grad((actual * probe).sum(), parameters, retain_graph=True)
        gb = torch.autograd.grad((reference * probe).sum(), parameters, retain_graph=True)
        for a, b in zip(ga, gb):
            assert_close(a, b, atol=1e-5, rtol=2e-4)
        cross = torch.autograd.grad(actual[0, :5].square().sum(), x)[0][0, 5:]
        assert cross.norm() > 0, 'Cross-chain information flow was blocked'
        with torch.no_grad(), chain_context(chains):
            perturbed = x.detach().clone(); perturbed[1, 9:] += 1000
            altered = layer(perturbed, valid)
        assert_close(actual[valid], altered[valid])
        details.append({'mode': mode, 'forward_max_abs_error': float((actual-reference).abs().max()),
                        'cross_chain_gradient_norm': float(cross.norm()),
                        'forward_and_all_parameter_gradients_match_dense_reference': True,
                        'padding_cannot_influence_valid_queries': True})
    return details


def tiny_adapter_checks(cfg, batch):
    cfg = {**cfg, 'attention_backend': 'math'}
    model = PairModel(None, cfg, tiny=True).cuda().eval()
    reference = copy.deepcopy(model.esmc)
    for block in reference.transformer.blocks:
        block.attn.forward = types.MethodType(MultiHeadAttention.forward, block.attn)
    with sdpa_kernel(SDPBackend.MATH):
        adapted = model.encode(batch['clean_ids'], batch['attention_mask'], batch['chain_ids'])
        original = reference(batch['clean_ids'], sequence_id=batch['attention_mask'].bool()).embeddings
    valid = batch['attention_mask'].bool()
    assert_close(adapted[valid], original[valid], atol=6e-6, rtol=6e-5)
    probe = torch.randn_like(adapted)
    (adapted[valid] * probe[valid]).sum().backward()
    (original[valid] * probe[valid]).sum().backward()
    for (name, p), (ref_name, q) in zip(model.esmc.named_parameters(), reference.named_parameters()):
        assert name == ref_name
        if p.grad is None or q.grad is None:
            assert p.grad is None and q.grad is None
        else:
            assert_close(p.grad, q.grad, atol=1e-4, rtol=5e-4)
    return {'original_sdk_valid_embeddings_and_encoder_gradients_match': True,
            'max_valid_embedding_error': float((adapted[valid]-original[valid]).abs().max())}


def recomputation_checks(cfg, batch):
    cfg = {**cfg, 'attention_mode': 'chain_aware', 'attention_backend': 'math'}
    model = PairModel(None, cfg, tiny=True).cuda().train()
    direct = copy.deepcopy(model); direct.gradient_checkpointing = False
    chains = batch['chain_ids']
    other_chains = chains.clone()
    other_chains[:, 2:5] = 1 - other_chains[:, 2:5]
    probe = torch.randn(len(chains), chains.shape[1], 64, device='cuda')
    values = []
    for current in [model, direct]:
        a = current.encode(batch['clean_ids'], batch['attention_mask'], chains)
        b = current.encode(batch['clean_ids'], batch['attention_mask'], other_chains)
        assert CHAIN_CONTEXT.get() is None
        # Both forward graphs exist before backward; late/global metadata is wrong.
        loss = ((a + .3 * b) * probe * batch['attention_mask'][..., None]).sum()
        loss.backward(); values.append(loss.detach())
    assert_close(*values)
    for (n, p), (m, q) in zip(model.named_parameters(), direct.named_parameters()):
        assert n == m
        if p.grad is None or q.grad is None:
            assert p.grad is None and q.grad is None
        else:
            assert_close(p.grad, q.grad, atol=1e-4, rtol=3e-4)
    return {'two_live_forward_graphs_with_different_chain_metadata': True,
            'checkpointed_and_direct_encoder_gradients_match': True}


def full_sdk_check(cfg, batch):
    torch.manual_seed(2)
    model = PairModel(ROOT / 'assets/esmc', cfg).cuda().eval()
    original_forwards = [block.attn.forward for block in model.esmc.transformer.blocks]
    with torch.inference_mode(), sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        adapted = model.encode(batch['clean_ids'], batch['attention_mask'], batch['chain_ids'])
        for block in model.esmc.transformer.blocks:
            block.attn.forward = types.MethodType(MultiHeadAttention.forward, block.attn)
        original = model.esmc(batch['clean_ids'], sequence_id=batch['attention_mask'].bool()).embeddings
        valid = batch['attention_mask'].bool()
        error = float((adapted[valid]-original[valid]).abs().max())
        assert_close(adapted[valid], original[valid], atol=5e-5, rtol=2e-4)
        for block, method in zip(model.esmc.transformer.blocks, original_forwards):
            block.attn.forward = method
        # Eval-created RoPE caches must also be safe on the following training step.
    model.train()
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss, *_ = model(**batch)
    loss.backward()
    assert torch.isfinite(loss) and all(p.grad is not None and torch.isfinite(p.grad).all()
        for p in model.parameters() if p.requires_grad)
    head_hash = sha_head(model)
    return {'full_pretrained_standard_adapter_max_valid_embedding_error': error,
            'full_pretrained_sdk_equivalence': True, 'inference_to_training_cache_transition': True,
            'fp32_parameters': all(p.dtype == torch.float32 for p in model.parameters()),
            'standard_initial_head_hash': head_hash}


def sha_head(model):
    import hashlib
    h = hashlib.sha256()
    for name, p in model.named_parameters():
        if name.startswith(('classifier.', 'readout_')):
            h.update(name.encode()); h.update(p.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def main():
    torch.set_num_threads(8); torch.manual_seed(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    verify_inputs(ROOT, images=False)
    data_root = ROOT / 'data/prepared'
    train, val = [PairData(data_root, 'official', s) for s in ['train', 'val']]
    assert (len(train), len(val)) == (163085, 59258)
    assert (train.rows[:, 2].sum(), val.rows[:, 2].sum()) == (81550, 29628)
    assert (train.lengths.max(), val.lengths.max()) == (16322, 39391)
    assert not set(train.rows[:, :2].ravel()) & set(val.rows[:, :2].ravel())
    for split, data in [('train', train), ('val', val)]:
        assert np.array_equal(data.rows, np.load(ROOT.parent / 'retrain-v3/data/prepared/official' / f'{split}.npy'))
    assert np.array_equal(np.sort(train.plan(0, 2, 64)), np.arange(len(train)))
    assert not list((ROOT / 'data').rglob('*test*'))
    try: PairData(data_root, 'official', 'test')
    except AssertionError: pass
    else: raise AssertionError('Test loader was accepted')
    old = ROOT.parent / 'retrain-v3/releases/20261001-final-training/code'
    assert sha256(ROOT / 'scripts/data.py') == sha256(old / 'data.py')
    assert sha256(ROOT / 'scripts/state.py') == sha256(old / 'state.py')
    assert sha256(ROOT / 'scripts/model_esm2.py') == sha256(old / 'model.py')
    assert (ROOT / 'scripts/train.py').read_text().split('    groups = ', 1)[1] == (old / 'train.py').read_text().split('    groups = ', 1)[1]
    configurations = [json.loads((ROOT / 'configs' / (run + '.json')).read_text()) for run in RUNS]
    for cfg in configurations: production_config(cfg)
    fixed = set(configurations[0]) - {'backbone', 'initialization', 'attention_mode', 'name', 'arm'}
    assert all({k: cfg[k] for k in fixed} == {k: configurations[0][k] for k in fixed} for cfg in configurations)
    vocab = (ROOT / 'assets/esm2/vocab.txt').read_text().splitlines()
    tokenizer = EsmSequenceTokenizer()
    symbols = {vocab[int(token)]: int(token) for token in np.unique(train.tokens)}
    for symbol, token in symbols.items():
        assert len(symbol) == 1
        assert tokenizer.encode(symbol, add_special_tokens=False) == [token], symbol
    assert (tokenizer.cls_token_id, tokenizer.pad_token_id, tokenizer.eos_token_id) == (0, 1, 2)
    short = np.flatnonzero((train.lengths >= 30) & (train.lengths <= 128))[:2]
    batch = train.batch(short, np.zeros(2, dtype=int), 2, False, 'cuda')
    for i, idx in enumerate(short):
        a, b = [train.sequence(p).tolist() for p in train.rows[idx, :2]]
        for j, (x, y) in enumerate([(a, b), (b, a)]):
            row = 2 * i + j; n = len(x) + len(y) + 3
            assert batch['clean_ids'][row, :n].tolist() == [0] + x + [2] + y + [2]
            assert int(batch['residue_mask'][row].sum()) == len(x) + len(y)
    cfg = configurations[1]
    result = {'data_unchanged': True, 'train_rows': len(train), 'val_rows': len(val),
              'test_data_used': False, 'token_ids_verified_by_symbol': symbols,
              'esm2_model_data_checkpoint_code_unchanged': True,
              'optimization_and_validation_loop_unchanged': True,
              'attention': attention_checks(), 'tiny_adapter': tiny_adapter_checks(cfg, batch),
              'checkpoint_context': recomputation_checks(cfg, batch)}
    result['full_sdk'] = full_sdk_check(cfg, batch)
    torch.cuda.empty_cache()
    torch.manual_seed(2)
    chain = PairModel(ROOT / 'assets/esmc', configurations[2])
    assert sha_head(chain) == result['full_sdk']['standard_initial_head_hash']
    result['esmc_initial_heads_identical_between_attention_arms'] = True
    result.update(passed=True, code=core_hashes(ROOT / 'scripts'),
                  source_sha256={'scripts/qualify_model.py': sha256(Path(__file__))},
                  data_manifest_sha256=sha256(data_root / 'manifest.json'))
    atomic_json(ROOT / 'qualification/model-and-data.json', result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__': main()
