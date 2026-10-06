"""Bounded scientific checks on real train/validation inputs and tiny ESM models."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from transformers import AutoTokenizer
from data import PairData
from model import PairModel, expand_chain_qk
from state import atomic_json, sha256

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/prepared'
torch.set_num_threads(8)
torch.manual_seed(17)
torch.backends.cuda.matmul.allow_tf32 = False
torch.use_deterministic_algorithms(True)
cfg = json.loads((ROOT / 'configs/reference-official-seed2.json').read_text())
results = {}
manifest = json.loads((DATA / 'manifest.json').read_text())
for name, expected in manifest['files'].items(): assert sha256(DATA / name) == expected
train, val = PairData(DATA, 'official', 'train'), PairData(DATA, 'official', 'val')
active = np.unique(np.concatenate((train.rows[:, :2].ravel(), val.rows[:, :2].ravel())))
assert np.array_equal(np.flatnonzero(np.diff(train.offsets)), active)
assert not set(train.rows[:, :2].ravel()) & set(val.rows[:, :2].ravel())
assert len(train) == 163085 and len(val) == 59258
assert len(PairData(DATA, 'official', 'train', 2193)) == 130461
assert not list(DATA.rglob('*test*'))
for fold in range(3):
    a, b = PairData(DATA, f'fold-{fold}', 'train'), PairData(DATA, f'fold-{fold}', 'val')
    c = np.load(DATA / f'fold-{fold}/crossing-unused.npy')
    reconstructed = np.concatenate((a.rows, b.rows, c))
    assert np.array_equal(reconstructed[np.argsort(reconstructed[:, 3])], train.rows[np.argsort(train.rows[:, 3])])
    assert not set(a.rows[:, :2].ravel()) & set(b.rows[:, :2].ravel())
groups = json.loads((DATA / 'fold-groups.json').read_text())
assignment = {p: f for f, ps in enumerate(groups['fold_proteins']) for p in ps}
for line in (ROOT / 'data/development-homology.tsv').read_text().splitlines():
    a, b, identity, qc, tc, _ = line.split('\t')
    if float(identity) >= .4 and min(float(qc), float(tc)) >= .8: assert assignment[int(a)] == assignment[int(b)]
results['data'] = {'train_rows': len(train), 'validation_rows': len(val), 'capped_train_rows': 130461,
    'active_sequence_count': len(active), 'no_unused_test_tokens': True, 'fold_edges_disjoint': True,
    'train_validation_disjoint': True, 'folds_reconstruct_parent_exactly': True}

# Check the entire first sampling cycle, its residual crossing, and reconstruction.
cursor = {'cycle': 0, 'offset': 0}; consumed = []; cycles = []
for _ in range((len(train) + 63) // 64):
    ids, cs, cursor = train.next_batch(cursor, 2, 64)
    assert len(ids) == len(cs) == 64
    consumed.extend(ids.tolist()); cycles.extend(cs.tolist())
assert np.array_equal(np.sort(np.array(consumed)[np.array(cycles) == 0]), np.arange(len(train)))
assert cursor['cycle'] * len(train) + cursor['offset'] == len(consumed)
fresh = PairData(DATA, 'official', 'train')
for left, right in zip(train.next_batch(cursor, 2, 64)[:2], fresh.next_batch(cursor, 2, 64)[:2]):
    assert np.array_equal(left, right)
results['sampler'] = {'fixed_global_exposure': True, 'every_row_once_per_cycle': True,
    'cycle_crossing_no_drop': True, 'cursor_reconstructs_next_batch': True}

ids = np.argsort(train.lengths)[:3]
batch = train.batch(ids, np.array([0, 1, 0]), 2, True, 'cuda')
repeat = train.batch(ids, np.array([0, 1, 0]), 2, True, 'cuda')
for key in batch: assert torch.equal(batch[key], repeat[key])
tokenizer = AutoTokenizer.from_pretrained(ROOT / 'assets/esm2', local_files_only=True)
vocab = (ROOT / 'assets/esm2/vocab.txt').read_text().splitlines()
for j, row in enumerate(ids):
    a, b = [train.sequence(p) for p in train.rows[row, :2]]
    # Native public format is CLS A EOS B EOS, including the middle separator.
    expected = tokenizer(''.join(vocab[t] for t in a) + '<eos>' + ''.join(vocab[t] for t in b))['input_ids']
    assert batch['clean_ids'][2*j, :len(expected)].tolist() == expected
    for key in ['masked_ids', 'mlm_labels']:
        assert torch.equal(batch[key][2*j, 1:1+len(a)], batch[key][2*j+1, 2+len(b):2+len(b)+len(a)])
        assert torch.equal(batch[key][2*j, 2+len(a):2+len(a)+len(b)], batch[key][2*j+1, 1:1+len(b)])
assert torch.all(batch['mlm_labels'][~batch['residue_mask']] == -100)
assert torch.all(batch['chain_ids'][batch['attention_mask'] == 0] == -1)
results['tokenization'] = {'hf_agreement': True, 'stateless_corruption': True,
    'mask_coupled_across_orientations': True, 'special_tokens_excluded_from_mlm_and_pooling': True}

# Algebraic oracle at the real 64-dimension attention head size, including padding.
leaves = [torch.randn(2, 3, 23, 64, device='cuda', requires_grad=True) for _ in range(5)]
q, k, qr, kr, v = leaves
chains = torch.tensor([[0]*8+[1]*12+[-1]*3, [0]*12+[1]*11], device='cuda')
same = chains[:, None, :, None].eq(chains[:, None, None, :])
valid = chains[:, None, None, :].ne(-1)
dense_scores = torch.where(same, qr @ kr.transpose(-1, -2), q @ k.transpose(-1, -2))
dense = dense_scores.masked_fill(~valid, -torch.inf).softmax(-1) @ v
xq, xk = expand_chain_qk(q, k, qr, kr, chains)
with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
    efficient = F.scaled_dot_product_attention(xq, xk, v, attn_mask=valid, scale=1.)
active_queries = chains.ne(-1)[:, None, :, None]
dense = dense * active_queries; efficient = efficient * active_queries
torch.testing.assert_close(efficient, dense, atol=2e-5, rtol=2e-5)
weights = torch.randn_like(dense)
g1 = torch.autograd.grad((dense * weights).sum(), leaves, retain_graph=True)
g2 = torch.autograd.grad((efficient * weights).sum(), leaves)
for a, b in zip(g1, g2): torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)
results['chain_attention'] = {'head_dimension': 64, 'expanded_query_key_dimension': 256,
    'dense_oracle_forward_max_abs': float((efficient-dense).abs().max()),
    'dense_oracle_gradient_max_abs': max(float((a-b).abs().max()) for a,b in zip(g1,g2)),
    'one_joint_softmax': True, 'padding_qualified': True}

# Numerically verify all four objective combinations and positive weight 10.
objective_results = []
for corruption, mlm_weight, positive_weight in [(True,1.,1.), (True,0.,1.), (False,0.,1.), (False,1.,1.), (True,1.,10.)]:
    config = {**cfg, 'classification_corruption': corruption, 'mlm_weight': mlm_weight, 'positive_weight': positive_weight}
    torch.manual_seed(3); m = PairModel(None, config, tiny=True).cuda().train()
    loss, cls, mlm, logits = m(**batch)
    with torch.no_grad():
        h = m.encode(batch['masked_ids'] if corruption else batch['clean_ids'], batch['attention_mask'], batch['chain_ids'])
        z = m.classify(h, batch['chain_ids'], batch['residue_mask'])
        expected_cls = F.binary_cross_entropy_with_logits(z, batch['labels'][:,None].expand_as(z),
                          pos_weight=z.new_tensor(positive_weight), reduction='none').mean(1).sum()
        expected_mlm = z.new_zeros(())
        if mlm_weight:
            hm = h if corruption else m.encode(batch['masked_ids'], batch['attention_mask'], batch['chain_ids'])
            all_logits = m.esm_mask.lm_head(hm)
            each = F.cross_entropy(all_logits.transpose(1,2), batch['mlm_labels'], ignore_index=-100, reduction='none')
            expected_mlm = (each.sum(1) / batch['mlm_labels'].ne(-100).sum(1)).view(-1,2).mean(1).sum()
        torch.testing.assert_close(cls, expected_cls)
        torch.testing.assert_close(mlm, expected_mlm)
        torch.testing.assert_close(loss, 10 * expected_cls + mlm_weight * expected_mlm)
    loss.backward()
    missing = [n for n,p in m.named_parameters() if p.requires_grad and p.grad is None]
    assert not missing, missing
    assert m.esm_mask.esm.embeddings.word_embeddings.weight.grad is not None
    m.eval()
    with torch.no_grad(): torch.testing.assert_close(m(**batch, compute_loss=False), m.classify(
        m.encode(batch['clean_ids'], batch['attention_mask'], batch['chain_ids']), batch['chain_ids'], batch['residue_mask']))
    objective_results.append({'classification_corruption': corruption, 'mlm_weight': mlm_weight,
                             'positive_weight': positive_weight, 'all_trainable_parameters_used': True})
    del m
results['objectives'] = objective_results

# Explicit reference compatibility with the qualified v1 model on identical inputs.
spec = importlib.util.spec_from_file_location('archived_v1_model', ROOT.parent / 'retrain-v1/scripts/model.py')
legacy_module = importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy_module)
torch.manual_seed(2); legacy = legacy_module.PairModel(None, mode='reference', tiny=True).cuda()
torch.manual_seed(2); current = PairModel(None, cfg, tiny=True).cuda()
current.load_state_dict(legacy.state_dict(), strict=True)
left = legacy(batch['masked_ids'], batch['attention_mask'], batch['labels'], batch['mlm_labels'])
right = current(**batch)
for a,b in zip(left, right): torch.testing.assert_close(a,b,rtol=0,atol=0)
left[0].backward(); right[0].backward()
for (an, ap), (bn, bp) in zip(legacy.named_parameters(), current.named_parameters()):
    assert an == bn
    if ap.grad is not None: torch.testing.assert_close(ap.grad,bp.grad,atol=0,rtol=0)
results['reference_compatibility'] = {'v1_forward_loss_gradient_bitwise_identical': True,
    'scope': 'Identical tiny-model parameters and inputs, FP32; sampler/selection changes documented separately.'}
del legacy, current

heads = {}; initial_logits = []
for readout in ['cls_linear', 'cls_mlp', 'residue_mean']:
    torch.manual_seed(2); m = PairModel(None, {**cfg, 'readout': readout}, tiny=True).cuda().eval()
    heads[readout] = sum(p.numel() for p in m.parameters() if p.requires_grad)
    with torch.no_grad(): initial_logits.append(m(**batch, compute_loss=False))
for z in initial_logits[1:]: torch.testing.assert_close(z, initial_logits[0], atol=0, rtol=0)
assert heads['cls_mlp'] == heads['residue_mean'] > heads['cls_linear']
results['readout'] = {'trainable_parameters': heads, 'capacity_matched': True, 'zero_residual_preserves_initial_logits': True}
del m

# Two live forward graphs with different chain boundaries, then backward: a mutable
# module attribute would use the wrong chain metadata during recomputation here.
config = {**cfg, 'classification_corruption': False, 'attention_mode': 'chain_aware', 'readout': 'residue_mean'}
torch.manual_seed(7); checkpointed = PairModel(None, config, tiny=True).cuda().train()
plain = copy.deepcopy(checkpointed); plain.esm_mask.gradient_checkpointing_disable()
other = train.batch(np.argsort(train.lengths)[10:12], np.zeros(2,dtype=int), 2, True, 'cuda')
for m in [checkpointed, plain]:
    first = m(**batch)[0]; second = m(**other)[0]
    (first + second).backward()
for (a,p), (b,q) in zip(checkpointed.named_parameters(), plain.named_parameters()):
    assert a == b
    if p.requires_grad:
        assert p.grad is not None and q.grad is not None, a
        torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-5)
results['checkpoint_chain_context'] = {'two_live_graphs_and_decoupled_mlm': True, 'gradients_match_no_checkpoint': True}
results.update(passed=True, data_manifest_sha256=sha256(DATA / 'manifest.json'),
    code={n: sha256(ROOT/'scripts'/n) for n in ['data.py','model.py','train.py','state.py','release.py']},
    device=torch.cuda.get_device_name(), torch=torch.__version__)
atomic_json(ROOT/'qualification/science.json', results)
print(json.dumps(results, indent=2))
