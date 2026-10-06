"""Independent official ESMC forward using the pair format actually trained.

The first diagnostic incorrectly assumed the tokenizer's two-string API adds
CLS/EOS. Its failed log is preserved. ESMC's two-string API concatenates bare
residues; both v5 training and inference explicitly use CLS,A,EOS,B,EOS.
Here each raw sequence is independently encoded by the official tokenizer,
then composed with those documented special tokens. Production is untouched.
"""
import types
import numpy as np
import torch
from scipy.special import expit
from esm.layers.attention import MultiHeadAttention
from esm.tokenization import EsmSequenceTokenizer
from bench_utils import ROOT, atomic, cuda, load_npz, now, read, record
import pair_infer as pair


@torch.inference_mode()
def main():
    name = 'ipin-esmc'
    device = cuda()
    meta = read(ROOT/'data/sequences.json')
    rows = np.load(ROOT/'data/union.npy')
    mapping = load_npz(ROOT/'data/pair-mapping.npz')
    data = pair.test_data()
    selected, selection = [], []
    for test in ['mouse', 'fly', 'worm', 'yeast', 'ecoli']:
        source = np.load(ROOT/'data'/f'{test}.npy')
        choices = sorted({0, 1, len(source)//2, int(np.argmax(data.lengths[mapping[test]]))})
        for i in choices:
            uid = int(mapping[test][i])
            selected.append(uid)
            selection.append({'species': test, 'source_row': i, 'union_id': uid})
    ids = np.array(sorted(set(selected)), dtype=np.int64)
    saved = load_npz(ROOT/'results'/f'{name}-union.npz')['logits'][ids]
    model = pair.load_model(name, device)
    fast = np.concatenate([pair.predict(model, name, data, np.array([uid]), device, fp32=True) for uid in ids])
    for block in model.esmc.transformer.blocks:
        block.attn.forward = types.MethodType(MultiHeadAttention.forward, block.attn)
    tokenizer = EsmSequenceTokenizer()
    cases = []
    for k, uid in enumerate(ids):
        a, b = map(int, rows[uid, :2])
        features = data.batch(np.array([uid]), np.array([0]), 2, False, device)
        gold = []
        for col, (x, y) in enumerate([(a, b), (b, a)]):
            aa = tokenizer.encode(meta['sequence'][x], add_special_tokens=False)
            bb = tokenizer.encode(meta['sequence'][y], add_special_tokens=False)
            tokens = torch.tensor([[tokenizer.cls_token_id] + aa + [tokenizer.eos_token_id] + bb + [tokenizer.eos_token_id]], device=device)
            assert tokens.shape[1] == data.lengths[uid]
            assert torch.equal(tokens[0], features['clean_ids'][col]), (int(uid), col, 'tokenization')
            hidden = model.esmc(tokens, sequence_id=torch.ones_like(tokens, dtype=torch.bool)).embeddings
            gold.append(float(model.classifier(torch.relu(hidden[:, 0]))[0, 0]))
        gold = np.array(gold)
        le = float(abs(fast[k]-gold).max())
        pe = float(abs(expit(fast[k])-expit(gold)).max())
        assert le < .002 and pe < .0002, (int(uid), le, pe)
        cases.append({'union_id': int(uid), 'tokens': int(data.lengths[uid]),
                      'saved_bf16_logits': saved[k].tolist(), 'fresh_fp32_optimized_logits': fast[k].tolist(),
                      'official_fp32_logits': gold.tolist(), 'optimized_official_max_logit_error': le,
                      'optimized_official_max_probability_error': pe,
                      'saved_bf16_official_max_probability_error': float(abs(expit(saved[k])-expit(gold)).max())})
        print(cases[-1], flush=True)
    atomic(ROOT/'qualification'/f'{name}-independent-nonhuman.json', {
        'at_utc': now(), 'passed': True, 'model': name,
        'independent_raw_string_pair_tokenization_matches': True,
        'pair_format': 'CLS A EOS B EOS, as explicitly used in v5 training and production inference',
        'official_backbone_fp32_forward_matches': True, 'source_selection': selection, 'cases': cases,
        'checkpoint': read(ROOT/'provenance/selection.json')['models'][name]['checkpoint'],
        'script': record(__file__), 'test_scores_not_used_for_fixture_selection': True,
        'production_weights_or_workers_modified': False,
        'earlier_diagnostic_failure': 'audit-native-forward-esmc.log: unsupported assumption about special-token insertion by official two-string tokenizer API; not a production path'})
    print('Independent ESMC forward audit passed', flush=True)


if __name__ == '__main__':
    main()
