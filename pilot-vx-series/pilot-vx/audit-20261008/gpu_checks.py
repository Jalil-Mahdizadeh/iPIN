"""Post-pilot, label-free diagnostics; does not alter pilot artifacts or fit a model."""
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'scripts'))
from common import ROOT, atomic, config, now, sha, text_hash, verify_freeze
from encoder import Encoder
from msa import LOOKUP, load_monomer, pair, shuffled
from MSA_Pairformer.dataset import aa2tok_d, prepare_msa_masks


@torch.inference_mode()
def forward(enc, tokens, boundary, autocast=True):
    t = torch.as_tensor(tokens.astype(np.int64), device='cuda').unsqueeze(0)
    masks = prepare_msa_masks(t, device=enc.device)
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=autocast):
        out = enc.model(
            msa=torch.nn.functional.one_hot(t, 28).float(),
            mask=masks[0], msa_mask=masks[1], full_mask=masks[2], pairwise_mask=masks[3],
            complex_chain_break_indices=[[boundary]], return_repr_after_layer_idx=15,
            return_cb_contacts=False, return_confind_contacts=False, return_seq_weights=True,
            store_pairwise_repr_cpu=False, store_msa_repr_cpu=False,
        )
        z = out['final_pairwise_repr'][0]
        cross = (z[:boundary, boundary:] + z[boundary:, :boundary].transpose(0, 1)) / 2
        good = t[0, 0] != 26
        projected = (cross @ enc.projection)[good[:boundary]][:, good[boundary:]].reshape(-1, 32).float()
    pooled = torch.cat([projected.mean(0), projected.std(0, unbiased=False),
                        projected.amax(0), torch.quantile(projected, .95, dim=0)])
    cross = cross[good[:boundary]][:, good[boundary:]].float().cpu().numpy()
    weights = {k: v.float().cpu().numpy().ravel() for k, v in out['seq_weights_list_d'].items()}
    return cross, pooled.cpu().numpy(), weights


def rel(a, b):
    return float(np.linalg.norm(a-b) / max(np.linalg.norm(a), 1e-30))


def main():
    started = time.monotonic()
    fingerprint = verify_freeze()
    assert all(aa2tok_d[k] == v for k, v in LOOKUP.items())
    enc = Encoder()
    samples = {r['uid']: r for r in json.loads((ROOT/'data/sample.json').read_text())}
    catalog = json.loads((ROOT/'data/monomer-catalog.json').read_text())
    cases = json.loads((ROOT/'qualification/real-encoder.json').read_text())['cases']
    report = {'at_utc': now(), 'fingerprint': fingerprint, 'script_sha256': sha(__file__),
              'selection': 'same three label-free cases chosen before pilot outcomes',
              'labels_used': False, 'heads_fitted': False, 'structural_heads_evaluated': False,
              'token_mapping_matches_upstream': True, 'cases': []}
    for case in cases:
        r = samples[case['uid']]; a, b = sorted([r['a'], r['b']])
        ma = load_monomer(ROOT/catalog[str(a)]['path']); mb = load_monomer(ROOT/catalog[str(b)]['path'])
        tokens, meta = pair(ma, mb); boundary = len(ma['query'])
        null, nm = shuffled(tokens, boundary, meta['taxonomy'], int(text_hash(f"{config()['seed']}:{r['uid']}")[:16], 16))
        good_b = tokens[0, boundary:] != 26
        changed = (tokens[1:, boundary:][:, good_b] != null[1:, boundary:][:, good_b])
        z, f, w = forward(enc, tokens, boundary)
        zn, fn, wn = forward(enc, null, boundary)
        direct = enc.one(tokens, boundary)
        reverse = np.concatenate([tokens[:, boundary:], tokens[:, :boundary]], axis=1)
        symmetric = (direct + enc.one(reverse, tokens.shape[1]-boundary))/2
        saved = json.loads((ROOT/f'features/{r["uid"]}.json').read_text())
        primary_gap = float(np.max(np.abs(symmetric - np.array(saved['true']))))
        comparison = float(np.max(np.abs(f - direct)))
        assert comparison == 0, comparison
        assert primary_gap == 0, primary_gap
        perm = np.r_[0, np.random.default_rng(45).permutation(np.arange(1, len(tokens)))]
        zp, fp, _ = forward(enc, tokens[perm], boundary)
        zq, fq, _ = forward(enc, tokens[:1], boundary)
        permutation = np.array(nm['permutation']); moved = permutation != np.arange(len(tokens))
        result = {'uid': r['uid'], 'length': tokens.shape[1], 'depth': len(tokens),
                  'upstream_forward_vs_pilot_max_feature_error': comparison,
                  'recomputed_symmetric_vs_saved_max_feature_error': primary_gap,
                  'null_row_index_changed_fraction': nm['changed_fraction'],
                  'null_actual_sequence_changed_fraction': float(changed.any(1).mean()),
                  'null_amino_acid_cell_changed_fraction': float(changed.mean()),
                  'null_raw_cross_repr_relative_change': rel(z, zn),
                  'null_pooled_features_relative_change': rel(f, fn),
                  'joint_row_permutation_raw_relative_change': rel(z, zp),
                  'joint_row_permutation_pooled_relative_change': rel(f, fp),
                  'query_only_raw_relative_change': rel(z, zq),
                  'query_only_pooled_relative_change': rel(f, fq),
                  'attention_by_layer': {k: {'query_weight': float(v[0]),
                      'weight_on_shuffled_rows': float(v[moved].sum()),
                      'effective_rows_inverse_sum_squared': float(1/np.square(v).sum())} for k, v in w.items()}}
        if len(report['cases']) == 0:
            z32, f32, _ = forward(enc, tokens, boundary, autocast=False)
            result['fp32_vs_bf16_raw_relative_change'] = rel(z32, z)
            result['fp32_vs_bf16_pooled_relative_change'] = rel(f32, f)
        report['cases'].append(result)
        report['seconds'] = time.monotonic()-started
        atomic(HERE/'gpu.json', report)
        print(json.dumps({k:v for k,v in result.items() if k != 'attention_by_layer'}), flush=True)
        del z, zn, zp, zq
    report['complete'] = True
    report['seconds'] = time.monotonic()-started
    report['additional_gpu_hours_upper_bound'] = report['seconds']/3600
    atomic(HERE/'gpu.json', report)
    print(json.dumps({'complete': True, 'seconds': report['seconds']}), flush=True)


if __name__ == '__main__':
    main()
