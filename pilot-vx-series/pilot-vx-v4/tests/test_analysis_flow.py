"""Synthetic complete analysis/audit with distinct Q/T/S/quality references."""
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from contextlib import ExitStack, redirect_stdout
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import study
import inputs
import analyze
import verify
from heads import evidence
from diagnostics import describe
from statistics_v4 import bootstrap, CONTRASTS


def fixture(root, cfg):
    n = 640; rng = np.random.default_rng(92); y = np.arange(n) % 2
    base = rng.normal(size=n) + .3 * y; gate = rng.uniform(.5, 1., size=n); gate[::9] = 0
    x = rng.normal(size=(n, 128)); x[:, :4] += y[:, None] * .6; x[gate == 0] = 0
    rows = []; pairs = []; audits = {}; monomers = {}
    for name in ('data', 'results', 'features', 'scripts'): (root / name).mkdir()
    (root / 'scripts/verify.py').write_bytes(Path(verify.__file__).read_bytes())
    study.atomic(root / 'config.json', cfg)
    study.atomic(root / 'data/original_config.json', {'minimum_homolog_query_coverage': .5})
    for i in range(n):
        role = 'train' if i < 320 else 'calibration' if i < 440 else 'assessment' if i < 600 else 'crossing'
        r = {'uid': str(i), 'a': 2*i, 'b': 2*i+1, 'label': int(y[i]), 'split': 'train' if i < 320 else 'val', 'role': role, 'length': 16}
        p = {'uid': str(i), 'a': r['a'], 'b': r['b'], 'available': bool(gate[i]), 'gate': float(gate[i]),
             'reason': 'available' if gate[i] else 'shallow_paired_alignment', 'length': 16, 'breakpoint': 8}
        if p['available']:
            import hashlib
            from msa import ALPHABET, encode
            mm=[]
            for pid in (p['a'], p['b']):
                q='ARNDCEQG';mask='***-****'
                seqs=[''.join(ALPHABET[v] for v in rng.integers(0,20,8)) for _ in range(18)]
                m={'query':q,'mask':mask,'query_sha256':hashlib.sha256(q.encode()).hexdigest(),
                   'rows':[(f'k{j}',seqs[j],f'g:f{j%3}:o:c:p') for j in range(18)]}
                monomers[str(pid)]=m;mm.append(m)
            t=np.concatenate([np.vstack([encode(m['query'])]+[encode(row[1]) for row in m['rows'][:8]]) for m in mm],axis=1)
            t[:,[3,11]]=26
            with patch.object(inputs,'ROOT',root):
                ii,selected,pools=inputs.independent_pair(mm[0],mm[1],t,p)
            path=root/'data/independent_inputs'/(p['uid']+'.npz');study.atomic_npz(path,true=t,I=ii)
            p.update({'paired_depth':9,'paired_tokens_sha256':study.token_hash(t),'q_tokens_sha256':study.token_hash(t[:1]),
                      'i_tokens_sha256':study.token_hash(ii),'i_selected_keys':selected,'i_input_file':str(path.relative_to(root)),
                      'i_input_sha256':study.sha(path)})
            oldnull={'null':{'permutation':[0,8,1,2,3,4,5,6,7]},'msa':{'genome_keys':[f'k{j}' for j in range(8)],'taxonomy':[f'g:f{j%3}:o:c:p' for j in range(8)]}}
            audits[p['uid']]=describe(t,ii,p,oldnull,selected,pools,mm,4)
        rows.append(r); pairs.append(p)
    study.atomic(root / 'data/test_monomers.json', monomers)
    for name, value in [('sample', rows), ('pairs', pairs), ('diagnostics', [None]*n), ('dev_family_witnesses', {}),
                        ('v2_metrics', {'diagnostic_stratum_cuts_from_eligible_train': {}}), ('input-audits', audits)]:
        study.atomic(root / 'data' / (name + '.json'), value)
    study.atomic(root / 'data/baseline.json', {r['uid']: {**r, 'score': float(base[i])} for i, r in enumerate(rows) if r['split'] == 'val'})
    def oldhead(dim, coefficient, alpha=1):
        return {'standard_mean': [0.]*dim, 'standard_scale': [1.]*dim, 'pca_mean': [0.]*dim,
                'pca_components': np.eye(dim)[:32].tolist(), 'coef': [[coefficient]*32], 'intercept': [.01], 'alpha': alpha}
    hh = {name: oldhead(dim, coefficient) for name, dim, coefficient in [('true', 128, .04), ('shuffled', 128, .03), ('quality', 68, .02)]}
    oldx = {name: rng.normal(size=(n, dim)) for name, dim in [('true', 128), ('shuffled', 128), ('quality', 68), ('Q', 128), ('C', 128)]}
    for values in oldx.values(): values[gate == 0] = 0
    study.atomic(root / 'data/source_heads.json', hh)
    hh['Q'] = oldhead(128, .015, .5); study.atomic(root / 'data/vz_heads.json', {'Q': hh['Q']})
    study.atomic_npz(root / 'data/source.npz', uids=np.array([r['uid'] for r in rows]), gate=gate,
        **{k: oldx[k] for k in ('true', 'shuffled', 'quality')})
    study.atomic_npz(root / 'data/q_reference_features.npz', uids=np.array([r['uid'] for r in rows]), Q=oldx['Q'])
    hh['C'] = oldhead(128, .023, 1); study.atomic(root / 'data/v3_heads.json', {'C': hh['C']})
    study.atomic_npz(root / 'data/c_reference_features.npz', uids=np.array([r['uid'] for r in rows]), C=oldx['C'])
    train = np.arange(n) < 320; cal = (np.arange(n) >= 320) & (np.arange(n) < 440); ass = (np.arange(n) >= 440) & (np.arange(n) < 600)
    dev = ~train; old = {'results': {}, 'fit_counts': {name: [int(np.sum(m & (y == c))) for c in (0, 1)] for name, m in [('fit', train & (gate > 0)), ('calibration', cal), ('assessment', ass)]}}
    ev = {k: evidence(h, oldx[k]) for k, h in hh.items()}
    scores = {'baseline': base, **{k: study.fuse(base, ev[k], gate, hh[k]['alpha']) for k in hh}}
    vz = {'results': {}}; v3 = {'results': {}}
    for pop, m in [('assessment', ass), ('assessment/eligible', ass & (gate > 0)), ('fixed_dev_sample', dev), ('calibration', cal)]:
        mm = {k: {'ap': float(average_precision_score(y[m], z[m])), 'auroc': float(roc_auc_score(y[m], z[m]))} for k, z in scores.items()}
        v3['results'][pop] = {'metrics': mm}
        vz['results'][pop] = {'metrics': {k:v for k,v in mm.items() if k != 'C'}}
        if pop != 'assessment/eligible': old['results'][pop] = {'metrics': {k: v for k, v in mm.items() if k not in ('Q','C')}}
    for pop, m in [('assessment', ass), ('fixed_dev_sample', dev)]:
        masks = {pop: np.ones(m.sum(), dtype=bool)}
        if pop == 'assessment': masks['assessment/eligible'] = (gate > 0)[m]
        ci = bootstrap([r for r, k in zip(rows, m) if k], y[m], {**{k: z[m] for k, z in scores.items()}, 'I': base[m]}, masks, cfg['fusion'], cfg['seed'])
        for group, intervals in ci.items():
            v3['results'][group]['ap_intervals'] = {k:v for k,v in intervals.items() if 'I' not in CONTRASTS[k]}
            vz['results'][group]['ap_intervals'] = {k:v for k,v in intervals.items() if 'I' not in CONTRASTS[k] and 'C' not in CONTRASTS[k]}
        old['results'][pop]['ap_intervals'] = {b: ci[pop][a] for a, b in [('true_minus_baseline', 'true-minus-baseline'), ('true_minus_shuffled', 'true-minus-shuffled')]}
    study.atomic(root / 'data/source_metrics.json', old); study.atomic(root / 'data/vz_metrics.json', vz); study.atomic(root / 'data/v3_metrics.json', v3)
    duids = np.array([r['uid'] for r in rows if r['split'] == 'val'])
    study.atomic_npz(root / 'data/vy_dev_predictions.npz', uids=duids, labels=y[dev], baseline=base[dev],
        **{'vx_' + k: scores[k][dev] for k in ('true', 'shuffled', 'quality')})
    study.atomic_npz(root / 'data/vz_dev_predictions.npz', uids=duids, labels=y[dev], gate=gate[dev], **{k: z[dev] for k, z in scores.items() if k != 'C'})
    study.atomic_npz(root / 'data/vz_dev_evidence.npz', **{k:z[dev] for k,z in ev.items() if k != 'C'})
    study.atomic_npz(root / 'data/v3_dev_predictions.npz', uids=duids, labels=y[dev], gate=gate[dev], **{k:z[dev] for k,z in scores.items()})
    study.atomic_npz(root / 'data/v3_dev_evidence.npz', **{k:z[dev] for k,z in ev.items()})
    with patch.object(study, 'ROOT', root):
        for p, xx in zip(pairs, x):
            timing = {'orientation_depths': [p['paired_depth']] * 2, 'orientation_layers': list(range(16)) * 2,
                      'length': p['length'], 'breakpoint': p['breakpoint']} if p['available'] else None
            study.save_record(p['uid'], xx, 'fixture', p, timing)
    return rows, pairs, x, base, gate


class AnalysisFlow(unittest.TestCase):
    def patches(self, root, cfg):
        stack = ExitStack()
        for module in (study, analyze, verify, inputs):
            stack.enter_context(patch.object(module, 'ROOT', root))
            stack.enter_context(patch.object(module, 'config', return_value=cfg))
        for module in (analyze, verify): stack.enter_context(patch.object(module, 'verify_freeze', return_value='fixture'))
        stack.enter_context(patch.object(analyze, 'check_budget'))
        monomers=study.read(root/'data/test_monomers.json')
        stack.enter_context(patch.object(inputs, 'monomer_reader', return_value=lambda pid: monomers[str(pid)]))
        stack.enter_context(redirect_stdout(io.StringIO()))
        return stack

    def test_full_pipeline_independent_audit_and_tampering(self):
        cfg = copy.deepcopy(study.config()); cfg['expected']['pairs'] = 640; cfg['fusion']['bootstrap_replicates'] = 20
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); rows, pairs, x, base, gate = fixture(root, cfg)
            with self.patches(root, cfg):
                analyze.main(); verify.main()
                v = study.read(root / 'results/verification.json'); self.assertTrue(v['passed']); self.assertEqual(v['pairs_verified'], 640)
                self.assertEqual(v['independent_sampling_input_audit']['eligible_inputs_verified'], int((gate > 0).sum()))
                h = study.read(root / 'results/heads.json')['I']
                fit = np.array([r['split'] == 'train' and p['available'] for r, p in zip(rows, pairs)])
                np.testing.assert_allclose(h['standard_mean'], x[fit].mean(0)); self.assertEqual(h['fit_rows'], fit.sum())
                with np.load(root / 'results/dev_predictions.npz') as f: saved = {k: f[k].copy() for k in f.files}
                self.assertFalse(np.array_equal(saved['Q'], saved['I']))
                self.assertFalse(np.array_equal(saved['true'], saved['shuffled']))
                np.testing.assert_array_equal(saved['I'][gate[320:] == 0], base[320:][gate[320:] == 0])
                saved['Q'][0] += .1; study.atomic_npz(root / 'results/dev_predictions.npz', **saved)
                with self.assertRaises(AssertionError): verify.main()
                with self.assertRaisesRegex(RuntimeError, 'already started'): analyze.main()

    def test_missing_output_aborts_before_fitting(self):
        cfg = copy.deepcopy(study.config()); cfg['expected']['pairs'] = 640; cfg['fusion']['bootstrap_replicates'] = 20
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); fixture(root, cfg)
            (root / 'features/1.json').unlink(); (root / 'features/1.sha.json').unlink()
            with self.patches(root, cfg), patch.object(analyze, 'fit_head') as fit:
                with self.assertRaisesRegex(RuntimeError, 'Incomplete computational coverage'): analyze.main()
                fit.assert_not_called()
            self.assertFalse((root / 'results/fit-start.json').exists())


if __name__ == '__main__': unittest.main()
