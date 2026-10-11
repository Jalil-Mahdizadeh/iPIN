"""Score once after complete extraction; never fit or select on TEST."""
import fcntl
import time
from collections import Counter
import numpy as np
from shared import ROOT, config, read, atomic, now, sha, save_npz, verify_freeze, source_contract, ARMS
from collect import collect
from predict import score, prohibit_training
from statistics_rx import summarize, intervals


def interpretation(primary, criteria):
    m = primary['metrics']; ci = primary['intervals']; delta = primary['contrasts']['true_minus_baseline']
    native = delta >= criteria['minimum_delta_ap'] and ci['true_minus_baseline']['low'] > criteria['minimum_ci_lower']
    quality = ci['true_minus_quality']['low'] > criteria['minimum_ci_lower']
    auroc = m['true']['auroc'] >= m['baseline']['auroc'] - criteria['maximum_auroc_loss']
    return {'status': 'useful_predictive_gain_supported' if native and quality and auroc else 'no_go_or_inconclusive',
        'primary_ap_change': delta, 'native_effect_size_and_interval_pass': native,
        'beyond_quality_supported': quality, 'auroc_noninferiority': auroc,
        'pairing_specific_claim': False, 'test_authorized': True, 'test_accessed': True,
        'automatic_followup': False}


def strata_masks(rows, records, available):
    length = np.asarray([r['length'] for r in rows]); depth = np.asarray([r['depth'] for r in records])
    masks = {'eligible': available, 'fallback': ~available}
    for lo, hi in [(0, 512), (512, 1024), (1024, 1536), (1536, 2048), (2048, 10**9)]:
        masks[f'length_{lo+1}_{hi}'] = (length > lo) & (length <= hi)
    for lo, hi in [(0, 0), (1, 32), (33, 64), (65, 127), (128, 128)]:
        masks[f'depth_{lo}_{hi}'] = (depth >= lo) & (depth <= hi)
    quality = np.asarray([min(r['msa']['quality_a'], r['msa']['quality_b']) if r.get('msa') else 0. for r in records])
    for lo, hi in [(.5, .75), (.75, .9), (.9, 1.01)]:
        masks[f'quality_mask_fraction_{lo}_{hi}'] = available & (quality >= lo) & (quality < hi)
    return masks


def evaluate():
    started = time.monotonic(); cfg = config(); fp = verify_freeze(); heads = source_contract()
    prohibit_training()
    if (ROOT / 'results/evaluation-start.json').exists():
        raise RuntimeError('TEST evaluation already attempted; no automatic second evaluation')
    rows = read(ROOT / 'data/pairs.json'); shards = read(ROOT / 'data/shards.json')
    features, records, workers = collect(ROOT, rows, shards, fp, cfg['chunk_rows'])
    if len(rows) != cfg['test_pairs']:
        raise ValueError('Wrong original TEST cohort')
    uids = np.asarray([r['uid'] for r in rows])
    with np.load(ROOT / 'data/labels.npz', allow_pickle=False) as f:
        y = f['y']; np.testing.assert_array_equal(f['uid'], uids)
    with np.load(ROOT / 'data/baseline.npz', allow_pickle=False) as f:
        base = f['baseline']; np.testing.assert_array_equal(f['uid'], uids)
        np.testing.assert_array_equal(base, f['orientations'].mean(axis=1))
    if y.shape != (len(rows),) or set(y.tolist()) != {0, 1} or base.shape != y.shape or not np.isfinite(base).all():
        raise ValueError('Incomplete labels/native TEST baseline')
    atomic(ROOT / 'results/evaluation-start.json', {'at_utc': now(), 'fingerprint': fp,
        'complete_rows': len(rows), 'heads_sha256': cfg['heads_sha256'], 'alphas': cfg['fixed_alphas'],
        'heads_fitted': False, 'calibration_performed': False, 'test_authorized': True})
    scores, evidence = score(features, base, heads, cfg['fixed_alphas'])
    available = features['available']; gate = features['gate']
    save_npz(ROOT / 'results/test_predictions.npz', uid=uids, y=y, gate=gate, available=available, **scores)
    save_npz(ROOT / 'results/test_evidence.npz', **evidence)
    primary = summarize(y, scores, available)
    primary['intervals'] = intervals(rows, y, scores, cfg['bootstrap_replicates'], cfg['seed'])
    strata = {name: summarize(y[mask], {k: z[mask] for k, z in scores.items()}, available[mask])
              for name, mask in strata_masks(rows, records, available).items()}
    def distribution(values):
        a = np.asarray(values, dtype=float)
        return {'count': len(a), 'quantiles_0_25_50_75_100': np.quantile(a, [0, .25, .5, .75, 1]).tolist()} if len(a) else {'count': 0}
    result = {'at_utc': now(), 'fingerprint': fp, 'primary': 'full_test', 'full_test': primary,
        'strata_descriptive': strata, 'availability_reasons': dict(Counter(r['reason'] for r in records)),
        'depth_eligible': distribution([r['depth'] for r in records if r['available']]),
        'neff_eligible': distribution([r['msa']['neff'] for r in records if r['available']]),
        'feature_source_counts': dict(Counter(r['kind'] for r in records)), 'alphas': cfg['fixed_alphas'],
        'heads_sha256': cfg['heads_sha256'], 'training_rows_used_by_frozen_heads': {k: h['fit_rows'] for k, h in heads.items()},
        'bootstrap': 'Matched protein Poisson multipliers; fixed fitted predictions; endpoint-product weights',
        'historically_evaluated_test': True, 'post_trainval_no_go_user_authorization': True,
        'heads_fitted': False, 'calibration_performed': False, 'shuffled_arm': False,
        'test_authorized': True, 'test_accessed': True, 'seconds': time.monotonic() - started}
    atomic(ROOT / 'results/metrics.json', result)
    answer = {**interpretation(primary, cfg['predictive_criteria']), 'at_utc': now(), 'fingerprint': fp,
              'metrics_sha256': sha(ROOT / 'results/metrics.json')}
    atomic(ROOT / 'results/decision.json', answer)
    lines = ['# Vx-128 full TEST benchmark', '', f"Fixed practical criteria: **{answer['status']}**.", '',
        f"All {len(rows):,} original TEST pairs are included; {int(available.sum()):,} receive MSA evidence. Unavailable pairs retain exact native scores.", '',
        '| Frozen model | AP | AUROC |', '|---|---:|---:|']
    names = {'baseline': 'Native PLM-interact', 'true': 'Native + true Vx-128', 'quality': 'Native + quality/profile'}
    for arm, m in primary['metrics'].items():
        lines.append(f"| {names[arm]} | {m['ap']:.6f} | {m['auroc']:.6f} |")
    lines += ['', '| Full TEST AP contrast | Estimate | 95% protein interval |', '|---|---:|---|']
    for name, ci in primary['intervals'].items():
        lines.append(f"| {name} | {primary['contrasts'][name]:+.6f} | [{ci['low']:+.6f}, {ci['high']:+.6f}] |")
    lines += ['', 'Both encoders and both TRAIN-fitted heads are frozen. VAL-selected fusion weights are true = 1 and quality = 1. No TEST fitting, calibration, model selection, or shuffled encoding.', '',
        'This evaluation was explicitly authorized after TRAIN/VAL missed the advancement criteria. The earlier no-go finding is unchanged. Original Bernett TEST has historical iPIN results; this is not a newly blind evaluation. Intervals condition on the fitted models and omit research-selection uncertainty.', '',
        'Practical criteria: at least +0.010 AP over native with a positive 95% lower bound, AUROC loss at most 0.005, and a positive AP interval over quality/profile. This benchmark cannot establish pairing-specific co-evolution. Fixed strata, coverage reasons and depth/Neff summaries are in metrics.json. No outcome-dependent follow-up is automatic.', '',
        'Independent verification and COMPLETE.json are required for accepted completion.']
    (ROOT / 'results/REPORT.md').write_text('\n'.join(lines) + '\n')
    print(answer, flush=True)


def main():
    with (ROOT / 'provenance/evaluation.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        evaluate()


if __name__ == '__main__':
    main()
