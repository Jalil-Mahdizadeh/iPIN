"""Label-independent GPU parity, masking, orientation and timing qualification."""
import time
import numpy as np
import torch
from study import ROOT, config, read, atomic, sha, now, code_hashes, remaining
from inputs import Inputs
from worker import extract
from contact_encoder import ContactEncoder
from contact_features import all_views


def main():
    started = time.monotonic(); cfg = config(); data = Inputs(); encoder = ContactEncoder()
    if torch.cuda.device_count() != 1 or 'GH200' not in torch.cuda.get_device_name(0):
        raise ValueError('Qualification requires exactly one allocated GH200')
    results = []; cases = read(ROOT/'data/qualification-cases.json')
    for j,i in enumerate(cases):
        remaining(1800); torch.cuda.synchronize(); t = time.monotonic()
        value,maps = extract(i,data,encoder)
        torch.cuda.synchronize(); seconds = time.monotonic()-t
        tokens,_,b = data.materialize(i)
        if j == 0:
            optimized,_ = encoder.one_contact(tokens,b)
            official = encoder.official(tokens,b)
            np.testing.assert_array_equal(optimized,official)
            reverse = np.concatenate([tokens[:,b:],tokens[:,:b]],axis=1)
            reversed_map,_,_ = encoder.symmetric_contacts(reverse,tokens.shape[1]-b)
            np.testing.assert_array_equal(reversed_map.T,maps['true'])
            av,bv = maps['valid_a'],maps['valid_b']
            a_id,b_id = sorted([data.sample[i]['a'],data.sample[i]['b']])
            forward,_ = all_views(maps['true'],av,bv,data.sample[i]['uid'],a_id,b_id)
            backward,_ = all_views(reversed_map,bv,av,data.sample[i]['uid'],b_id,a_id)
            for arm in forward:
                np.testing.assert_allclose(forward[arm],backward[arm],atol=1e-10,rtol=1e-10)
        results.append({'uid':data.sample[i]['uid'],'length':data.sample[i]['length'],
            'depth':len(tokens),'seconds':seconds,'old_v2_seconds':data.diag[i]['elapsed_seconds'],
            'global_errors':value['global_error'], 'minimum_quality':float(min(maps['valid_a'].mean(),maps['valid_b'].mean()))})
        print(results[-1],flush=True)
    # Length-stratified ratios against the completed, identical-input Vx-v2 extraction.
    estimates = []; total = 0.
    for lo,hi in [(0,512),(512,1024),(1024,1536)]:
        measured = [r for r in results if lo < r['length'] <= hi]
        ratio = max(r['seconds']/max(.01,r['old_v2_seconds']) for r in measured)
        ids = [i for i,r in enumerate(data.sample) if data.meta[i]['available'] and lo < r['length'] <= hi]
        old = sum(data.diag[i]['elapsed_seconds'] for i in ids)
        estimate = old*ratio; total += estimate
        estimates.append({'length_upper':hi,'pairs':len(ids),'maximum_measured_ratio':ratio,'prior_seconds':old,'projected_seconds':estimate})
    native = read(ROOT/'qualification/native.json')
    if not native['passed']:
        raise ValueError('Native qualification missing')
    native_estimate = 1825 * max(r['seconds']/r['pairs'] for r in native['checks']) + native['seconds']
    conservative = (total+native_estimate)*cfg['budgets']['timing_safety_factor'] + cfg['budgets']['analysis_reserve_seconds']
    peak = int(torch.cuda.max_memory_allocated())
    report = {'at_utc':now(),'passed':True,'labels_used':False,'test_accessed':False,
        'cases':results,'length_estimates':estimates,'native_estimated_seconds':native_estimate,
        'conservative_total_seconds':conservative,'fits_remaining_budget':conservative < remaining(cfg['budgets']['shutdown_reserve_seconds']),
        'official_api_exact':True,'chain_swap_exact':True,'global_features_exact':True,
        'peak_allocated_gpu_bytes':peak,'seconds':time.monotonic()-started,'code_sha256':code_hashes(),
        'input_manifest_sha256':sha(ROOT/'provenance/inputs.json')}
    atomic(ROOT/'qualification/real.json',report)
    print({k:v for k,v in report.items() if k not in ['code_sha256','cases']},flush=True)
    if not report['fits_remaining_budget']:
        raise TimeoutError('Qualified full pilot does not fit remaining fixed budget')


if __name__ == '__main__':
    main()
