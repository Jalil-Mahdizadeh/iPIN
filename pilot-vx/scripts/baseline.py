"""Reuse only hash-verified native Bernett DEV predictions with the exact contract."""
import json
import numpy as np
from common import ROOT, PROJECT, atomic, config, now, sha
import hashlib


def main():
    src=PROJECT/'benchmark-v1';cfg=config();selection=json.loads((src/'provenance/selection.json').read_text())
    native=selection['models']['native-bernett'];assert native['sha256']==cfg['baseline']['sha256']
    assert sha(PROJECT/cfg['baseline']['checkpoint'])==native['sha256']
    oldcfg=json.loads((src/'config.json').read_text())
    assert oldcfg['scoring']=='mean AB/BA logits' and oldcfg['test_sequence_truncation'] is None
    assert oldcfg['precision']=='FP32 parameters; BF16 autocast; TF32 disabled'
    names=['config.json','provenance/selection.json']+[f'scripts/{n}' for n in ['common.py','infer.py','frozen_data.py','frozen_model.py']]
    fingerprint=hashlib.sha256(json.dumps({n:sha(src/n) for n in names},sort_keys=True).encode()).hexdigest()
    qualified=json.loads((src/'provenance/qualification.json').read_text());assert qualified['passed'] and qualified['prediction_fingerprint']==fingerprint
    assert sha(src/'data/val.npy')==selection['data']['sha256']['val.npy']
    rows=np.load(src/'data/val.npy');found={};sources=[]
    for p in sorted((src/'predictions/native-bernett-val').glob('rank-*-chunk-*.npz')):
        meta=json.loads(p.with_suffix('.json').read_text());assert meta['fingerprint']==fingerprint and sha(p)==meta['sha256']
        with np.load(p,allow_pickle=False) as a:z=a['predictions']
        assert z.shape[1]==4 and np.isfinite(z).all()
        ids=z[:,0].astype(int);assert np.array_equal(z[:,0],ids)
        assert np.array_equal(z[:,1],rows[ids,2])
        for i,ab,ba in zip(ids,z[:,2],z[:,3]):
            r=rows[i];original=int(r[3]);assert original not in found
            found[original]={'a':int(r[0]),'b':int(r[1]),'label':int(r[2]),'ab':float(ab),'ba':float(ba),'score':float((ab+ba)/2)}
        sources.append({'path':str(p.relative_to(PROJECT)),'sha256':meta['sha256']})
    assert len(found)==len(rows)
    sample=json.loads((ROOT/'data/sample.json').read_text());out={};missing=[]
    for r in sample:
        if r['split']!='val':continue
        if r['row'] not in found:missing.append(r['uid']);continue
        b=found[r['row']];assert all(r[k]==b[k] for k in ['a','b','label']);out[r['uid']]=b
    atomic(ROOT/'data/baseline.json',out)
    atomic(ROOT/'provenance/baseline-reuse.json',{'at_utc':now(),'native_sha256':native['sha256'],'fingerprint':fingerprint,
           'rows_reused':len(out),'missing_rows':missing,'sources':sources,'test_predictions_read':False,
           'baseline_sha256':sha(ROOT/'data/baseline.json')})
    print(json.dumps({'rows_reused':len(out),'missing_rows':missing}),flush=True)
    if missing:raise RuntimeError('Original DEV rows absent from audited predecessor require qualified fresh baseline inference')


if __name__=='__main__':main()
