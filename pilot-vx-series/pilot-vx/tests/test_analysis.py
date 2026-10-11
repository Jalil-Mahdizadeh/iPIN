"""End-to-end analysis fixture verifies TRAIN-only transforms and saved heads."""
import json
import io
from contextlib import redirect_stdout
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import analyze
from common import atomic, config
from features import save_record


class AnalysisInvariants(unittest.TestCase):
    def test_train_only_transform_and_serialized_head(self):
        rng=np.random.default_rng(91);cfg=config();cfg['fusion']['bootstrap_replicates']=25
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'features').mkdir();(root/'data').mkdir();sample=[];baseline={};train_x=[]
            for i in range(800):
                split='train' if i<400 else 'val';label=i%2
                role='train' if i<400 else ('calibration' if i<600 else 'assessment')
                r={'uid':f'{split}-{i:06d}','split':split,'label':label,'role':role,'a':2*i,'b':2*i+1,'length':600}
                sample.append(r);true=rng.normal(size=128);true[0:8]+=6*label
                # Large DEV nuisance shift would pollute normalization if DEV were used in fitting.
                if split=='val':true[120:]+=10
                else:train_x.append(true.copy())
                value={'true':true.tolist(),'shuffled':rng.normal(size=128).tolist(),'quality':rng.normal(size=68).tolist(),
                       'gate':1.,'available':True,'null':{'changed_fraction':1.}}
                save_record(root/'features',r['uid'],value,'fixture')
                if split=='val':baseline[r['uid']]={'score':float(rng.normal(scale=.05))}
            atomic(root/'data/sample.json',sample);atomic(root/'data/baseline.json',baseline)
            with patch.object(analyze,'ROOT',root),patch.object(analyze,'verify_freeze',return_value='fixture'),patch.object(analyze,'config',return_value=cfg),redirect_stdout(io.StringIO()):
                analyze.main()
            heads=json.loads((root/'results/heads.json').read_text());head=heads['true']
            np.testing.assert_allclose(head['standard_mean'],np.mean(train_x,axis=0),rtol=1e-12,atol=1e-12)
            report=json.loads((root/'results/metrics.json').read_text())
            self.assertFalse(report['test_used']);self.assertFalse(report['structural_heads_used'])
            self.assertGreater(report['results']['assessment']['metrics']['true']['ap'],.95)
            # Saved numeric state reconstructs finite predictions without unsafe pickle loading.
            x=np.asarray(train_x[:3]);standard=(x-head['standard_mean'])/head['standard_scale']
            reduced=(standard-head['pca_mean'])@np.asarray(head['pca_components']).T
            logits=reduced@np.asarray(head['coef']).T+head['intercept']
            self.assertTrue(np.isfinite(logits).all())


if __name__=='__main__':unittest.main()
