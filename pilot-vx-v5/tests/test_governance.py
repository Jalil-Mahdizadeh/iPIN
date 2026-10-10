import copy,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.metrics import average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import study
from heads import fit_head
from statistics_v5 import CONTRASTS,ap_setup,weighted_ap
from decision import decide
from preflight import validate_preparation


class Governance(unittest.TestCase):
    def test_preparation_metadata_contract(self):
        cfg=study.config();e=cfg['expected'];v={'required_pairs':e['eligible_train']+e['eligible_dev'],'counts':e,
            'increased_depth_pairs':4000,'test_accessed':False,'outcome_heads_fitted':False,'R_evaluated':False,'labels_used_for_sampling':False}
        validate_preparation(v,cfg)
        for key,value in [('required_pairs',4452),('increased_depth_pairs',0),('increased_depth_pairs',4454),
                          ('test_accessed',True),('outcome_heads_fitted',True),('labels_used_for_sampling',True)]:
            with self.assertRaises(ValueError):validate_preparation({**v,key:value},cfg)
    def test_train_only_fit_calibration_only_selection(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(300,128));y=np.arange(300)%2
        fit=np.arange(300)<150;cal=(np.arange(300)>=150)&(np.arange(300)<230);base=rng.normal(size=300);gate=np.ones(300)
        a,_,_=fit_head(x,y,fit,cal,base,gate,study.config());xx=x.copy();xx[230:]+=30;yy=y.copy();yy[230:]=1-yy[230:]
        b,_,_=fit_head(xx,yy,fit,cal,base,gate,study.config())
        np.testing.assert_array_equal(a['coef'],b['coef']);self.assertEqual(a['alpha'],b['alpha']);np.testing.assert_allclose(a['standard_mean'],x[fit].mean(0))
    def test_all_arm_integrity_and_fallback(self):
        with tempfile.TemporaryDirectory() as d,patch.object(study,'ROOT',Path(d)):
            p={'uid':'x','available':False,'gate':0.};f={k:np.zeros(n) for k,n in study.DIMENSIONS.items()}
            study.save_record('x',f,'fp',p);self.assertIsNotNone(study.load_record('x','fp',p))
            with self.assertRaises(ValueError):study.validate_features({k:v for k,v in f.items() if k!='S256'},p)
            f['T256'][0]=1
            with self.assertRaises(ValueError):study.validate_features(f,p)
            (Path(d)/'features/x.json').write_text('{}')
            with self.assertRaises(ValueError):study.load_record('x','fp',p)
    def test_finite_even_alpha_zero_and_exact_fallback(self):
        base=np.array([.2,.3]);gate=np.array([1.,0.]);ev=np.array([.7,2.])
        self.assertEqual(study.fuse(base,ev,gate,1)[1],base[1]);ev[0]=np.nan
        with self.assertRaises(FloatingPointError):study.fuse(base,ev,gate,0)
    def test_weighted_ap_ties_and_zero_weights(self):
        y=np.array([0,1,1,0,1]);z=np.array([1.,1.,0.,2.,0.]);w=np.array([0.,2.,3.,2.,1.])
        self.assertAlmostEqual(weighted_ap(ap_setup(y,z),w),average_precision_score(y,z,sample_weight=w),places=12)
    def test_decision_does_not_treat_uncertainty_as_equivalence(self):
        cfg=study.config();m={'contrasts':{k:0. for k in CONTRASTS},'ap_intervals':{k:{'low':-.02,'high':.02} for k in CONTRASTS},'metrics':{'true':{'auroc':.7},'T256':{'auroc':.7}}}
        groups={'assessment':m};d=decide(groups,cfg);self.assertEqual(d['status'],'depth_comparison_inconclusive');self.assertFalse(d['depths_equivalent_within_margin'])
        m['contrasts']['T256_minus_true']=.015;m['ap_intervals']['T256_minus_true']={'low':.002,'high':.028}
        self.assertTrue(decide(groups,cfg)['meaningful_depth_gain'])
        m['metrics']['T256']['auroc']=.69;self.assertFalse(decide(groups,cfg)['meaningful_depth_gain'])
        m['contrasts']['T256_minus_true']=.004;m['ap_intervals']['T256_minus_true']={'low':.001,'high':.008}
        d=decide(groups,cfg);self.assertTrue(d['material_gain_ruled_out_within_interval']);self.assertFalse(d['material_depth_loss'])


if __name__=='__main__':unittest.main()
