import sys
import unittest
from pathlib import Path
import numpy as np
from sklearn.metrics import average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from contact_features import summarize,all_views,boundary_mask,window_sums
from study import fuse
from heads import crossfit,choose_alpha
from statistics_v7 import ap_setup,weighted_ap


class ContactSummaryTests(unittest.TestCase):
    def test_masks_remove_evidence_and_denominator(self):
        p=np.zeros((10,10));a=np.arange(10)<5;b=np.arange(10)<5;p[~a,:]=1;p[:,~b]=1
        x,m=summarize(p,a,b)
        np.testing.assert_array_equal(x,np.zeros(6));self.assertEqual(m['valid_cells'],25)

    def test_transpose_and_coordinate_invariants(self):
        rng=np.random.default_rng(123);p=rng.random((21,17));a=rng.random(21)>.2;b=rng.random(17)>.2
        x,_=all_views(p,a,b,'train-case',11,22);y,_=all_views(p.T,b,a,'train-case',22,11)
        for key in x:np.testing.assert_allclose(x[key],y[key],atol=1e-12)
        np.testing.assert_allclose(x['standard'][:5],x['coordinate'][:5],atol=1e-12)

    def test_localized_patch_differs_from_coordinate_null(self):
        p=np.zeros((30,30));p[10:15,10:15]=1;good=np.ones(30,dtype=bool)
        views,_=all_views(p,good,good,'train-case',11,22)
        self.assertEqual(views['standard'][5],1.)
        self.assertLess(views['coordinate'][5],.5)
        np.testing.assert_allclose(views['standard'][:5],views['coordinate'][:5],atol=1e-12)

    def test_boundary_corner_artifacts_excluded(self):
        p=np.zeros((20,20));p[-5:,:5]=1;p[:5,-5:]=1;g=np.ones(20,dtype=bool)
        raw,_=summarize(p,g,g);clean,_=summarize(p,g,g,boundary_mask(p.shape))
        self.assertEqual(raw[0],1.);np.testing.assert_array_equal(clean,np.zeros(6))

    def test_small_maps_no_supported_window(self):
        x,m=summarize(np.full((2,3),.6),np.ones(2,bool),np.ones(3,bool))
        np.testing.assert_allclose(x,[.6,1,1,1,1,0]);self.assertEqual(m['supported_windows'],0)

    def test_insufficient_window_support(self):
        p=np.ones((5,5));a=np.array([1,1,1,0,0],bool);b=np.ones(5,bool)
        x,m=summarize(p,a,b);self.assertEqual(x[5],0.);self.assertEqual(m['supported_windows'],0)

    def test_invalid_values_fail_closed(self):
        for bad in [np.nan,np.inf,-.01,1.01]:
            with self.assertRaises(ValueError):summarize(np.array([[bad]]),[True],[True])

    def test_window_integral_against_direct_sums(self):
        x=np.random.default_rng(17).random((11,13))
        direct=np.array([[x[i:i+5,j:j+5].sum() for j in range(9)] for i in range(7)])
        np.testing.assert_allclose(window_sums(x,5),direct,atol=1e-12)


class GovernanceTests(unittest.TestCase):
    def test_exact_fallback(self):
        base=np.array([1.1,-3.,8.]);out=fuse(base,np.array([2.,4.,6.]),np.array([0.,1.,0.]),.5)
        np.testing.assert_array_equal(out[[0,2]],base[[0,2]]);self.assertEqual(out[1],-1.)

    def test_alpha_zero_does_not_hide_invalid_computation(self):
        with self.assertRaises(ValueError):fuse([0.],[np.nan],[1.],0.)

    def test_val_labels_cannot_change_fits_or_calibration(self):
        rng=np.random.default_rng(5);x=rng.normal(size=(100,6));y=np.tile([0,1],50)
        train=np.arange(100)<80;fold=np.full(100,-1);fold[:80]=np.arange(80)%5
        h,z,o=crossfit(x,y,train,fold);changed=y.copy();changed[~train]=1-changed[~train]
        hh,zz,oo=crossfit(x,changed,train,fold)
        self.assertEqual(h,hh);np.testing.assert_array_equal(z,zz);np.testing.assert_array_equal(o,oo)
        b=rng.normal(size=100);g=np.ones(100)
        self.assertEqual(choose_alpha(b,o,g,y,train),choose_alpha(b,oo,g,changed,train))

    def test_val_marked_as_calibration_fails(self):
        x=np.ones((20,6));y=np.tile([0,1],10);train=np.arange(20)<10;fold=np.arange(20)%5
        with self.assertRaises(ValueError):crossfit(x,y,train,fold)

    def test_alpha_tie_chooses_zero(self):
        y=np.array([0,1,0,1]);result=choose_alpha(np.arange(4.),np.zeros(4),np.ones(4),y,np.ones(4,bool))
        self.assertEqual(result['alpha'],0)

    def test_protein_bootstrap_ap_ties_match_sklearn(self):
        y=np.array([0,1,1,0,1,0]);z=np.array([.2,.2,.7,.5,.7,.1]);w=np.array([2,1,0,3,4,0])
        self.assertAlmostEqual(weighted_ap(ap_setup(y,z),w),average_precision_score(y,z,sample_weight=w),places=14)


class ReleasedHeadTests(unittest.TestCase):
    def test_probability_symmetrization_order(self):
        import torch
        from MSA_Pairformer.regression import LogisticRegressionContactHead
        torch.manual_seed(10);head=LogisticRegressionContactHead(dim_pairwise=4)
        z=torch.randn(1,5,5,4)
        with torch.no_grad():
            raw=torch.sigmoid(torch.nn.functional.linear(head.init_ln(z),head.weight)+head.bias).squeeze(-1)
            expected=(raw+raw.transpose(-1,-2))/2;actual=head(z)
            torch.testing.assert_close(actual,expected,atol=0,rtol=0)
            wrong=head((z+z.transpose(1,2))/2)
            self.assertGreater(float((actual-wrong).abs().max()),.001)


if __name__=='__main__':unittest.main()
