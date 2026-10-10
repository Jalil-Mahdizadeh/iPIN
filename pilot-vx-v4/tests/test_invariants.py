"""Meaningful sampling, corruption, governance and numerical controls."""
import copy,hashlib,json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from sklearn.metrics import average_precision_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import inputs,study
from heads import fit_head
from statistics_v4 import weighted_ap,ap_setup
from decision import decide
from msa import encode


class Sampling(unittest.TestCase):
    def setUp(self):
        self.cfg=study.config();self.p={'uid':'train-1','a':11,'b':22,'breakpoint':6}
        rng=np.random.default_rng(13)
        def monomer(prefix):
            q='ARNDCE';letters='ARNDCEQGHILKMFPSTWYV'
            return {'query':q,'mask':'**-***','query_sha256':hashlib.sha256(q.encode()).hexdigest(),
                    'rows':[(prefix+str(j),''.join(letters[k] for k in rng.integers(0,20,6)),f'g:f{j%3}:o:c:p') for j in range(24)]}
        self.a=monomer('a');self.b=monomer('b')
        self.t=np.concatenate([np.vstack([encode(m['query'])]+[encode(v[1]) for v in m['rows'][:8]]) for m in (self.a,self.b)],axis=1)
        self.t[:,[2,8]]=26
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);(self.root/'data').mkdir()
        study.atomic(self.root/'data/original_config.json',{'minimum_homolog_query_coverage':.5})
        self.patch=patch.object(inputs,'ROOT',self.root);self.patch.start()
    def tearDown(self):self.patch.stop();self.tmp.cleanup()
    def draw(self,a=None,b=None,p=None,rep=0):return inputs.independent_pair(a or self.a,b or self.b,self.t,p or self.p,rep)
    def test_disjoint_accessions_and_intact_rows(self):
        before=copy.deepcopy((self.a,self.b));t=self.t.copy();i,k,_=self.draw()
        self.assertEqual(i.shape,t.shape);np.testing.assert_array_equal(i[0],t[0]);np.testing.assert_array_equal(i==26,t==26)
        for name,m,start in [('A',self.a,0),('B',self.b,6)]:
            self.assertEqual(len(k[name]),8);self.assertEqual(len(set(k[name])),8)
            lookup={a:b for a,b,_ in m['rows']}
            for row,key in enumerate(k[name],1):
                expected=encode(lookup[key]).copy();expected[2]=26;np.testing.assert_array_equal(i[row,start:start+6],expected)
        self.assertEqual(before,(self.a,self.b));np.testing.assert_array_equal(t,self.t)
    def test_deterministic_order_and_label_independence(self):
        first,keys,_=self.draw();a=copy.deepcopy(self.a);b=copy.deepcopy(self.b);a['rows'].reverse();b['rows'].reverse()
        again,other,_=self.draw(a,b,{**self.p,'label':1,'role':'assessment'})
        np.testing.assert_array_equal(first,again);self.assertEqual(keys,other)
    def test_partner_pool_independence_and_distinct_streams(self):
        _,keys,_=self.draw();b=copy.deepcopy(self.b);b['rows']=b['rows'][5:]
        _,other,_=self.draw(b=b);self.assertEqual(keys['A'],other['A'])
        self.assertNotEqual(inputs.digest(self.p,'A','selection'),inputs.digest(self.p,'B','selection'))
        _,alternate,_=self.draw(rep=1);self.assertNotEqual(keys,alternate)
    def test_no_pairwise_rejection(self):
        a=copy.deepcopy(self.a);b=copy.deepcopy(a)
        for m in (a,b):m['rows']=[(k,m['query'],tax) for k,_,tax in m['rows']]
        i,keys,_=self.draw(a,b)
        self.assertEqual(i.shape,self.t.shape);self.assertEqual(len(keys['A']),8)
    def test_insufficient_pool_and_duplicate_fail(self):
        a=copy.deepcopy(self.a);a['rows']=a['rows'][:7]
        with self.assertRaisesRegex(ValueError,'Insufficient'):self.draw(a=a)
        a=copy.deepcopy(self.a);a['rows'].append(a['rows'][0])
        with self.assertRaisesRegex(ValueError,'Duplicate'):self.draw(a=a)
    def test_padding_query_and_depth_corruption(self):
        i,_,_=self.draw()
        changed_query=i.copy();changed_query[0,0]=(changed_query[0,0]+1)%20
        for bad in [i[:-1],changed_query]:
            with self.assertRaises(ValueError):inputs.validate_pair(self.t,bad,6)
        bad=i.copy();bad[1,2]=0
        with self.assertRaises(ValueError):inputs.validate_pair(self.t,bad,6)
    def test_prepared_replay_and_corrupted_intact_membership(self):
        i,keys,_=self.draw();path=self.root/'data/i.npz';study.atomic_npz(path,true=self.t,I=i)
        p={**self.p,'available':True,'paired_depth':len(i),'length':i.shape[1],
           'i_input_file':'data/i.npz','i_input_sha256':study.sha(path),'i_tokens_sha256':study.token_hash(i),
           'paired_tokens_sha256':study.token_hash(self.t),'q_tokens_sha256':study.token_hash(self.t[:1]),'i_selected_keys':keys}
        mono=lambda pid:self.a if pid==11 else self.b
        inputs.prepared(p,replay=True,mono=mono)
        bad=i.copy();bad[1,0]=(bad[1,0]+1)%20;study.atomic_npz(path,true=self.t,I=bad)
        with self.assertRaisesRegex(ValueError,'checksum'):inputs.prepared(p,replay=True,mono=mono)
        p['i_input_sha256']=study.sha(path);p['i_tokens_sha256']=study.token_hash(bad)
        with self.assertRaisesRegex(ValueError,'intact'):inputs.prepared(p,replay=True,mono=mono)
    def test_source_hash_and_coverage_only_monomer(self):
        import gzip
        base=self.root/'base';base.mkdir();m=self.a
        with gzip.open(base/'m.json.gz','wt') as f:json.dump(m,f)
        digest=study.sha(base/'m.json.gz')
        study.atomic(self.root/'data/monomer-catalog.json',{'11':{'path':'m.json.gz','sha256':digest}})
        study.atomic(self.root/'data/monomer-digests.json',{})
        study.atomic(self.root/'data/proteins.json',{'11':{'sequence':m['query'],'sha256':m['query_sha256']}})
        with patch.object(inputs,'BASE',base):
            reader=inputs.monomer_reader();self.assertEqual(reader(11)['query'],m['query']);self.assertEqual(reader.observed_digests,{'11':digest})
            (base/'m.json.gz').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'Changed'):inputs.monomer_reader()(11)


class Governance(unittest.TestCase):
    def test_preflight_preparation_contract_and_incomplete_rejection(self):
        from preflight import validate_preparation
        cfg=study.config();expected=cfg['expected']
        manifest={'required_pairs':expected['eligible_train']+expected['eligible_dev'],
            'counts':copy.deepcopy(expected),'verified_C_feature_records':expected['pairs'],
            'primary_I_replicate':0,'test_accessed':False,'outcome_heads_fitted':False,
            'R_evaluated':False,'labels_used_for_sampling':False}
        validate_preparation(manifest,cfg)
        for key,value in [('required_pairs',4452),('verified_C_feature_records',7999),
                          ('primary_I_replicate',1),('test_accessed',True),
                          ('outcome_heads_fitted',True),('R_evaluated',True),('labels_used_for_sampling',True)]:
            bad=copy.deepcopy(manifest);bad[key]=value
            with self.assertRaises(ValueError):validate_preparation(bad,cfg)
        bad=copy.deepcopy(manifest);bad['counts']['assessment']-=1
        with self.assertRaises(ValueError):validate_preparation(bad,cfg)
        del bad['required_pairs']
        with self.assertRaises(KeyError):validate_preparation(bad,cfg)

    def test_alpha_zero_still_rejects_nonfinite_and_fallback_exact(self):
        base=np.array([.2,.3]);gate=np.array([1.,0.]);ev=np.array([.7,2.])
        self.assertEqual(study.fuse(base,ev,gate,1)[1],base[1])
        ev[0]=np.nan
        with self.assertRaises(FloatingPointError):study.fuse(base,ev,gate,0)
    def test_train_only_transforms_and_calibration_only_selection(self):
        rng=np.random.default_rng(7);x=rng.normal(size=(300,128));y=np.arange(300)%2
        fit=np.arange(300)<150;cal=(np.arange(300)>=150)&(np.arange(300)<230)
        base=rng.normal(size=300);gate=np.ones(300);cfg=study.config()
        h,_,_=fit_head(x,y,fit,cal,base,gate,cfg)
        xx=x.copy();xx[230:]+=30;yy=y.copy();yy[230:]=1-yy[230:]
        hh,_,_=fit_head(xx,yy,fit,cal,base,gate,cfg)
        np.testing.assert_array_equal(h['coef'],hh['coef']);self.assertEqual(h['alpha'],hh['alpha'])
        np.testing.assert_allclose(h['standard_mean'],x[fit].mean(0))
    def test_ap_with_ties_zero_weights(self):
        y=np.array([0,1,1,0,1]);z=np.array([1.,1.,0.,2.,0.]);w=np.array([0.,2.,3.,2.,1.])
        self.assertAlmostEqual(weighted_ap(ap_setup(y,z),w),average_precision_score(y,z,sample_weight=w),places=12)
    def test_feature_corruption_and_identity_fail(self):
        with tempfile.TemporaryDirectory() as d,patch.object(study,'ROOT',Path(d)):
            p={'uid':'x','available':False,'gate':0.};study.save_record('x',np.zeros(128),'f',p)
            self.assertIsNotNone(study.load_record('x','f',p))
            with self.assertRaises(ValueError):study.load_record('x','f',{**p,'gate':1.})
            (Path(d)/'features/x.json').write_text('{}')
            with self.assertRaises(ValueError):study.load_record('x','f',p)
    def test_decision_requires_useful_gain_and_bounded_loss(self):
        from statistics_v4 import CONTRASTS
        pts={k:0. for k in CONTRASTS};ci={k:{'low':-.02,'high':.02} for k in CONTRASTS}
        group={'contrasts':pts,'ap_intervals':ci,'metrics':{'I':{'auroc':.7},'baseline':{'auroc':.7}}}
        groups={'assessment':group,'assessment/eligible':{'ap_intervals':ci}}
        self.assertFalse(decide(groups,study.config())['independent_recovery_within_margin'])
        pts['I_minus_baseline']=.02;ci['I_minus_baseline']={'low':.005,'high':.03}
        for k in ('I_minus_true','I_minus_shuffled'):ci[k]={'low':-.009,'high':.005}
        d=decide(groups,study.config());self.assertTrue(d['independent_recovery_within_margin'])
        self.assertFalse(d['all_I_true_shuffled_equivalent'])
        ci['I_minus_shuffled']['low']=-.011
        self.assertFalse(decide(groups,study.config())['independent_recovery_within_margin'])


if __name__=='__main__':unittest.main()
