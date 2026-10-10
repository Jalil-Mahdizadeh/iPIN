import copy,hashlib,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import inputs,study
from msa import pair,quality_features


class DepthInputs(unittest.TestCase):
    def setUp(self):
        self.oc=study.read(study.BASE/'config.json');rng=np.random.default_rng(44)
        def mono(offset):
            letters='ARNDCEQGHILKMFPSTWYV';q=letters*2
            return {'query':q,'query_sha256':hashlib.sha256(q.encode()).hexdigest(),'mask':'*'*35+'-'*5,
                'rows':[(f'key{k}',''.join(letters[i] for i in rng.integers(0,20,40)),f'g:f{k%40}:o:c:p') for k in range(320)]}
        self.a,self.b=mono(0),mono(1);self.t,self.meta=pair(self.a,self.b,self.oc)
        self.p={'uid':'x','a':1,'b':2,'gate':self.meta['gate'],'breakpoint':40}
    def draw(self,a=None,b=None,t=None,meta=None,p=None):
        return inputs.extend(a or self.a,b or self.b,self.t if t is None else t,self.meta if meta is None else meta,self.p if p is None else p,self.oc)
    def test_nested_original_pair_algorithm_and_shuffle(self):
        before=copy.deepcopy((self.a,self.b));t,s,m,n=self.draw()
        self.assertEqual(len(t),256);np.testing.assert_array_equal(t[:128],self.t)
        self.assertEqual(m['genome_keys'][:127],self.meta['genome_keys']);self.assertEqual(m['gate'],self.meta['gate'])
        np.testing.assert_array_equal(s[:,:40],t[:,:40]);np.testing.assert_array_equal(s[:,40:],t[n['permutation'],40:])
        self.assertEqual(before,(self.a,self.b))
    def test_label_and_cache_order_invariance(self):
        t,s,m,n=self.draw();a=copy.deepcopy(self.a);b=copy.deepcopy(self.b);a['rows'].reverse();b['rows'].reverse()
        tt,ss,mm,nn=self.draw(a,b,p={**self.p,'label':1,'role':'assessment'})
        np.testing.assert_array_equal(t,tt);np.testing.assert_array_equal(s,ss);self.assertEqual(m,mm);self.assertEqual(n,nn)
    def test_no_padding_of_shallow_or_partial_depth(self):
        for count in (20,160):
            a=copy.deepcopy(self.a);b=copy.deepcopy(self.b);a['rows']=a['rows'][:count];b['rows']=b['rows'][:count]
            old,meta=pair(a,b,self.oc);p={**self.p,'gate':meta['gate']}
            t,s,m,n=self.draw(a,b,old,meta,p)
            self.assertEqual(len(t),count+1);self.assertEqual(len(m['genome_keys']),len(set(m['genome_keys'])))
    def test_prefix_mask_gate_corruption_fails(self):
        bad=self.t.copy();bad[1,1]=(bad[1,1]+1)%20
        with self.assertRaisesRegex(ValueError,'prefix'):self.draw(t=bad)
        with self.assertRaisesRegex(ValueError,'gate'):self.draw(p={**self.p,'gate':.1})
        bad=self.t.copy();bad[1,35]=1
        with self.assertRaises(ValueError):inputs.validate_tokens(bad,40)
    def test_quality_update_matches_original_constructor(self):
        _,_,meta,_=self.draw();old=quality_features(self.a,self.b,self.meta)
        actual=inputs.quality_at_depth(old,self.meta,meta)
        np.testing.assert_array_equal(actual,quality_features(self.a,self.b,meta))
        np.testing.assert_array_equal(actual[np.arange(68)!=65],old[np.arange(68)!=65])
    def test_corrupt_prepared_input_rejected(self):
        t,s,m,n=self.draw();q=inputs.quality_at_depth(quality_features(self.a,self.b,self.meta),self.meta,m)
        with tempfile.TemporaryDirectory() as d,patch.object(inputs,'ROOT',Path(d)):
            root=Path(d);path=root/'i.npz';study.atomic_npz(path,T256=t,S256=s,P256=q)
            p={**self.p,'available':True,'depth_input_file':'i.npz','depth_input_sha256':study.sha(path),
               'depth_256':256,'length':80,'paired_depth':128,'paired_tokens_sha256':study.token_hash(self.t),
               'q_tokens_sha256':study.token_hash(self.t[:1]),'T256_tokens_sha256':study.token_hash(t),
               'S256_tokens_sha256':study.token_hash(s),'P256_sha256':inputs.vector_hash(q),'null256':n,'msa256':m}
            inputs.prepared(p)
            s=s.copy();s[1,1]=(s[1,1]+1)%20;study.atomic_npz(path,T256=t,S256=s,P256=q)
            with self.assertRaisesRegex(ValueError,'checksum'):inputs.prepared(p)
            p['depth_input_sha256']=study.sha(path);p['S256_tokens_sha256']=study.token_hash(s)
            with self.assertRaises(AssertionError):inputs.prepared(p)


if __name__=='__main__':unittest.main()
