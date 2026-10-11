"""Scientific invariants: query coordinates, pairing keys, controls and ambiguity."""
import io
import tarfile
import sys
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from msa import aligned, cached_monomer, records, pair, shuffled, quality_features, encode, LOOKUP
from common import config, text_hash


class MSAInvariants(unittest.TestCase):
    def test_insertions_and_multiline(self):
        self.assertEqual(aligned('ACde.D-F'),'ACD-F')
        self.assertEqual(list(records(io.StringIO('>mask\n**\n**\n>query\nAC\nDE\n'))),[('mask','****'),('query','ACDE')])

    def test_fast_ascii_parser_matches_reference(self):
        rng=np.random.default_rng(13);alphabet=np.array(list('ARNDCEQGHILKMFPSTWYVXBZUO-abcdefghijklmnopqrstuv.'))
        for _ in range(100):
            text=''.join(rng.choice(alphabet,1000))
            expected=''.join(c for c in text if not c.islower() and c!='.')
            self.assertEqual(aligned(text),expected)
            np.testing.assert_array_equal(encode(expected),[LOOKUP[c] for c in expected])

    def test_streaming_tar_requires_no_seeking(self):
        blob=io.BytesIO();content=b'>mask\n****\n>query\nACDE\n>genome g:f\nACDF\n'
        with tarfile.open(fileobj=blob,mode='w:gz') as tf:
            info=tarfile.TarInfo('some/path.a3m');info.size=len(content);tf.addfile(info,io.BytesIO(content))
        blob.seek(0)
        with tarfile.open(fileobj=blob,mode='r|gz') as tf:
            for member in tf:
                with tf.extractfile(member) as fh:
                    result=list(records(line.decode('ascii') for line in fh))
                    self.assertEqual(result[1],('query','ACDE'))

    def test_exact_context_exclusion_and_conflicting_keys(self):
        r=[('a x:y','ACDE'),('b x:y','ACDF'),('b x:y','ACDG'),('c x:y','ACDH')]
        m=cached_monomer(iter(r),'ACDE','****',{text_hash('ACDE')},10)
        self.assertEqual([x[0] for x in m['rows']],['c'])
        self.assertEqual(m['stats']['excluded_human_context'],1)
        self.assertEqual(m['stats']['conflicting_keys'],1)

    def test_archive_J_ambiguity_preserves_coordinates(self):
        m=cached_monomer(iter([('genome g:f','ACJE')]),'ACDE','****',set(),10)
        self.assertEqual(m['rows'][0][1],'ACXE')
        self.assertEqual(m['stats']['ambiguous_J_residues_to_X'],1)

    def test_cache_selection_order_independent(self):
        rows=[(f'genome{i} g:f','ACD'+c) for i,c in enumerate('EFGHIKLMNP')]
        a=cached_monomer(iter(rows),'ACDE','****',set(),4)
        b=cached_monomer(iter(reversed(rows)),'ACDE','****',set(),4)
        self.assertEqual(a['rows'],b['rows'])

    def fixture(self):
        rng=np.random.default_rng(5);aa=np.array(list('ARNDCEQGHILKMFPSTWYV'))
        def mono(prefix):
            return {'query':''.join(rng.choice(aa,12)), 'mask':'***-********',
                    'rows':[(f'genome{i}',''.join(rng.choice(aa,12)),f'g:f{i%3}:o:c:p') for i in range(20)]}
        return mono('a'),mono('b')

    def test_pairing_keys_masks_symmetry_and_null_marginals(self):
        a,b=self.fixture();tokens,meta=pair(a,b)
        self.assertEqual(meta['reason'],'available');self.assertEqual(tokens.shape[1],24)
        self.assertTrue((tokens[:,[3,15]]==26).all())
        swapped,sm=pair(b,a)
        np.testing.assert_array_equal(tokens,np.c_[swapped[:,12:],swapped[:,:12]])
        np.testing.assert_array_equal(quality_features(a,b,meta),quality_features(b,a,sm))
        null,n=shuffled(tokens,12,meta['taxonomy'],7)
        np.testing.assert_array_equal(null[0],tokens[0]);np.testing.assert_array_equal(null[:,:12],tokens[:,:12])
        np.testing.assert_array_equal(np.sort(null[1:,12:],axis=0),np.sort(tokens[1:,12:],axis=0))
        self.assertGreater(n['changed_fraction'],.9)

    def test_reused_homologs_and_wrong_keys_never_create_evidence(self):
        a,b=self.fixture();b['rows']=a['rows']
        tokens,meta=pair(a,b);self.assertIsNone(tokens);self.assertEqual(meta['reused_homolog_rows'],20)
        a,b=self.fixture();b['rows']=[('other'+k,s,t) for k,s,t in b['rows']]
        tokens,meta=pair(a,b);self.assertIsNone(tokens);self.assertEqual(meta['common_keys'],0)

    def test_homologs_must_cover_retained_quality_columns(self):
        a,b=self.fixture();a['mask']=b['mask']='******------'
        for m in [a,b]:m['rows']=[(k,'------'+s[6:],t) for k,s,t in m['rows']]
        tokens,meta=pair(a,b)
        self.assertIsNone(tokens);self.assertEqual(meta['low_good_column_coverage_rows'],20)

    def test_null_uses_order_for_unique_families_and_preserves_unknowns(self):
        tokens=np.arange(5*8,dtype=np.uint8).reshape(5,8)
        tax=['g:f1:o:c:p','g:f2:o:c:p','g:f3:other:c:p','na:na:na:na:na']
        out,meta=shuffled(tokens,4,tax,7)
        self.assertEqual(meta['level_counts'],{'family':0,'order':2,'class':0,'unchanged':2})
        np.testing.assert_array_equal(out[0],tokens[0]);np.testing.assert_array_equal(out[4],tokens[4])
        self.assertEqual(meta['changed_fraction'],.5)

    def test_bad_coordinates_rejected(self):
        with self.assertRaises(ValueError):cached_monomer(iter([('x g:f','ACD')]),'ACDE','****',set())
        with self.assertRaises(ValueError):cached_monomer(iter([]),'ACDE','***',set())


if __name__=='__main__':unittest.main()
