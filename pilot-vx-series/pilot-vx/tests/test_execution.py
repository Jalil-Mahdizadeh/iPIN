"""Corruption refusal, atomic resume, and exact missing-evidence fallback."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from analyze import fuse
from features import load_record, save_record


class ExecutionInvariants(unittest.TestCase):
    def test_exact_fallback_even_with_missing_nan(self):
        base=np.array([-1.,.2,3.]);gate=np.array([0.,.5,0.]);evidence=np.array([np.nan,2.,np.nan])
        result=fuse(base,evidence,gate,.25)
        np.testing.assert_array_equal(result[[0,2]],base[[0,2]])
        self.assertEqual(result[1],.45)
        np.testing.assert_array_equal(fuse(base,evidence,gate,0),base)
        with self.assertRaises(ValueError):fuse(base,[0,np.nan,0],gate,1)

    def test_atomic_resume_and_corruption_refusal(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);value={'true':[0.]*128,'shuffled':[0.]*128,'quality':[0.]*68}
            self.assertIsNone(load_record(p,'train-000001','frozen'))
            save_record(p,'train-000001',value,'frozen')
            self.assertEqual(load_record(p,'train-000001','frozen')['true'],value['true'])
            with self.assertRaises(ValueError):load_record(p,'train-000001','changed')
            (p/'train-000001.json').write_text('{}')
            with self.assertRaises(ValueError):load_record(p,'train-000001','frozen')

    def test_interrupted_commit_is_not_a_valid_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);(p/'train-000001.json').write_text('{}')
            self.assertIsNone(load_record(p,'train-000001','frozen'))


if __name__=='__main__':unittest.main()
