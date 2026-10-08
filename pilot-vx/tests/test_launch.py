"""The interactive allocation must not constrain a new four-GPU batch job."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from launch import submission_environment


class LaunchInvariants(unittest.TestCase):
    def test_preserve_site_configuration_remove_parent_resources(self):
        env={'PATH':'/bin','SLURM_CONF':'/site/slurm.conf','SLURM_STEP_ID':'interactive',
             'SLURM_GPUS_PER_NODE':'1','CUDA_VISIBLE_DEVICES':'0','SBATCH_TIME':'12:00:00','SRUN_DEBUG':'2'}
        with patch.dict(os.environ,env,clear=True):clean=submission_environment()
        self.assertEqual(clean,{'PATH':'/bin','SLURM_CONF':'/site/slurm.conf'})


if __name__=='__main__':unittest.main()
