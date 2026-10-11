import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from runtime import topology


class Runtime(unittest.TestCase):
    def records(self):
        return [{'rank': i, 'world': 8, 'local_rank': i % 4, 'host': f'n{i // 4}',
            'gpu_uuid': f'GPU-{i}', 'gpu_name': 'NVIDIA GH200 120GB', 'job_id': '123', 'fingerprint': 'f'} for i in range(8)]

    def test_two_nodes_four_distinct_gpus_each(self):
        topology(self.records(), '123', 'f')

    def test_oversubscription_and_partial_start_rejected(self):
        records = self.records(); records[1]['gpu_uuid'] = records[0]['gpu_uuid']
        with self.assertRaises(ValueError):
            topology(records, '123', 'f')
        with self.assertRaises(ValueError):
            topology(self.records()[:4], '123', 'f')

    def test_wrong_node_distribution_or_stale_job_rejected(self):
        records = self.records(); records[7]['host'] = 'n0'
        with self.assertRaises(ValueError):
            topology(records, '123', 'f')
        with self.assertRaises(ValueError):
            topology(self.records(), '456', 'f')


if __name__ == '__main__':
    unittest.main()
